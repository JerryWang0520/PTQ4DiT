import torch
import numpy as np
import csv
import os
from typing import Dict, Any, Optional
from .base import TensorAnalyzer


class BitsliceAnalyzer(TensorAnalyzer):
    def __init__(self, strategy):
        self.strategy = strategy
    
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        self.strategy._move_to_device(tensor.device)
        
        values = tensor.detach().flatten().int()
        indices = values + self.strategy.offset
        indices = torch.clamp(indices, 0, len(self.strategy.gpu_nonzero_slices) - 1)
        
        nonzero_slices = self.strategy.gpu_nonzero_slices[indices]
        has_overflow = self.strategy.gpu_has_overflow[indices]
        
        total_bitslices = nonzero_slices.sum().item()
        overflow_count = has_overflow.sum().item()
        
        unique_slices, counts = torch.unique(nonzero_slices, return_counts=True)
        
        # Only transfer the small unique arrays to CPU
        unique_slices_cpu = unique_slices.cpu()
        counts_cpu = counts.cpu()
        
        bitslice_distribution = {
            int(slices): int(count) 
            for slices, count in zip(unique_slices_cpu, counts_cpu)
        }
        
        return {
            'total_bitslice_amount': int(total_bitslices),
            'total_elements': int(values.numel()),
            'bitslice_distribution': bitslice_distribution,
            'overflow_count': int(overflow_count),
        }

    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        csv_data = []
        all_slice_counts = set()  # Track all possible slice counts
        
        # First pass: collect all data and find all slice counts
        for module_name, module_data in results.items():
            for step, step_data in module_data.items():
                for tensor_type, analysis_results in step_data.items():
                    if 'bitslice' in analysis_results:
                        data = analysis_results['bitslice']
                        
                        all_slice_counts.update(data['bitslice_distribution'].keys())
                        
                        row = {
                            'module': module_name,
                            'step': step,
                            'tensor_type': tensor_type,
                            'total_bitslice_amount': data['total_bitslice_amount'],
                            'total_elements': data['total_elements'],
                            'overflow_count': data['overflow_count'],
                        }
                        
                        for slice_count in sorted(all_slice_counts):
                            row[f'slices_{slice_count}_count'] = 0
                        
                        for slice_count, freq in data['bitslice_distribution'].items():
                            row[f'slices_{slice_count}_count'] = freq
                        
                        csv_data.append(row)
        
        # Second pass: ensure all rows have the same fields
        if csv_data:
            base_fields = ['module', 'step', 'tensor_type', 'total_bitslice_amount', 'total_elements', 'overflow_count']
            slice_fields = [f'slices_{slice_count}_count' for slice_count in sorted(all_slice_counts)]
            fieldnames = base_fields + slice_fields
            
            for row in csv_data:
                for field in fieldnames:
                    if field not in row:
                        row[field] = 0
            
            base_filename = 'bitslice_analysis'
            csv_path = os.path.join(output_dir, f'{base_filename}.csv')
            
            # # If file exists, add counter
            # counter = 1
            # while os.path.exists(csv_path):
            #     csv_path = os.path.join(output_dir, f'{base_filename}_{counter}.csv')
            #     counter += 1
                
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Bitslice analysis results saved to {csv_path}")