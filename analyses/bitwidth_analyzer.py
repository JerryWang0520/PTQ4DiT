import os
import csv
import torch
import numpy as np
from typing import Dict, Optional, Any
from collections import defaultdict
from .base import TensorAnalyzer


class BitwidthAnalyzer(TensorAnalyzer):
    def __init__(self, signed: bool = True, max_bits: int = 10):
        self.signed = signed
        self.max_bits = max_bits
        self._bit_ranges = self._create_bit_ranges()
        # Pre-compute int9 ranges for efficiency
        self._int9_neg_range = torch.arange(-256, -128, dtype=torch.int32)
        self._int9_pos_range = torch.arange(128, 256, dtype=torch.int32)
    
    def _create_bit_ranges(self):
        if self.signed:
            return [(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) for bits in range(1, self.max_bits + 1)]
        else:
            return [(0, 2 ** bits - 1) for bits in range(1, self.max_bits + 1)]
    
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        values = tensor.detach().flatten()
        total_elements = values.numel()
        
        if total_elements == 0:
            return {
                "zero": 0, "total": 0, "required_bitwidth": 1, "overflow": 0,
                "int9_neg_distribution": {}, "int9_pos_distribution": {}
            }
        
        zero_count = (values == 0).sum().item()
        
        if self.signed:
            min_val, max_val = values.min().item(), values.max().item()
            min_val_bw = int(np.ceil(np.log2(abs(min_val)))) + 1 if min_val != 0 else 1
            max_val_bw = int(np.ceil(np.log2(abs(max_val) + 1))) + 1 if max_val != 0 else 1
            required_bw = max(min_val_bw, max_val_bw)
        else:
            max_val = values.max().item()
            required_bw = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
        
        counts = {"zero": zero_count, "total": total_elements, "required_bitwidth": required_bw}
        
        int9_distribution = self._get_value_distribution(values, -256, 256)
        counts["int9_distribution"] = int9_distribution
        
        int9_neg_distribution = {k: v for k, v in int9_distribution.items() 
                               if int(k) >= -256 and int(k) < -128 and v > 0}
        int9_pos_distribution = {k: v for k, v in int9_distribution.items() 
                               if int(k) > 127 and int(k) <= 255 and v > 0}
        
        counts["int9_neg_distribution"] = int9_neg_distribution
        counts["int9_pos_distribution"] = int9_pos_distribution
        counts["int9_neg_total"] = sum(int9_neg_distribution.values())
        counts["int9_pos_total"] = sum(int9_pos_distribution.values())
        
        for i, (min_val, max_val) in enumerate(self._bit_ranges):
            mask = (values >= min_val) & (values <= max_val)
            counts[str(i + 1)] = mask.sum().item()
        
        min_range_val, max_range_val = self._bit_ranges[-1]
        overflow_mask = (values < min_range_val) | (values > max_range_val)
        counts["overflow"] = overflow_mask.sum().item()
        
        for bits in range(self.max_bits, 1, -1):
            counts[str(bits)] -= counts[str(bits-1)]
        counts["1"] -= counts["zero"]
        
        return counts
    
    def _get_value_distribution(self, values: torch.Tensor, min_val: int, max_val: int) -> Dict[str, int]:
        range_mask = (values >= min_val) & (values < max_val)
        range_values = values[range_mask]
        
        if range_values.numel() == 0:
            return {}
        
        # Use bincount for efficient counting
        # Shift values to start from 0 for bincount
        shifted_values = (range_values - min_val).long()
        max_shift = max_val - min_val
        
        counts_tensor = torch.bincount(shifted_values, minlength=max_shift)
        
        distribution = {}
        nonzero_indices = torch.nonzero(counts_tensor, as_tuple=True)[0]
        
        for idx in nonzero_indices:
            original_val = idx.item() + min_val
            count = counts_tensor[idx].item()
            distribution[str(original_val)] = count
        
        return distribution
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        self._save_main_results(results, output_dir)
        self._save_int9_full_distribution(results, output_dir)
    
    def _save_main_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save main bitwidth analysis results"""
        by_tensor_type = defaultdict(list)
        
        for module_name, module_data in results.items():
            for step, step_data in module_data.items():
                for tensor_type, analysis_results in step_data.items():
                    if 'bitwidth' in analysis_results:
                        # Extract main results (exclude distributions)
                        main_results = {k: v for k, v in analysis_results['bitwidth'].items() 
                                      if not k.endswith('_distribution')}
                        
                        row_data = {'module': module_name, 'step': step, **main_results}
                        by_tensor_type[tensor_type].append(row_data)
        
        for tensor_type, data in by_tensor_type.items():
            if data:
                csv_path = os.path.join(output_dir, f'bitwidth_{tensor_type}.csv')
                fieldnames = data[0].keys()
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
    
    def _save_int9_full_distribution(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save complete int9 distribution (-256 to 255) by tensor type"""
        by_tensor_type = defaultdict(list)
        
        for module_name, module_data in results.items():
            for step, step_data in module_data.items():
                for tensor_type, analysis_results in step_data.items():
                    if 'bitwidth' in analysis_results:
                        bitwidth_data = analysis_results['bitwidth']
                        
                        row_data = {
                            'module': module_name,
                            'step': step
                        }
                        
                        int9_dist = bitwidth_data.get('int9_distribution', {})
                        for val in range(-256, 256):
                            row_data[str(val)] = int9_dist.get(str(val), 0)
                        
                        by_tensor_type[tensor_type].append(row_data)
        
        for tensor_type, data in by_tensor_type.items():
            if data:
                csv_path = os.path.join(output_dir, f'int9_full_distribution_{tensor_type}.csv')
                
                fieldnames = ['module', 'step'] + [str(i) for i in range(-256, 256)]
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
                    writer.writeheader()
                    writer.writerows(data)