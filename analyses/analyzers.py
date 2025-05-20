import os
import csv
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any
from .base import TensorAnalyzer


class BitwidthAnalyzer(TensorAnalyzer):
    def __init__(self, signed: bool = True, max_bits: int = 10):
        self.signed = signed
        self.max_bits = max_bits
    
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, int]:
        values = tensor.detach().cpu().flatten().numpy()
        counts = {}
        bit_ranges = []

        if self.signed:
            for bits in range(1, self.max_bits + 1):
                min_val = -(2 ** (bits - 1))
                max_val = 2 ** (bits - 1) - 1
                bit_ranges.append((bits, min_val, max_val))
        else:
            for bits in range(1, self.max_bits + 1):
                max_val = 2 ** bits - 1
                bit_ranges.append((bits, 0, max_val))
        
        # 1. Count zeros
        zero_count = np.sum(values == 0)
        if zero_count >= 0:
            counts["zero"] = int(zero_count)
        
        # 2. Count values within max_bits
        for bits, min_val, max_val in bit_ranges:
            count = np.sum((values >= min_val) & (values <= max_val))
            if count >= 0:
                counts[str(bits)] = int(count)
        
        # 3. Count overflows
        _, min_range_val, max_range_val = bit_ranges[-1]
        overflow_count = np.sum((values < min_range_val) | (values > max_range_val))
        if overflow_count >= 0:
            counts["overflow"] = int(overflow_count)
        
        # 4. Count total
        counts["total"] = len(values)

        # 5. Count required bitwidth
        if self.signed:
            min_val = np.min(values)
            max_val = np.max(values)
            min_val_bw = int(np.ceil(np.log2(abs(min_val)  ))) + 1 if min_val != 0 else 1
            max_val_bw = int(np.ceil(np.log2(abs(max_val)+1))) + 1 if max_val != 0 else 1
            required_bw = max(min_val_bw, max_val_bw)
        else:
            max_val = np.max(values)
            required_bw = int(np.ceil(np.log2(max_val+1)))
        counts["required_bitwidth"] = required_bw
        
        # Reconstruct counts (subtract lower bitwidths)
        for bits in range(self.max_bits, 1, -1):
            counts[str(bits)] -= counts[str(bits-1)]
        counts["1"] -= counts["zero"]
        
        return counts
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Reorganize results by tensor type
        by_tensor_type = {}
        for module_name, module_data in results.items():
            for step, step_data in module_data.items():
                for tensor_type, analysis_results in step_data.items():
                    if 'bitwidth' in analysis_results:
                        if tensor_type not in by_tensor_type:
                            by_tensor_type[tensor_type] = []
                        row_data = {
                            'module': module_name,
                            'step': step,
                            **analysis_results['bitwidth']
                        }
                        by_tensor_type[tensor_type].append(row_data)
        
        # Save each tensor type to a separate CSV
        for tensor_type, data in by_tensor_type.items():
            csv_path = os.path.join(output_dir, f'bitwidth_{tensor_type}.csv')
            if data:
                fieldnames = data[0].keys()
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)


class StatisticsAnalyzer(TensorAnalyzer):
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, float]:
        values = tensor.detach().cpu().flatten()
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "sparsity": float((values == 0).sum() / values.numel()),
            "num_elements": int(values.numel())
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_data = {}
        for module_name, module_data in results.items():
            json_data[module_name] = {}
            for step, step_data in module_data.items():
                json_data[module_name][str(step)] = {}
                for tensor_type, analysis_results in step_data.items():
                    if 'statistics' in analysis_results:
                        json_data[module_name][str(step)][tensor_type] = analysis_results['statistics']
        
        json_path = os.path.join(output_dir, 'statistics.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)


class DistributionAnalyzer(TensorAnalyzer):
    def __init__(self, num_bins: int = 20):
        self.num_bins = num_bins
    
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        values = tensor.detach().cpu().flatten().numpy()
        hist, bins = np.histogram(values, bins=self.num_bins)
        
        return {
            "histogram": hist.tolist(),
            "bin_edges": bins.tolist(),
            "percentiles": {
                "25": float(np.percentile(values, 25)),
                "50": float(np.percentile(values, 50)),
                "75": float(np.percentile(values, 75)),
                "90": float(np.percentile(values, 90)),
                "99": float(np.percentile(values, 99))
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_data = {}
        for module_name, module_data in results.items():
            json_data[module_name] = {}
            for step, step_data in module_data.items():
                json_data[module_name][str(step)] = {}
                for tensor_type, analysis_results in step_data.items():
                    if 'distribution' in analysis_results:
                        json_data[module_name][str(step)][tensor_type] = analysis_results['distribution']
        
        json_path = os.path.join(output_dir, 'distribution.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)