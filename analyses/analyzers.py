import os
import csv
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any
from .base import TensorAnalyzer


# import os
# import csv
# import torch
# import numpy as np
# from typing import Dict, Optional, Any
# from collections import defaultdict
# from .base import TensorAnalyzer


# import os
# import csv
# import torch
# import numpy as np
# from typing import Dict, Optional, Any
# from collections import defaultdict
# from .base import TensorAnalyzer


# class BitwidthAnalyzer(TensorAnalyzer):
#     def __init__(self, signed: bool = True, max_bits: int = 10):
#         self.signed = signed
#         self.max_bits = max_bits
#         self._bit_ranges = self._create_bit_ranges()
#         # Pre-compute int9 ranges for efficiency
#         self._int9_neg_range = torch.arange(-256, -128, dtype=torch.int32)
#         self._int9_pos_range = torch.arange(128, 256, dtype=torch.int32)
    
#     def _create_bit_ranges(self):
#         """Pre-compute bit ranges once during initialization"""
#         if self.signed:
#             return [(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) for bits in range(1, self.max_bits + 1)]
#         else:
#             return [(0, 2 ** bits - 1) for bits in range(1, self.max_bits + 1)]
    
#     def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
#         """Optimized GPU-based analysis"""
#         values = tensor.detach().flatten()
#         total_elements = values.numel()
        
#         if total_elements == 0:
#             return {
#                 "zero": 0, "total": 0, "required_bitwidth": 1, "overflow": 0,
#                 "int9_neg_distribution": {}, "int9_pos_distribution": {}
#             }
        
#         # Basic counts
#         zero_count = (values == 0).sum().item()
        
#         # Required bitwidth calculation
#         if self.signed:
#             min_val, max_val = values.min().item(), values.max().item()
#             min_val_bw = int(np.ceil(np.log2(abs(min_val)))) + 1 if min_val != 0 else 1
#             max_val_bw = int(np.ceil(np.log2(abs(max_val) + 1))) + 1 if max_val != 0 else 1
#             required_bw = max(min_val_bw, max_val_bw)
#         else:
#             max_val = values.max().item()
#             required_bw = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
        
#         counts = {"zero": zero_count, "total": total_elements, "required_bitwidth": required_bw}
        
#         # Optimized int9 distribution using bincount
#         int9_neg_distribution = self._get_value_distribution(values, -256, -128)
#         int9_pos_distribution = self._get_value_distribution(values, 128, 256)
        
#         counts["int9_neg_distribution"] = int9_neg_distribution
#         counts["int9_pos_distribution"] = int9_pos_distribution
#         counts["int9_neg_total"] = sum(int9_neg_distribution.values())
#         counts["int9_pos_total"] = sum(int9_pos_distribution.values())
        
#         # Vectorized bitwidth counting
#         for i, (min_val, max_val) in enumerate(self._bit_ranges):
#             mask = (values >= min_val) & (values <= max_val)
#             counts[str(i + 1)] = mask.sum().item()
        
#         # Overflow count
#         min_range_val, max_range_val = self._bit_ranges[-1]
#         overflow_mask = (values < min_range_val) | (values > max_range_val)
#         counts["overflow"] = overflow_mask.sum().item()
        
#         # Convert cumulative to differential counts
#         for bits in range(self.max_bits, 1, -1):
#             counts[str(bits)] -= counts[str(bits-1)]
#         counts["1"] -= counts["zero"]
        
#         return counts
    
#     def _get_value_distribution(self, values: torch.Tensor, min_val: int, max_val: int) -> Dict[str, int]:
#         """Optimized value distribution using GPU operations"""
#         # Filter values in range
#         range_mask = (values >= min_val) & (values < max_val)
#         range_values = values[range_mask]
        
#         if range_values.numel() == 0:
#             return {}
        
#         # Use bincount for efficient counting
#         # Shift values to start from 0 for bincount
#         shifted_values = (range_values - min_val).long()
#         max_shift = max_val - min_val
        
#         # Get counts using bincount
#         counts_tensor = torch.bincount(shifted_values, minlength=max_shift)
        
#         # Convert back to original values and filter non-zero counts
#         distribution = {}
#         nonzero_indices = torch.nonzero(counts_tensor, as_tuple=True)[0]
        
#         for idx in nonzero_indices:
#             original_val = idx.item() + min_val
#             count = counts_tensor[idx].item()
#             distribution[str(original_val)] = count
        
#         return distribution
    
#     def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save main bitwidth analysis
#         self._save_main_results(results, output_dir)
        
#         # Save int9 distribution by tensor type
#         self._save_int9_distribution_by_type(results, output_dir)
    
#     def _save_main_results(self, results: Dict[str, Any], output_dir: str) -> None:
#         """Save main bitwidth analysis results"""
#         by_tensor_type = defaultdict(list)
        
#         for module_name, module_data in results.items():
#             for step, step_data in module_data.items():
#                 for tensor_type, analysis_results in step_data.items():
#                     if 'bitwidth' in analysis_results:
#                         # Extract main results (exclude distributions)
#                         main_results = {k: v for k, v in analysis_results['bitwidth'].items() 
#                                       if not k.endswith('_distribution')}
                        
#                         row_data = {'module': module_name, 'step': step, **main_results}
#                         by_tensor_type[tensor_type].append(row_data)
        
#         # Write main results by tensor type
#         for tensor_type, data in by_tensor_type.items():
#             if data:
#                 csv_path = os.path.join(output_dir, f'bitwidth_{tensor_type}.csv')
#                 fieldnames = data[0].keys()
#                 with open(csv_path, 'w', newline='') as f:
#                     writer = csv.DictWriter(f, fieldnames=fieldnames)
#                     writer.writeheader()
#                     writer.writerows(data)
    
#     def _save_int9_distribution_by_type(self, results: Dict[str, Any], output_dir: str) -> None:
#         """Save int9 distribution results separated by tensor type with values as columns"""
#         by_tensor_type = defaultdict(list)
        
#         for module_name, module_data in results.items():
#             for step, step_data in module_data.items():
#                 for tensor_type, analysis_results in step_data.items():
#                     if 'bitwidth' in analysis_results:
#                         bitwidth_data = analysis_results['bitwidth']
                        
#                         # Create single row for this module/step
#                         row_data = {
#                             'module': module_name,
#                             'step': step
#                         }
                        
#                         # Add negative distribution as columns (neg_-256, neg_-255, etc.)
#                         for value, count in bitwidth_data.get('int9_neg_distribution', {}).items():
#                             row_data[f'neg_{value}'] = count
                        
#                         # Add positive distribution as columns (pos_128, pos_129, etc.)
#                         for value, count in bitwidth_data.get('int9_pos_distribution', {}).items():
#                             row_data[f'pos_{value}'] = count
                        
#                         by_tensor_type[tensor_type].append(row_data)
        
#         # Write distribution results by tensor type
#         for tensor_type, data in by_tensor_type.items():
#             if data:
#                 csv_path = os.path.join(output_dir, f'int9_distribution_{tensor_type}.csv')
                
#                 # Get all possible column names from all rows
#                 all_columns = set(['module', 'step'])
#                 for row in data:
#                     all_columns.update(row.keys())
                
#                 # Sort columns: module, step, then neg values, then pos values
#                 neg_cols = sorted([col for col in all_columns if col.startswith('neg_')], 
#                                 key=lambda x: int(x.split('_')[1]))
#                 pos_cols = sorted([col for col in all_columns if col.startswith('pos_')], 
#                                 key=lambda x: int(x.split('_')[1]))
#                 fieldnames = ['module', 'step'] + neg_cols + pos_cols
                
#                 with open(csv_path, 'w', newline='') as f:
#                     writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
#                     writer.writeheader()
#                     writer.writerows(data)


# class BitwidthAnalyzer(TensorAnalyzer):
#     def __init__(self, signed: bool = True, max_bits: int = 10, use_gpu: bool = True):
#         self.signed = signed
#         self.max_bits = max_bits
#         self.use_gpu = use_gpu
#         # Pre-compute bit ranges
#         self._bit_ranges = self._create_bit_ranges()
    
#     def _create_bit_ranges(self):
#         """Pre-compute bit ranges once during initialization"""
#         if self.signed:
#             return [(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) for bits in range(1, self.max_bits + 1)]
#         else:
#             return [(0, 2 ** bits - 1) for bits in range(1, self.max_bits + 1)]
    
#     def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, int]:
#         if self.use_gpu and tensor.is_cuda:
#             return self._analyze_gpu(tensor)
#         else:
#             return self._analyze_cpu(tensor)
    
#     def _analyze_gpu(self, tensor: torch.Tensor) -> Dict[str, int]:
#         """GPU-based analysis using PyTorch operations"""
#         values = tensor.detach().flatten()
#         total_elements = values.numel()
        
#         if total_elements == 0:
#             return {"zero": 0, "total": 0, "required_bitwidth": 1, "overflow": 0}
        
#         # Fast zero count on GPU
#         zero_count = (values == 0).sum().item()
        
#         # Required bitwidth calculation on GPU
#         if self.signed:
#             min_val = values.min().item()
#             max_val = values.max().item()
#             min_val_bw = int(np.ceil(np.log2(abs(min_val)))) + 1 if min_val != 0 else 1
#             max_val_bw = int(np.ceil(np.log2(abs(max_val) + 1))) + 1 if max_val != 0 else 1
#             required_bw = max(min_val_bw, max_val_bw)
#         else:
#             max_val = values.max().item()
#             required_bw = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
        
#         counts = {"zero": zero_count, "total": total_elements, "required_bitwidth": required_bw}
        
#         # Vectorized bitwidth counting on GPU
#         for i, (min_val, max_val) in enumerate(self._bit_ranges):
#             mask = (values >= min_val) & (values <= max_val)
#             counts[str(i + 1)] = mask.sum().item()
        
#         # Overflow count
#         min_range_val, max_range_val = self._bit_ranges[-1]
#         overflow_mask = (values < min_range_val) | (values > max_range_val)
#         counts["overflow"] = overflow_mask.sum().item()
        
#         # Convert cumulative to differential counts
#         for bits in range(self.max_bits, 1, -1):
#             counts[str(bits)] -= counts[str(bits-1)]
#         counts["1"] -= counts["zero"]
        
#         return counts
    
#     def _analyze_cpu(self, tensor: torch.Tensor) -> Dict[str, int]:
#         """CPU-based analysis using NumPy operations"""
#         values = tensor.detach().cpu().flatten().numpy()
#         total_elements = len(values)
        
#         if total_elements == 0:
#             return {"zero": 0, "total": 0, "required_bitwidth": 1, "overflow": 0}
        
#         # Fast zero count
#         zero_count = np.sum(values == 0)
        
#         # Required bitwidth calculation
#         if self.signed:
#             min_val, max_val = np.min(values), np.max(values)
#             min_val_bw = int(np.ceil(np.log2(abs(min_val)))) + 1 if min_val != 0 else 1
#             max_val_bw = int(np.ceil(np.log2(abs(max_val) + 1))) + 1 if max_val != 0 else 1
#             required_bw = max(min_val_bw, max_val_bw)
#         else:
#             max_val = np.max(values)
#             required_bw = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
        
#         counts = {"zero": int(zero_count), "total": total_elements, "required_bitwidth": required_bw}
        
#         # Vectorized bitwidth counting
#         for i, (min_val, max_val) in enumerate(self._bit_ranges):
#             mask = (values >= min_val) & (values <= max_val)
#             counts[str(i + 1)] = int(np.sum(mask))
        
#         # Overflow count
#         min_range_val, max_range_val = self._bit_ranges[-1]
#         overflow_mask = (values < min_range_val) | (values > max_range_val)
#         counts["overflow"] = int(np.sum(overflow_mask))
        
#         # Convert cumulative to differential counts
#         for bits in range(self.max_bits, 1, -1):
#             counts[str(bits)] -= counts[str(bits-1)]
#         counts["1"] -= counts["zero"]
        
#         return counts
    
#     def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Group by tensor type for batch writing
#         by_tensor_type = {}
#         for module_name, module_data in results.items():
#             for step, step_data in module_data.items():
#                 for tensor_type, analysis_results in step_data.items():
#                     if 'bitwidth' in analysis_results:
#                         if tensor_type not in by_tensor_type:
#                             by_tensor_type[tensor_type] = []
                        
#                         row_data = {
#                             'module': module_name,
#                             'step': step,
#                             **analysis_results['bitwidth']
#                         }
#                         by_tensor_type[tensor_type].append(row_data)
        
#         # Batch write each tensor type
#         for tensor_type, data in by_tensor_type.items():
#             if data:
#                 csv_path = os.path.join(output_dir, f'bitwidth_{tensor_type}.csv')
#                 fieldnames = data[0].keys()
#                 with open(csv_path, 'w', newline='') as f:
#                     writer = csv.DictWriter(f, fieldnames=fieldnames)
#                     writer.writeheader()
#                     writer.writerows(data)


# class BitwidthAnalyzer(TensorAnalyzer):
#     def __init__(self, signed: bool = True, max_bits: int = 10):
#         self.signed = signed
#         self.max_bits = max_bits
#         # Pre-compute bit ranges to avoid recreation
#         self._bit_ranges = self._create_bit_ranges()
    
#     def _create_bit_ranges(self):
#         """Pre-compute bit ranges once during initialization"""
#         if self.signed:
#             return [(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) for bits in range(1, self.max_bits + 1)]
#         else:
#             return [(0, 2 ** bits - 1) for bits in range(1, self.max_bits + 1)]
    
#     def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, int]:
#         values = tensor.detach().cpu().flatten().numpy()
#         total_elements = len(values)
        
#         # Fast zero count
#         zero_mask = (values == 0)
#         zero_count = np.sum(zero_mask)
        
#         # Fast min/max for required bitwidth
#         if total_elements > 0:
#             min_val, max_val = np.min(values), np.max(values)
#             if self.signed:
#                 min_val_bw = int(np.ceil(np.log2(abs(min_val)))) + 1 if min_val != 0 else 1
#                 max_val_bw = int(np.ceil(np.log2(abs(max_val) + 1))) + 1 if max_val != 0 else 1
#                 required_bw = max(min_val_bw, max_val_bw)
#             else:
#                 required_bw = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
#         else:
#             required_bw = 1
        
#         # Vectorized bitwidth counting
#         counts = {"zero": int(zero_count), "total": total_elements, "required_bitwidth": required_bw}
        
#         # Create cumulative masks for all bit ranges at once
#         cumulative_counts = np.zeros(self.max_bits + 1, dtype=int)
        
#         for i, (min_val, max_val) in enumerate(self._bit_ranges):
#             mask = (values >= min_val) & (values <= max_val)
#             cumulative_counts[i] = np.sum(mask)
#             counts[str(i + 1)] = int(cumulative_counts[i])
        
#         # Overflow count (values outside max_bits range)
#         min_range_val, max_range_val = self._bit_ranges[-1]
#         overflow_mask = (values < min_range_val) | (values > max_range_val)
#         counts["overflow"] = int(np.sum(overflow_mask))
        
#         # Convert cumulative to differential counts in reverse order
#         for bits in range(self.max_bits, 1, -1):
#             counts[str(bits)] -= counts[str(bits-1)]
#         counts["1"] -= counts["zero"]
        
#         return counts
    
#     def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Group by tensor type for batch writing
#         by_tensor_type = {}
#         for module_name, module_data in results.items():
#             for step, step_data in module_data.items():
#                 for tensor_type, analysis_results in step_data.items():
#                     if 'bitwidth' in analysis_results:
#                         if tensor_type not in by_tensor_type:
#                             by_tensor_type[tensor_type] = []
                        
#                         row_data = {
#                             'module': module_name,
#                             'step': step,
#                             **analysis_results['bitwidth']
#                         }
#                         by_tensor_type[tensor_type].append(row_data)
        
#         # Batch write each tensor type
#         for tensor_type, data in by_tensor_type.items():
#             if data:
#                 csv_path = os.path.join(output_dir, f'bitwidth_{tensor_type}.csv')
#                 fieldnames = data[0].keys()
#                 with open(csv_path, 'w', newline='') as f:
#                     writer = csv.DictWriter(f, fieldnames=fieldnames)
#                     writer.writeheader()
#                     writer.writerows(data)


# class BitwidthAnalyzer(TensorAnalyzer):
#     def __init__(self, signed: bool = True, max_bits: int = 10):
#         self.signed = signed
#         self.max_bits = max_bits
    
#     def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, int]:
#         values = tensor.detach().cpu().flatten().numpy()
#         counts = {}
#         bit_ranges = []

#         if self.signed:
#             for bits in range(1, self.max_bits + 1):
#                 min_val = -(2 ** (bits - 1))
#                 max_val = 2 ** (bits - 1) - 1
#                 bit_ranges.append((bits, min_val, max_val))
#         else:
#             for bits in range(1, self.max_bits + 1):
#                 max_val = 2 ** bits - 1
#                 bit_ranges.append((bits, 0, max_val))
        
#         # 1. Count zeros
#         zero_count = np.sum(values == 0)
#         if zero_count >= 0:
#             counts["zero"] = int(zero_count)
        
#         # 2. Count values within max_bits
#         for bits, min_val, max_val in bit_ranges:
#             count = np.sum((values >= min_val) & (values <= max_val))
#             if count >= 0:
#                 counts[str(bits)] = int(count)
        
#         # 3. Count overflows
#         _, min_range_val, max_range_val = bit_ranges[-1]
#         overflow_count = np.sum((values < min_range_val) | (values > max_range_val))
#         if overflow_count >= 0:
#             counts["overflow"] = int(overflow_count)
        
#         # 4. Count total
#         counts["total"] = len(values)

#         # 5. Count required bitwidth
#         if self.signed:
#             min_val = np.min(values)
#             max_val = np.max(values)
#             min_val_bw = int(np.ceil(np.log2(abs(min_val)  ))) + 1 if min_val != 0 else 1
#             max_val_bw = int(np.ceil(np.log2(abs(max_val)+1))) + 1 if max_val != 0 else 1
#             required_bw = max(min_val_bw, max_val_bw)
#         else:
#             max_val = np.max(values)
#             required_bw = int(np.ceil(np.log2(max_val+1)))
#         counts["required_bitwidth"] = required_bw
        
#         # Reconstruct counts (subtract lower bitwidths)
#         for bits in range(self.max_bits, 1, -1):
#             counts[str(bits)] -= counts[str(bits-1)]
#         counts["1"] -= counts["zero"]
        
#         return counts
    
#     def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Reorganize results by tensor type
#         by_tensor_type = {}
#         for module_name, module_data in results.items():
#             for step, step_data in module_data.items():
#                 for tensor_type, analysis_results in step_data.items():
#                     if 'bitwidth' in analysis_results:
#                         if tensor_type not in by_tensor_type:
#                             by_tensor_type[tensor_type] = []
#                         row_data = {
#                             'module': module_name,
#                             'step': step,
#                             **analysis_results['bitwidth']
#                         }
#                         by_tensor_type[tensor_type].append(row_data)
        
#         # Save each tensor type to a separate CSV
#         for tensor_type, data in by_tensor_type.items():
#             csv_path = os.path.join(output_dir, f'bitwidth_{tensor_type}.csv')
#             if data:
#                 fieldnames = data[0].keys()
#                 with open(csv_path, 'w', newline='') as f:
#                     writer = csv.DictWriter(f, fieldnames=fieldnames)
#                     writer.writeheader()
#                     writer.writerows(data)


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