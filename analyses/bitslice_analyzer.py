import torch
import numpy as np
import csv
import os
from typing import Dict, Any, Optional
from collections import defaultdict
from .base import TensorAnalyzer


class BitsliceAnalyzer(TensorAnalyzer):
    def __init__(self, strategy, scheme=2, performance=False):
        self.strategy = strategy
        self.GLOBAL_TILE_SIZE = (128, 256, 256)
        self.LOCAL_TILE_SIZE  = (8, 32, 16)
        self.scheme = scheme
        self.performance = performance
    
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


    def analyze_performance(self, tensor, module_info=None, output_dir=None):
        self.strategy._move_to_device(tensor.device)
        
        module_name = module_info.get('name', 'Unknown') if module_info else 'Unknown'
        step = module_info.get('step', 'Unknown') if module_info else 'Unknown'
        
        if tensor.dim() != 3:
            raise ValueError(
                f"Tensor must be 3D for tiling analysis.\n"
                f"Module: {module_name}\n"
                f"Step: {step}\n"
                f"Got tensor shape: {tensor.shape} (dim={tensor.dim()})"
            )
        
        values = tensor.detach().int()
        batch0, batch1 = values[0], values[1]
        M, K = batch0.shape
        
        indices0 = torch.clamp(batch0 + self.strategy.offset, 0, len(self.strategy.gpu_nonzero_slices) - 1)
        indices1 = torch.clamp(batch1 + self.strategy.offset, 0, len(self.strategy.gpu_nonzero_slices) - 1)
        
        slice_counts0 = self.strategy.gpu_nonzero_slices[indices0]
        slice_counts1 = self.strategy.gpu_nonzero_slices[indices1]

        onehot_pairs = defaultdict(int)
        global_tile_latencies = []
        
        for m_g in range(0, M, self.GLOBAL_TILE_SIZE[0]):
            for k_g in range(0, K, self.GLOBAL_TILE_SIZE[1]):
                m_end = min(m_g + self.GLOBAL_TILE_SIZE[0], M)
                k_end = min(k_g + self.GLOBAL_TILE_SIZE[1], K)
                
                gt_counts0 = slice_counts0[m_g:m_end, k_g:k_end]
                gt_counts1 = slice_counts1[m_g:m_end, k_g:k_end]
                gt_indice0 = indices0[m_g:m_end, k_g:k_end]
                gt_indice1 = indices1[m_g:m_end, k_g:k_end]

                if gt_counts0.shape[0] != self.GLOBAL_TILE_SIZE[0] or gt_counts0.shape[1] != self.GLOBAL_TILE_SIZE[1]:
                    pad_h = self.GLOBAL_TILE_SIZE[0] - gt_counts0.shape[0]
                    pad_w = self.GLOBAL_TILE_SIZE[1] - gt_counts0.shape[1]
                    gt_counts0 = torch.nn.functional.pad(gt_counts0, (0, pad_w, 0, pad_h), value=1)
                    gt_counts1 = torch.nn.functional.pad(gt_counts1, (0, pad_w, 0, pad_h), value=1)
                    gt_indice0 = torch.nn.functional.pad(gt_indice0, (0, pad_w, 0, pad_h))
                    gt_indice1 = torch.nn.functional.pad(gt_indice1, (0, pad_w, 0, pad_h))
                
                # Reshape (128, 256) -> (16, 8, 8, 32) -> (16, 8, 8, 32)
                num_m_tiles = self.GLOBAL_TILE_SIZE[0] // self.LOCAL_TILE_SIZE[0]
                num_k_tiles = self.GLOBAL_TILE_SIZE[1] // self.LOCAL_TILE_SIZE[1]
                num_n_tiles = self.GLOBAL_TILE_SIZE[2] // self.LOCAL_TILE_SIZE[2]
                
                gt_counts0 = gt_counts0.view(num_m_tiles, self.LOCAL_TILE_SIZE[0], num_k_tiles, self.LOCAL_TILE_SIZE[1]).permute(0, 2, 1, 3)
                gt_counts1 = gt_counts1.view(num_m_tiles, self.LOCAL_TILE_SIZE[0], num_k_tiles, self.LOCAL_TILE_SIZE[1]).permute(0, 2, 1, 3)

                # (16, 8, 8, 32) -> max per element -> sum across cols -> (16, 8, 8)
                if self.scheme == 1:    #* SCHEME1
                    row_latencies = torch.clamp(torch.max(gt_counts0, gt_counts1), min=1).sum(dim=3)            
                else:                   #* SCHEME2
                    row_latencies = torch.clamp(torch.ceil((gt_counts0 + gt_counts1) / 2.0), min=1).sum(dim=3)

                # Max across k_tiles -> (16, 8)
                row_latencies = row_latencies.max(dim=1)[0]
                
                # Reshape to pairs: (16, 4, 2) then max across pairs
                row_latencies = row_latencies.view(num_m_tiles, self.LOCAL_TILE_SIZE[0] // 2, 2)
                local_latencies = row_latencies.max(dim=2)[0].sum(dim=1)
                
                gt_latency = local_latencies.sum().item() * num_n_tiles
                global_tile_latencies.append(gt_latency)
                
                # Pair counting
                pairs = torch.stack([gt_indice0.flatten(), gt_indice1.flatten()], dim=1)
                pairs = torch.stack([
                    self.strategy.gpu_nonzero_slices[pairs[:, 0]], 
                    self.strategy.gpu_nonzero_slices[pairs[:, 1]]
                ], dim=1)
                unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)
                
                for pair, count in zip(unique_pairs.cpu().numpy(), counts.cpu().numpy()):
                    onehot_pairs[(int(pair[0]), int(pair[1]))] += int(count)
        
        compute_latency = sum(global_tile_latencies)

        num_global_tiles = len(global_tile_latencies)
        num_local_tiles_row_per_global = num_m_tiles * num_n_tiles
        avg_local_latency = compute_latency / (num_global_tiles * num_local_tiles_row_per_global) if num_global_tiles > 0 else 0

        if module_info and "layer_size" in module_info:
            N = module_info["layer_size"]["N"]
            N_tiles = (N + self.GLOBAL_TILE_SIZE[2] - 1) // self.GLOBAL_TILE_SIZE[2]  # Ceiling division for 256-width tiles
            compute_latency *= N_tiles
        
        avg_global_latency = compute_latency / (num_global_tiles * N_tiles)
        mem_transfer_latency = 3500 * num_global_tiles * N_tiles
        total_latency = mem_transfer_latency + compute_latency
        
        result = {
            "total_latency": compute_latency,
            # "total_latency": total_latency,
            # "mem_transfer_latency": mem_transfer_latency,
            # "compute_latency": compute_latency,
            # "avg_global_latency": avg_global_latency,
            "avg_local_latency": avg_local_latency,
            "onehot_pairs": dict(onehot_pairs),
            "global_tile_latencies": global_tile_latencies
        }
        
        if output_dir:
            self._save_single_step_performance(result, module_name, step, output_dir)
        
        return result


    def _save_single_step_performance(self, result, module_name, step, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine max number of slices
        num_slices = (self.strategy.bits + self.strategy.slice_width - 1) // self.strategy.slice_width
        if not self.strategy.discard_overflow:
            num_slices += 1
        
        all_pairs = [(i, j) for i in range(num_slices + 1) for j in range(num_slices + 1)]
        
        # Create row for this timestep
        row = {
            'module': module_name,
            'step': step,
            'tensor_type': 'act',
            'total_latency': result['total_latency'],
            # 'mem_transfer_latency': result['mem_transfer_latency'],
            # 'compute_latency': result['compute_latency'],
            # 'avg_global_latency': result['avg_global_latency'],
            'avg_local_latency': result['avg_local_latency'],
        }
        
        # Initialize all pairs with 0
        for pair in all_pairs:
            row[f'pair_{pair[0]}_{pair[1]}'] = 0
        
        # Fill in actual counts
        for pair, count in result['onehot_pairs'].items():
            row[f'pair_{pair[0]}_{pair[1]}'] = count
        
        # Prepare CSV path
        csv_path = os.path.join(output_dir, "performance_analysis.csv")
        
        # Write header if file doesn't exist, append otherwise
        file_exists = os.path.exists(csv_path)
        base_fields = ['module', 'step', 'tensor_type', 'total_latency', 'avg_local_latency']
        # base_fields = ['module', 'step', 'tensor_type', 'total_latency', 'mem_transfer_latency', 'compute_latency', 'avg_global_latency', 'avg_local_latency']
        pair_fields = [f'pair_{pair[0]}_{pair[1]}' for pair in all_pairs]
        fieldnames = base_fields + pair_fields
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        print(f"Performance for {module_name} step {step} saved to {csv_path}")


    def save_performance_result(self, results: Dict[str, Any], output_dir: str, filename: str = "performance_analysis.csv") -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine the max number of slices possible
        num_slices = (self.strategy.bits + self.strategy.slice_width - 1) // self.strategy.slice_width
        if not self.strategy.discard_overflow:
            num_slices += 1  # Include overflow slice
        
        # Generate all possible pair combinations
        all_pairs = [(i, j) for i in range(num_slices + 1) for j in range(num_slices + 1)]
        
        csv_data = []
        
        for module_name, module_data in results.items():
            for step, step_data in module_data.items():
                for tensor_type, analysis_results in step_data.items():
                    if 'performance' in analysis_results:
                        perf_data = analysis_results['performance']
                        
                        row = {
                            'module': module_name,
                            'step': step,
                            'tensor_type': tensor_type,
                            'total_latency': perf_data['total_latency'],
                            # 'mem_transfer_latency': perf_data['mem_transfer_latency'],
                            # 'compute_latency': perf_data['compute_latency'],
                            # 'avg_global_latency': perf_data['avg_global_latency'],
                            'avg_local_latency': perf_data['avg_local_latency'],
                        }
                        
                        # Initialize all pairs with 0
                        for pair in all_pairs:
                            row[f'pair_{pair[0]}_{pair[1]}'] = 0
                        
                        # Fill in actual counts
                        for pair, count in perf_data['onehot_pairs'].items():
                            row[f'pair_{pair[0]}_{pair[1]}'] = count
                        
                        csv_data.append(row)
        
        if csv_data:
            base_fields = ['module', 'step', 'tensor_type', 'total_latency', 'avg_local_latency']
            # base_fields = ['module', 'step', 'tensor_type', 'total_latency', 'mem_transfer_latency', 'compute_latency', 'avg_global_latency', 'avg_local_latency']
            pair_fields = [f'pair_{pair[0]}_{pair[1]}' for pair in all_pairs]
            fieldnames = base_fields + pair_fields
            
            csv_path = os.path.join(output_dir, filename)
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Performance analysis results saved to {csv_path}")
            print(f"Total pair combinations in header: {len(pair_fields)}")