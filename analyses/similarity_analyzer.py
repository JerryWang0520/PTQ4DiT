import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import TensorAnalyzer


class StatefulSimilarityAnalyzer(TensorAnalyzer):
    def __init__(self, active_analysis=None):
        super().__init__()
        self.prev_tensors = {}  # {module_name: {step: tensor}}
        self.active_analysis = active_analysis
        
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        if module_info is None:
            return {}
            
        module_name = module_info.get('name', '')
        step = module_info.get('step', 0)
        fwd_func = module_info.get('fwd_func', None)

        results = {}
        
        # 1. Spatial similarity
        if self.active_analysis == 'spatial':
            if fwd_func == F.conv2d:
                results['spatial'] = self._analyze_conv2d_similarity(tensor)
            elif fwd_func == F.linear:
                results['spatial'] = self._analyze_linear_similarity(tensor)
        
        # 2. Temporal similarity
        if self.active_analysis == 'temporal':
            if module_name not in self.prev_tensors:
                self.prev_tensors[module_name] = {}
            
            prev_step = step - 1
            if prev_step in self.prev_tensors.get(module_name, {}):
                prev_tensor = self.prev_tensors[module_name][prev_step]
                results['temporal'] = self._analyze_temporal_similarity(tensor, prev_tensor)
        
            self.prev_tensors[module_name][step] = tensor.clone()
        
        # 3. Conditional/unconditional similarity
        if self.active_analysis == 'conditional':
            if tensor.size(0) % 2 == 0:
                results['conditional'] = self._analyze_conditional_similarity(tensor)
                
        return results
    
    def _analyze_conv2d_similarity(self, tensor: torch.Tensor, ref_first: bool = False) -> Dict[str, float]:
        if tensor.dim() != 3:
            return {}
        
        similarities = []
        batch_size, seq_length, embedding_dim = tensor.shape
        
        # Analyze similarity between channels (columns)
        if ref_first:
            for batch_idx in range(batch_size):
                channel1 = tensor[batch_idx, :, 0]
                for i in range(1, embedding_dim):
                    channel2 = tensor[batch_idx, :, i]
                    
                    if channel1.norm() > 0 and channel2.norm() > 0:
                        sim = F.cosine_similarity(channel1.unsqueeze(0), channel2.unsqueeze(0)).item()
                        similarities.append(sim)
        else:
            for batch_idx in range(batch_size):
                for i in range(embedding_dim - 1):
                    channel1 = tensor[batch_idx, :, i]
                    channel2 = tensor[batch_idx, :, i + 1]
                    
                    if channel1.norm() > 0 and channel2.norm() > 0:
                        sim = F.cosine_similarity(channel1.unsqueeze(0), channel2.unsqueeze(0)).item()
                        similarities.append(sim)
        
        if not similarities:
            return {}
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities))
        }
    
    def _analyze_linear_similarity(self, tensor: torch.Tensor, ref_first: bool = False) -> Dict[str, float]:
        if tensor.dim() not in [2, 3]:
            return {}
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        similarities = []
        batch_size, seq_length, embedding_dim = tensor.shape
        
        # Analyze similarity between tokens (rows)
        if ref_first:
            for batch_idx in range(batch_size):
                token1 = tensor[batch_idx, 0, :]
                for i in range(1, seq_length):
                    token2 = tensor[batch_idx, i, :]
                    
                    if token1.norm() > 0 and token2.norm() > 0:
                        sim = F.cosine_similarity(token1.unsqueeze(0), token2.unsqueeze(0)).item()
                        similarities.append(sim)
        else:
            for batch_idx in range(batch_size):
                for i in range(seq_length - 1):
                    token1 = tensor[batch_idx, i, :]
                    token2 = tensor[batch_idx, i + 1, :]
                    
                    if token1.norm() > 0 and token2.norm() > 0:
                        sim = F.cosine_similarity(token1.unsqueeze(0), token2.unsqueeze(0)).item()
                        similarities.append(sim)
        
        if not similarities:
            return {}
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities))
        }
    
    def _analyze_temporal_similarity(self, current: torch.Tensor, previous: torch.Tensor) -> Dict[str, float]:
        if current.shape != previous.shape:
            return {}

        # Flatten tensors for overall similarity
        current_flat = current.flatten()
        previous_flat = previous.flatten()
        
        if current_flat.norm() == 0 or previous_flat.norm() == 0:
            return {'similarity': 0.0}
        
        similarity = F.cosine_similarity(current_flat.unsqueeze(0), previous_flat.unsqueeze(0)).item()
        
        return {'similarity': float(similarity)}
    
    def _analyze_conditional_similarity(self, tensor: torch.Tensor) -> Dict[str, float]:
        half_batch = tensor.size(0) // 2
        cond_tensor = tensor[:half_batch]
        uncond_tensor = tensor[half_batch:]
        
        # Calculate similarity between corresponding pairs
        similarities = []
        for i in range(half_batch):
            cond_flat = cond_tensor[i].flatten()
            uncond_flat = uncond_tensor[i].flatten()
            
            if cond_flat.norm() > 0 and uncond_flat.norm() > 0:
                sim = F.cosine_similarity(cond_flat.unsqueeze(0), uncond_flat.unsqueeze(0)).item()
                similarities.append(sim)
        
        if not similarities:
            return {}
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities))
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        import csv
        import os
        
        os.makedirs(output_path, exist_ok=True)
        
        spatial_data = []
        temporal_data = []
        conditional_data = []
        
        for module_name, module_data in results.items():
            for step, step_data in module_data.items():
                for tensor_type, analysis_results in step_data.items():
                    if 'similarity' in analysis_results:
                        sim_data = analysis_results['similarity']
                        
                        base_row = {
                            'module': module_name,
                            'step': step,
                            'tensor_type': tensor_type
                        }
                        
                        # 1. Add spatial similarity data
                        if 'spatial' in sim_data:
                            spatial_row = {**base_row, **sim_data['spatial']}
                            spatial_data.append(spatial_row)
                        
                        # 2. Add temporal similarity data
                        if 'temporal' in sim_data:
                            temporal_row = {**base_row, **sim_data['temporal']}
                            temporal_data.append(temporal_row)
                        
                        # 3. Add conditional similarity data
                        if 'conditional' in sim_data:
                            conditional_row = {**base_row, **sim_data['conditional']}
                            conditional_data.append(conditional_row)
        
        # 1. Save spatial similarity data
        if spatial_data:
            csv_path = os.path.join(output_path, 'spatial_similarity.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['module', 'step', 'tensor_type', 'mean', 'std', 'min', 'max'])
                writer.writeheader()
                writer.writerows(spatial_data)
        
        # 2. Save temporal similarity data
        if temporal_data:
            csv_path = os.path.join(output_path, 'temporal_similarity.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['module', 'step', 'tensor_type', 'similarity'])
                writer.writeheader()
                writer.writerows(temporal_data)
        
        # 3. Save conditional similarity data
        if conditional_data:
            csv_path = os.path.join(output_path, 'conditional_similarity.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['module', 'step', 'tensor_type', 'mean', 'std', 'min', 'max'])
                writer.writeheader()
                writer.writerows(conditional_data)