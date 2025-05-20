"""Shape analyzer for tensor shapes and element counts"""

from typing import Dict, Any, Optional
import torch
import json
from .base import TensorAnalyzer


class ShapeAnalyzer(TensorAnalyzer):
    """Analyze input/output shapes and element counts for tensors"""
    
    def __init__(self):
        self.first_step_only = True
        
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Record tensor shape and element count"""
        
        # Only analyze first step
        if module_info and self.first_step_only:
            if module_info.get('step', 1) > 1:
                return {}
        
        shape = list(tensor.shape)
        num_elements = tensor.numel()
        
        return {
            "shape": shape,
            "num_elements": num_elements
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save shape analysis results to CSV"""
        import csv
        import os

        os.makedirs(output_dir, exist_ok=True)
        
        shape_data = []
        
        for module_name, module_data in results.items():
            for step, step_data in module_data.items():
                for tensor_type, analysis_results in step_data.items():
                    if 'shape' in analysis_results and analysis_results['shape']:
                        shape_info = analysis_results['shape']
                        row = {
                            'module': module_name,
                            'tensor_type': tensor_type,
                            'shape': str(shape_info['shape']),
                            'num_elements': shape_info['num_elements']
                        }
                        shape_data.append(row)
                if shape_data and shape_data[-1]['module'] == module_name:
                    break  # Only process first step
        
        csv_path = os.path.join(output_dir, 'shapes.csv')
        if shape_data:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['module', 'tensor_type', 'shape', 'num_elements'])
                writer.writeheader()
                writer.writerows(shape_data)