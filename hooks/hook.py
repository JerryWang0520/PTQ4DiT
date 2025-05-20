import torch
from typing import Dict, Any


class HookAnalysis:
    def __init__(self, computation_strategy, analysis_manager):
        self.computation_strategy = computation_strategy
        self.analysis_manager = analysis_manager
    
    def __call__(self, module, input, output, name: str) -> torch.Tensor:
        if hasattr(module, 'step'):
            module.step += 1
        else:
            module.step = 1
        
        module_info = {
            'name': name,
            'step': module.step,
            'fwd_func': getattr(module, 'fwd_func', None)
        }

        out_computed = self.computation_strategy.compute(module, input, output, name)        
        tensors_to_analyze = self.computation_strategy.get_tensors_to_analyze()
        
        if self.analysis_manager.analyzers:
            if "similarity" in self.analysis_manager.analyzers.keys():
                results = self.analysis_manager.analyze_tensor(tensors_to_analyze["act"], module_info=module_info)
                self.analysis_manager.store_results(name, module.step, "act", results)
            else:
                for tensor_name, tensor in tensors_to_analyze.items():
                    results = self.analysis_manager.analyze_tensor(tensor, module_info=module_info)
                    self.analysis_manager.store_results(name, module.step, tensor_name, results)
        
        assert torch.allclose(out_computed, output, atol=1e-3, rtol=1e-3), \
            f"out_computed != output in {name}: max diff = {torch.max(torch.abs(out_computed - output))}"
        
        return out_computed     # return computed result