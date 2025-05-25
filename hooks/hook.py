import torch
from typing import Dict, Any


class HookAnalysis:
    def __init__(self, computation_strategy, analysis_manager):
        self.computation_strategy = computation_strategy
        self.analysis_manager = analysis_manager
    
    def __call__(self, module, input, output, name: str) -> torch.Tensor:
        if hasattr(module, 'step'):
            if module.step == 50:
                module.step = 1
            else:
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

        # Debug infinite differences
        # if not torch.allclose(out_computed, output, atol=1e-1, rtol=1e-1):
        #     print(f"[DEBUG] Issue in {name}:")
        #     print(f"  out_computed stats: min={out_computed.min()}, max={out_computed.max()}, mean={out_computed.mean()}")
        #     print(f"  output       stats: min={output.min()}, max={output.max()}, mean={output.mean()}")
        #     print(f"  out_computed has NaN: {torch.isnan(out_computed).any()}")
        #     print(f"  output       has NaN: {torch.isnan(output).any()}")
        #     print(f"  out_computed has Inf: {torch.isinf(out_computed).any()}")
        #     print(f"  output       has Inf: {torch.isinf(output).any()}")
            
        #     diff = torch.abs(out_computed - output)
        #     print(f"  max diff: {diff.max()}")
        #     print(f"  diff has Inf: {torch.isinf(diff).any()}")
            
        #     # Find where the infinite differences occur
        #     inf_mask = torch.isinf(diff)
        #     if inf_mask.any():
        #         inf_indices = torch.where(inf_mask)
        #         print(f"  Infinite diff locations count: {inf_mask.sum()}")
                
        #         # Flatten tensors to avoid indexing issues
        #         out_flat    = out_computed.flatten()
        #         output_flat = output.flatten()
        #         inf_flat    = inf_mask.flatten()
                
        #         print(f"  out_computed at inf locations: {out_flat[inf_flat][:5]}")
        #         print(f"  output       at inf locations: {output_flat[inf_flat][:5]}")
        
        return out_computed     # return computed result
        # return output     # return golden result