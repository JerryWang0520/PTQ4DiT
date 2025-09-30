import torch
from typing import Dict, Any


class HookAnalysis:
    def __init__(self, computation_strategy, analysis_manager, tensor_saver=None):
        self.computation_strategy = computation_strategy
        self.analysis_manager = analysis_manager
        self.tensor_saver = tensor_saver
    
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
        
        if self.analysis_manager.analyzers:
            tensors_to_analyze = self.computation_strategy.get_tensors_to_analyze()

            if "similarity" in self.analysis_manager.analyzers.keys() or \
               "bitslice"   in self.analysis_manager.analyzers.keys():
            #    "bitwidth"   in self.analysis_manager.analyzers.keys() or \
                results = self.analysis_manager.analyze_tensor(tensors_to_analyze["act"], module_info=module_info)
                self.analysis_manager.store_results(name, module.step, "act", results)
            elif "bitwidth" in self.analysis_manager.analyzers.keys():
                for tensor_name, tensor in tensors_to_analyze.items():
                    results = self.analysis_manager.analyze_tensor(tensor, module_info=module_info)
                    self.analysis_manager.store_results(name, module.step, tensor_name, results)
            else:
                for tensor_name, tensor in tensors_to_analyze.items():
                    results = self.analysis_manager.analyze_tensor(tensor, module_info=module_info)
                    self.analysis_manager.store_results(name, module.step, tensor_name, results)
        
        if self.tensor_saver:
            tensors_to_save = self.computation_strategy.get_tensors_to_save()
            self.tensor_saver.save_global_tiles(tensors_to_save, name, module.step)
            self.tensor_saver.save_local_tiles(name, module.step)

        # assert torch.allclose(out_computed, output, atol=1e-3, rtol=1e-3), \
        #     f"out_computed != output in {name}: max diff = {torch.max(torch.abs(out_computed - output))}"
        return out_computed


class HookAnalysisNoStrategy:
    def __init__(self, analysis_manager):
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

        if self.analysis_manager.analyzers:
            self._analyze_by_type(module, input, output, name, module_info)

        return output
    
    def _analyze_by_type(self, module, input, output, name, module_info):
        for analyzer_name in self.analysis_manager.analyzers.keys():
            if analyzer_name == "shape":
                self._analyze_for_shape(module, input, output, name, module_info)

    def _analyze_for_shape(self, module, input, output, name, module_info):       
        # Analyze input
        if input and len(input) > 0:
            results = self.analysis_manager.analyze_tensor(input[0], module_info=module_info)
            if results:
                self.analysis_manager.store_results(name, module.step, "input", results)
        
        # Analyze weight
        if hasattr(module, 'weight'):
            results = self.analysis_manager.analyze_tensor(module.weight, module_info=module_info)
            if results:
                self.analysis_manager.store_results(name, module.step, "weight", results)
        
        # Analyze output
        results = self.analysis_manager.analyze_tensor(output, module_info=module_info)
        if results:
            self.analysis_manager.store_results(name, module.step, "output", results)
    