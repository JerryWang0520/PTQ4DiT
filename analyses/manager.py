import os
from collections import defaultdict
from typing import Dict, List, Optional, Any
from .analyzers import BitwidthAnalyzer, StatisticsAnalyzer, DistributionAnalyzer
from .similarity_analyzer import StatefulSimilarityAnalyzer
from .shape_analyzer import ShapeAnalyzer
from .base import TensorAnalyzer


class TensorAnalysisManager:
    def __init__(self, active_analyzer=None, analyzer_configs=None):
        self.analyzer_configs = analyzer_configs or {}

        self.all_analyzers = {
            "bitwidth": BitwidthAnalyzer(),
            "statistics": StatisticsAnalyzer(),
            "distribution": DistributionAnalyzer(),
            "similarity": StatefulSimilarityAnalyzer(**self.analyzer_configs.get('similarity', {})),
            "shape": ShapeAnalyzer()
        }
        
        # Set active analyzer
        if active_analyzer:
            self.analyzers = {k: v for k, v in self.all_analyzers.items() if k == active_analyzer}
            print("active_analyzer:", active_analyzer)
        else:
            self.analyzers = None
        
        self.results = defaultdict(lambda: defaultdict(dict))
        self.output_dir = None
    
    def add_analyzer(self, name: str, analyzer: TensorAnalyzer) -> None:
        self.analyzers[name] = analyzer
    
    def remove_analyzer(self, name: str) -> None:
        if name in self.analyzers:
            del self.analyzers[name]
    
    def analyze_tensor(self, tensor, module_info=None, **kwargs):
        if not self.analyzers:
            return {}
        
        results = {}
        for analyzer_name, analyzer in self.analyzers.items():
            try:
                results[analyzer_name] = analyzer.analyze(tensor, module_info=module_info, **kwargs)
            except Exception as e:
                print(f"Error in {analyzer_name} analyzer: {e}")
                results[analyzer_name] = {"error": str(e)}
                return {}
        
        return results
    
    def store_results(
        self, 
        module_name: str, 
        step: int, 
        tensor_type: str, 
        results: Dict[str, Any]
    ) -> None:
    
        self.results[module_name][step][tensor_type] = results
    
    def save_all_results(self, output_dir: Optional[str] = None) -> None:
        output_dir = output_dir or self.output_dir
        if output_dir is None:
            raise ValueError("No output directory specified")
        
        for analyzer_name, analyzer in self.analyzers.items():
            analyzer_output_dir = os.path.join(output_dir, analyzer_name)
            analyzer.save_results(self.results, analyzer_output_dir)
    
    def clear_results(self) -> None:
        self.results.clear()
    
    def get_results_summary(self) -> Dict[str, Any]:
        summary = {
            "num_modules": len(self.results),
            "modules": list(self.results.keys()),
            "total_steps": sum(len(steps) for steps in self.results.values()),
            "analyzers_used": list(self.analyzers.keys())
        }
        return summary