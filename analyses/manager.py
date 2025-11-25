import os
from collections import defaultdict
from typing import Dict, List, Optional, Any
from .analyzers import StatisticsAnalyzer, DistributionAnalyzer
from .bitwidth_analyzer import BitwidthAnalyzer
from .similarity_analyzer import StatefulSimilarityAnalyzer
from .shape_analyzer import ShapeAnalyzer
# from .bitslice_amount_analyzer import BitSliceAmountAnalyzer
from .base import TensorAnalyzer
import logging

logger = logging.getLogger(__name__)


class TensorAnalysisManager:
    def __init__(self, sample=False, active_analyzer=None, analyzer_configs=None, output_dir=None):
        self.sample = sample
        self.analyzer_configs = analyzer_configs or {}

        self.all_analyzers = {
            "bitwidth": BitwidthAnalyzer(),
            "statistics": StatisticsAnalyzer(),
            "distribution": DistributionAnalyzer(),
            "similarity": StatefulSimilarityAnalyzer(**self.analyzer_configs.get('similarity', {})),
            "shape": ShapeAnalyzer(),
            # "bitslice": BitsliceAnalyzer(),
        }
        
        # Set active analyzer
        if self.sample:
            self.analyzers = {}
        else:
            if active_analyzer:
                self.analyzers = {k: v for k, v in self.all_analyzers.items() if k == active_analyzer}
                logger.info(f"active_analyzer: {active_analyzer}")
            else:
                self.analyzers = {}
        
        self.results = defaultdict(lambda: defaultdict(dict))
        self.output_dir = output_dir or None
    
    def add_analyzer(self, name: str, analyzer: TensorAnalyzer) -> None:
        if not self.sample:
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
                if analyzer_name == "bitslice":
                    if analyzer.performance:
                        results['performance'] = analyzer.analyze_performance(tensor, module_info=module_info)
                    else:
                        results[analyzer_name] = analyzer.analyze(tensor, module_info=module_info, **kwargs)
                else:
                    results[analyzer_name] = analyzer.analyze(tensor, module_info=module_info, **kwargs)
            except Exception as e:
                logger.error(f"Error in {analyzer_name} analyzer: {e}")
                results[analyzer_name] = {"error": str(e)}
        
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
            if analyzer_name == "bitslice":
                if analyzer.performance:
                    analyzer_output_dir = os.path.join(output_dir, "performance")
                    analyzer.save_performance_result(self.results, analyzer_output_dir)
                else:
                    analyzer_output_dir = os.path.join(output_dir, analyzer_name)
                    analyzer.save_results(self.results, analyzer_output_dir)
                pass
            else:
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