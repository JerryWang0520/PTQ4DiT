from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch


class TensorAnalyzer(ABC):
    @abstractmethod
    def analyze(self, tensor: torch.Tensor, module_info: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        pass