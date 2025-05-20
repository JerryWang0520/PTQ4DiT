from functools import partial
from typing import Optional, List, Any, Callable, Dict
import logging

from .hook import HookAnalysis
from computations.strategies import create_strategy
from analyses.manager import TensorAnalysisManager

logger = logging.getLogger(__name__)


def register_hooks_with_strategy(
    model, 
    computation_type: str, 
    analysis_manager: TensorAnalysisManager,
    exclude_modules: Optional[List[str]] = None,
    **strategy_kwargs: Any
) -> List[Callable]:

    strategy = create_strategy(computation_type, **strategy_kwargs)
    logger.info(f"Created {computation_type} computation strategy")
    
    hook = HookAnalysis(strategy, analysis_manager)
    
    handles = []
    
    from quant.quant_layer import QuantModule
    
    for name, module in model.named_modules():
        should_hook = False
        
        if exclude_modules is not None:
            should_hook = name not in exclude_modules and isinstance(module, QuantModule)
        else:
            should_hook = isinstance(module, QuantModule)
        
        if should_hook:
            handle = module.register_forward_hook(partial(hook, name=name))
            handles.append(handle)
            logger.debug(f"Registered hook on module: {name}")
    
    logger.info(f"Registered {len(handles)} hooks with {computation_type} strategy")
    return handles


def remove_hooks(handles: List[Callable]) -> None:
    for handle in handles:
        handle.remove()
    logger.info(f"Removed {len(handles)} hooks")


def setup_analysis_hooks(
    model,
    computation_type: str,
    output_dir: str,
    analyzers: Optional[List[str]] = None,
    analyzer_configs: Optional[Dict] = None,
    **strategy_kwargs: Any
) -> tuple[TensorAnalysisManager, List[Callable]]:
    
    analysis_manager = TensorAnalysisManager(analyzer_configs=analyzer_configs)
    
    if analyzers is not None:
        available_analyzers = list(analysis_manager.analyzers.keys())
        for analyzer_name in available_analyzers:
            if analyzer_name not in analyzers:
                del analysis_manager.analyzers[analyzer_name]
    
    # Register hooks
    handles = register_hooks_with_strategy(
        model=model,
        computation_type=computation_type,
        analysis_manager=analysis_manager,
        **strategy_kwargs
    )
    
    analysis_manager.output_dir = output_dir
    
    return analysis_manager, handles