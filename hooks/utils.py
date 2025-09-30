from functools import partial
from typing import Optional, List, Any, Callable, Dict, Tuple
import logging

from .hook import HookAnalysis, HookAnalysisNoStrategy
from computations.strategies import create_strategy
from analyses.manager import TensorAnalysisManager
from saver import TensorSaver

logger = logging.getLogger(__name__)


def register_hooks_without_strategy(
    model,
    analysis_manager: TensorAnalysisManager,
    include_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None
) -> List[Callable]:

    hook = HookAnalysisNoStrategy(analysis_manager)

    handles = []
    
    try:
        from quant.quant_layer import QuantModule   # DiT in PTQ4DiT
    except ImportError:
        from qdiff.quant_layer import QuantModule   # SDv1-4 in Q-Diffusion
    
    for name, module in model.named_modules():
        should_hook = False
        
        if include_modules is not None:
            included = any(pattern in name for pattern in include_modules)
            should_hook = included and isinstance(module, QuantModule)
        elif exclude_modules is not None:
            excluded = any(pattern in name for pattern in exclude_modules)
            should_hook = not excluded and isinstance(module, QuantModule)
        else:
            should_hook = isinstance(module, QuantModule)
        
        if should_hook:
            handle = module.register_forward_hook(partial(hook, name=name))
            handles.append(handle)
            logger.debug(f"Registered hook on module: {name}")
    
    logger.info(f"Registered {len(handles)} hooks without computation strategy")
    return handles


def register_hooks_with_strategy(
    model, 
    computation_type: str, 
    analysis_manager: TensorAnalysisManager,
    bitslice_method: Optional[str] = None,
    bitslice_kwargs: Optional[Dict] = None,
    include_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    saver_kwargs: Optional[Dict] = None,
    **strategy_kwargs: Any
) -> List[Callable]:

    strategy = create_strategy(computation_type, 
                               bitslice_method=bitslice_method, 
                               bitslice_kwargs=bitslice_kwargs, 
                               analysis_manager=analysis_manager, 
                               **strategy_kwargs)
    logger.info(f"Created {computation_type} computation strategy")

    tensor_saver = None
    if saver_kwargs and saver_kwargs.get("enabled", False):
        tensor_saver = TensorSaver(**saver_kwargs)
        logger.info(f"Created TensorSaver")
    
    hook = HookAnalysis(strategy, analysis_manager, tensor_saver)

    handles = []
    
    try:
        from quant.quant_layer import QuantModule   # DiT in PTQ4DiT
    except ImportError:
        from qdiff.quant_layer import QuantModule   # SDv1-4 in Q-Diffusion
    
    for name, module in model.named_modules():
        should_hook = False
        
        if include_modules is not None:
            included = any(pattern in name for pattern in include_modules)
            should_hook = included and isinstance(module, QuantModule)
        elif exclude_modules is not None:
            excluded = any(pattern in name for pattern in exclude_modules)
            should_hook = not excluded and isinstance(module, QuantModule)
        else:
            should_hook = isinstance(module, QuantModule)
        
        if should_hook:
            handle = module.register_forward_hook(partial(hook, name=name))
            handles.append(handle)
            logger.debug(f"Registered hook on module: {name}")
    
    logger.info(f"Registered {len(handles)} hooks with {computation_type} strategy")
    return handles


def register_analysis_hooks(
    model,
    analysis_manager: TensorAnalysisManager,
    computation_strategy: Optional[str] = None,
    bitslice_method: Optional[str] = None,
    bitslice_kwargs: Optional[Dict] = None,
    include_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    saver_kwargs: Optional[Dict] = None,
    **strategy_kwargs: Any
) -> List[Callable]:
    
    no_computation_analyses = {"shape", "statistics", "distribution"}
    computation_analyses    = {"bitwidth", "similarity"}
    
    active_analyzers = set(analysis_manager.analyzers.keys()) if analysis_manager.analyzers else set()
    
    needs_computation    = bool(active_analyzers & computation_analyses)
    needs_no_computation = bool(active_analyzers & no_computation_analyses)

    # Validation
    if needs_computation and not computation_strategy:
        raise ValueError(f"Computation strategy required for analyses: {list(active_analyzers & computation_analyses)}")

    # Early return if no analyzers and no strategy provided
    if not needs_computation and not needs_no_computation and computation_strategy is None:
        logger.warning("No recognized analyzers found and no computation strategy provided, no hooks registered")
        return []
    
    handles = []
    
    # Register no-computation hooks
    if needs_no_computation:
        logger.info(f"Registering hooks for no-computation analyses: {list(active_analyzers & no_computation_analyses)}")
        handles.extend(register_hooks_without_strategy(model=model, 
                                                       analysis_manager=analysis_manager, 
                                                       include_modules=include_modules, 
                                                       exclude_modules=exclude_modules))

    # Register computation hooks
    if needs_computation:
        logger.info(f"Registering hooks for computation analyses: {list(active_analyzers & computation_analyses)}")
        handles.extend(register_hooks_with_strategy(model=model, 
                                                    computation_type=computation_strategy, 
                                                    analysis_manager=analysis_manager, 
                                                    bitslice_method=bitslice_method,
                                                    bitslice_kwargs=bitslice_kwargs,
                                                    include_modules=include_modules, 
                                                    exclude_modules=exclude_modules,
                                                    saver_kwargs=saver_kwargs,
                                                    **strategy_kwargs))

    # Fallback: register computation hooks when no analyzers but strategy provided
    if not needs_computation and not needs_no_computation and computation_strategy is not None:
        logger.info(f"No analyzers specified, but computation strategy provided - registering computation hooks")
        handles.extend(register_hooks_with_strategy(model=model, 
                                                    computation_type=computation_strategy, 
                                                    analysis_manager=analysis_manager, 
                                                    bitslice_method=bitslice_method,
                                                    bitslice_kwargs=bitslice_kwargs,
                                                    include_modules=include_modules, 
                                                    exclude_modules=exclude_modules,
                                                    saver_kwargs=saver_kwargs,
                                                    **strategy_kwargs))

    logger.info(f"Total hooks registered: {len(handles)}")
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
) -> Tuple[TensorAnalysisManager, List[Callable]]:
    
    analysis_manager = TensorAnalysisManager(analyzer_configs=analyzer_configs)
    
    if analyzers is not None:
        available_analyzers = list(analysis_manager.analyzers.keys())
        for analyzer_name in available_analyzers:
            if analyzer_name not in analyzers:
                del analysis_manager.analyzers[analyzer_name]
    
    # Register hooks
    handles = register_analysis_hooks(
        model=model,
        analysis_manager=analysis_manager,
        computation_strategy=computation_type,
        **strategy_kwargs
    )
    
    analysis_manager.output_dir = output_dir
    
    return analysis_manager, handles