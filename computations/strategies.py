import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base import ComputationStrategy


class OriginalComputation(ComputationStrategy):
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)
        xw_q = module.fwd_func(params['x_q'] - params['x_zp'], params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        
        if module.fwd_func == F.conv2d:
            fold_params = self._get_fold_params(module)
            self.tensors = {
                "act": F.unfold(params['x_q'] - params['x_zp'], **fold_params),
                "output": xw_q
            }
        elif module.fwd_func == F.linear:
            self.tensors = {
                "act": params['x_q'] - params['x_zp'],
                "output": xw_q
            }
        else:
            raise Exception("Unsupported fwd_func")
        
        return self._output_rescaling(module, xw_q, params['x_scale'], params['w_scale'])
    
    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        return self.tensors


class SpatialDifferenceComputation(ComputationStrategy):
    def __init__(self, ref_first: bool = False):
        super().__init__()
        self.ref_first = ref_first
        print("ref_first:", ref_first)
    
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)
        xw_q = module.fwd_func(params['x_q'] - params['x_zp'], params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        if module.fwd_func == F.conv2d:            
            fold_params = self._get_fold_params(module)
            x_delta = F.unfold(params['x_q'] - params['x_zp'], **fold_params)
            if self.ref_first:
                x_delta[:, :, 1:] -= x_delta[:, :, 0:1]
            else:
                x_prev = torch.roll(x_delta, 1, dims=-1)
                x_prev[:, :, 0:1] = 0
                x_delta -= x_prev
            
            w_col = params['w_q'] - params['w_zp']
            w_col = w_col.view(xw_q.shape[1], -1)
            xw_raw = torch.matmul(w_col, x_delta)   # differential computing

            if self.ref_first:
                xw_raw[:, :, 1:]  += xw_raw[:, :, 0:1]
            else:
                xw_raw = torch.cumsum(xw_raw, dim=-1)
            
            xw_raw = xw_raw.view(xw_q.shape[0], xw_q.shape[1], xw_q.shape[2], xw_q.shape[3])
            
            self.tensors = {
                "act": x_delta,
                "output": xw_raw
            }
        elif module.fwd_func == F.linear:
            x_delta = params['x_q'].clone()
            if x_delta.dim() in [2, 3]:
                if self.ref_first:
                    x_delta[..., 1:, :] -= x_delta[..., 0:1, :]
                    x_delta[..., 0:1, :] -= params['x_zp']

                    xw_raw = module.fwd_func(x_delta, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
                    xw_raw[..., 1:, :] += xw_raw[..., 0:1, :]
                else:
                    x_prev = torch.roll(x_delta, 1, dims=-2)
                    x_prev[..., 0:1, :] = params['x_zp']
                    x_delta -= x_prev

                    xw_raw = module.fwd_func(x_delta, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
                    xw_raw = torch.cumsum(xw_raw, dim=-2)

                self.tensors = {
                    "act": x_delta,
                    "output": xw_raw
                }
            else:
                raise Exception(f"Unsupported input dimension {x_delta.dim()}")
        else:
            raise Exception("Unsupported fwd_func")
        
        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])
    
    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        return self.tensors


class TemporalDifferenceComputation(ComputationStrategy):
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)
        xw_q = module.fwd_func(params['x_q'] - params['x_zp'], params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        if hasattr(module, 'x_prev'):  # 2nd~ step
            x_delta = params['x_q'] - module.x_prev
            xw_delta = module.fwd_func(x_delta, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
            xw_raw   = xw_delta + module.output_prev

            if module.fwd_func == F.conv2d:            
                fold_params = self._get_fold_params(module)
                self.tensors = {
                    "act": F.unfold(x_delta, **fold_params),
                    "output": xw_delta,
                }
            elif module.fwd_func == F.linear:
                self.tensors = {
                    "act": x_delta,
                    "output": xw_delta,
                }
            else:
                raise Exception("Unsupported fwd_func")
        else:  # 1st step
            xw_raw = xw_q

            if module.fwd_func == F.conv2d:            
                fold_params = self._get_fold_params(module)
                self.tensors = {
                    "act": F.unfold(params['x_q'] - params['x_zp'], **fold_params),
                    "output": xw_raw
                }
            elif module.fwd_func == F.linear:
                self.tensors = {
                    "act": params['x_q'] - params['x_zp'],
                    "output": xw_raw
                }
            else:
                raise Exception("Unsupported fwd_func")
        
        module.x_prev = params['x_q']
        module.output_prev = xw_raw

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])
    
    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        return self.tensors


class CFGDifferenceComputation(ComputationStrategy):
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)
        xw_q = module.fwd_func(params['x_q'] - params['x_zp'], params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        x_cond   = params['x_q'][0] - params['x_zp']
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        x_delta  = x_cond - x_uncond

        xw_cond   = module.fwd_func(x_cond  , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_uncond = module.fwd_func(x_uncond, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_delta  = module.fwd_func(x_delta , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_raw    = xw_delta + xw_uncond

        x_delta  = torch.stack([x_delta, x_uncond], dim=0)
        xw_delta = torch.stack([xw_delta, xw_uncond], dim=0)

        if module.fwd_func == F.conv2d:            
            fold_params = self._get_fold_params(module)
            self.tensors = {
                "act": F.unfold(x_delta, **fold_params),
                "output": xw_delta,
            }
        elif module.fwd_func == F.linear:
            self.tensors = {
                "act": x_delta,
                "output": xw_delta,
            }
        else:
            raise Exception("Unsupported fwd_func")

        xw_raw = torch.stack([xw_raw, xw_uncond], dim=0)

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])
    
    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        return self.tensors


class SpatialCFGDifferenceComputation(ComputationStrategy):
    def __init__(self, ref_first: bool = False):
        super().__init__()
        self.ref_first = ref_first
        print("ref_first:", ref_first)
    
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)
        xw_q = module.fwd_func(params['x_q'] - params['x_zp'], params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        # Step 1: CUD
        x_cond   = params['x_q'][0] - params['x_zp']
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        x_delta  = x_cond - x_uncond

        xw_cond   = module.fwd_func(x_cond  , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_uncond = module.fwd_func(x_uncond, params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        # Step 2: uncond SD
        if module.fwd_func == F.conv2d:
            fold_params = self._get_fold_params(module)
            x_uncond = F.unfold(x_uncond, **fold_params)
            if self.ref_first:
                x_uncond[:, 1:] -= x_uncond[:, 0:1]
            else:
                x_prev = torch.roll(x_uncond, 1, dims=-1)
                x_prev[:, 0:1] = 0
                x_uncond -= x_prev
            
            w_col = params['w_q'] - params['w_zp']
            w_col = w_col.view(xw_q.shape[1], -1)
            xw_uncond_raw = torch.matmul(w_col, x_uncond)   # differential computing

            if self.ref_first:
                xw_uncond_raw[:, :, 1:]  += xw_uncond_raw[:, :, 0:1]
            else:
                xw_uncond_raw = torch.cumsum(xw_uncond_raw, dim=-1)
            
            xw_uncond_raw = xw_uncond_raw.view(xw_q.shape[1], xw_q.shape[2], xw_q.shape[3])
        elif module.fwd_func == F.linear:
            if x_uncond.dim() in [2, 3]:
                if self.ref_first:
                    x_uncond[..., 1:, :] -= x_uncond[..., 0:1, :]

                    xw_uncond_raw = module.fwd_func(x_uncond, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
                    xw_uncond_raw[..., 1:, :] += xw_uncond_raw[..., 0:1, :]
                else:
                    x_prev = torch.roll(x_uncond, 1, dims=-2)
                    x_prev[..., 0:1, :] = 0
                    x_uncond -= x_prev

                    xw_uncond_raw = module.fwd_func(x_uncond, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
                    xw_uncond_raw = torch.cumsum(xw_uncond_raw, dim=-2)
            else:
                raise Exception(f"Unsupported input dimension {x_uncond.dim()} in {name}")
        else:
            raise Exception("Unsupported fwd_func")
        assert torch.equal(xw_uncond_raw, xw_uncond), f"xw_uncond_raw != xw_uncond in {name}: max diff = {torch.max(torch.abs(xw_uncond_raw - xw_uncond))}"

        # Step 3: fwd_func
        xw_delta  = module.fwd_func(x_delta , params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        # Step 4: SD recovery

        # Step 5: CUD recovery
        xw_raw = xw_delta + xw_uncond_raw

        # xw_delta = torch.stack([xw_delta, xw_uncond_raw], dim=0)

        if module.fwd_func == F.conv2d:            
            fold_params = self._get_fold_params(module)
            self.tensors = {
                "act": torch.stack([F.unfold(x_delta, **fold_params), x_uncond], dim=0),
                "act_cond": F.unfold(x_delta, **fold_params),
                "act_uncond": x_uncond,
                # "output": xw_delta,
            }
        elif module.fwd_func == F.linear:
            self.tensors = {
                "act": torch.stack([x_delta, x_uncond], dim=0),
                "act_cond": x_delta,
                "act_uncond": x_uncond,
                # "output": xw_delta,
            }
        else:
            raise Exception("Unsupported fwd_func")

        xw_raw = torch.stack([xw_raw, xw_uncond_raw], dim=0)

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])
    
    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        return self.tensors


class LargeNumbersCFGDifferenceComputation(ComputationStrategy):
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)
        xw_q = module.fwd_func(params['x_q'] - params['x_zp'], params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        x_cond   = params['x_q'][0] - params['x_zp']
        x_uncond = params['x_q'][1] - params['x_zp']    # reference

        # Compare x_cond values
        # small_mask = (x_cond >= -4) & (x_cond <= 3)   # 3-bit range
        # small_mask = (x_cond >= -8) & (x_cond <= 7)   # 4-bit range
        small_mask = (x_cond >= -16) & (x_cond <= 15)   # 5-bit range
        large_mask = ~small_mask

        x_cond_small = torch.where(small_mask, x_cond, torch.zeros_like(x_cond))
        x_cond_large = torch.where(large_mask, x_cond, torch.zeros_like(x_cond))

        x_uncond_small = torch.where(small_mask, x_uncond, torch.zeros_like(x_uncond))
        x_uncond_large = torch.where(large_mask, x_uncond, torch.zeros_like(x_uncond))

        x_delta = x_cond_large - x_uncond_large
        x_raw = torch.stack([x_cond_small + x_delta, x_uncond_small + x_uncond_large], dim=0)

        xw_cond_small   = module.fwd_func(x_cond_small  , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_delta        = module.fwd_func(x_delta       , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_uncond_small = module.fwd_func(x_uncond_small, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_uncond_large = module.fwd_func(x_uncond_large, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_cond_large   = xw_delta + xw_uncond_large

        xw_cond = xw_cond_small + xw_cond_large
        xw_uncond = xw_uncond_small + xw_uncond_large

        xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)

        if module.fwd_func == F.conv2d:            
            fold_params = self._get_fold_params(module)
            self.tensors = {
                "act": F.unfold(x_raw, **fold_params),
                "act_delta": F.unfold(x_delta, **fold_params),
                "act_cond_large": F.unfold(x_cond_large, **fold_params),
                "act_cond": F.unfold(x_cond_small + x_delta, **fold_params),
                "act_uncond": F.unfold(x_uncond_small + x_uncond_large, **fold_params),
                # "output": xw_delta,
            }
        elif module.fwd_func == F.linear:
            self.tensors = {
                "act": x_raw,
                "act_delta": x_delta,
                "act_cond_large": x_cond_large,
                "act_cond": x_cond_small + x_delta,
                "act_uncond": x_uncond_small + x_uncond_large,
                # "output": xw_delta,
            }
        else:
            raise Exception("Unsupported fwd_func")

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])

    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        return self.tensors


class OptimalCFGDifferenceComputation(ComputationStrategy):
    def _get_bitwidth_type(self, tensor, bit_th: int = 4) -> torch.Tensor:
        tensor_type = torch.zeros_like(tensor)

        lower_bound = -1*2**(bit_th-1)
        upper_bound = 2**(bit_th-1)-1
        # print("bit_th, lower_bound, upper_bound:", bit_th, lower_bound, upper_bound)

        small_mask = (tensor >= lower_bound) & (tensor <= upper_bound) & (tensor != 0)
        tensor_type[small_mask] = 1

        large_mask = (tensor < lower_bound) | (tensor > upper_bound)
        tensor_type[large_mask] = 2

        return tensor_type

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)
        xw_q = module.fwd_func(params['x_q'] - params['x_zp'], params['w_q'] - params['w_zp'], **module.fwd_kwargs)

        x_cond   = params['x_q'][0] - params['x_zp']
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        x_delta  = x_cond - x_uncond

        x_cond_type   = self._get_bitwidth_type(x_cond)
        x_delta_type  = self._get_bitwidth_type(x_delta)

        cfg_mask = x_delta_type < x_cond_type

        x_cond_raw     = torch.where(~cfg_mask, x_cond  , torch.zeros_like(x_cond))
        x_cond_delta   = torch.where( cfg_mask, x_delta , torch.zeros_like(x_delta))
        x_uncond_raw   = torch.where(~cfg_mask, x_uncond, torch.zeros_like(x_uncond))
        x_uncond_delta = torch.where( cfg_mask, x_uncond, torch.zeros_like(x_uncond))
 
        x_raw = torch.stack([x_cond_raw + x_cond_delta, x_uncond], dim=0)

        xw_cond_raw     = module.fwd_func(x_cond_raw    , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_cond_delta   = module.fwd_func(x_cond_delta  , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_uncond_raw   = module.fwd_func(x_uncond_raw  , params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_uncond_delta = module.fwd_func(x_uncond_delta, params['w_q'] - params['w_zp'], **module.fwd_kwargs)
        xw_cond         = xw_cond_raw + xw_cond_delta + xw_uncond_delta
        xw_uncond       = xw_uncond_raw + xw_uncond_delta

        xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)

        if module.fwd_func == F.conv2d:            
            fold_params = self._get_fold_params(module)
            self.tensors = {
                "act": F.unfold(x_raw, **fold_params),
                "act_cond": F.unfold(x_cond, **fold_params),
                "act_delta": F.unfold(x_delta, **fold_params),
                "act_cond_raw": F.unfold(x_cond_raw, **fold_params),
                "act_cond_delta": F.unfold(x_cond_delta, **fold_params),
                "act_uncond": F.unfold(x_uncond, **fold_params),
                "act_uncond_raw": F.unfold(x_uncond_raw, **fold_params),
                "act_uncond_delta": F.unfold(x_uncond_delta, **fold_params),
            }
        elif module.fwd_func == F.linear:
            self.tensors = {
                "act": x_raw,
                "act_cond": x_cond,
                "act_delta": x_delta,
                "act_cond_raw": x_cond_raw,
                "act_cond_delta": x_cond_delta,
                "act_uncond": x_uncond,
                "act_uncond_raw": x_uncond_raw,
                "act_uncond_delta": x_uncond_delta,
            }
        else:
            raise Exception("Unsupported fwd_func")

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])

    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        return self.tensors


def create_strategy(strategy_type: str, **kwargs) -> ComputationStrategy:
    strategies = {
        'original': OriginalComputation,
        'spatial': SpatialDifferenceComputation,
        'temporal': TemporalDifferenceComputation,
        'cfg': CFGDifferenceComputation,
        'spatial_cfg': SpatialCFGDifferenceComputation,
        'cfg_large': LargeNumbersCFGDifferenceComputation,
        'cfg_opt': OptimalCFGDifferenceComputation,
    }
    
    if strategy_type not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {available}")
    
    return strategies[strategy_type](**kwargs)