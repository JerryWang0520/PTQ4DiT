import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base import ComputationStrategy
from .bitslice_strategies import create_bitslice_strategy
import logging

logger = logging.getLogger(__name__)

class OriginalComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, save=False):
        super().__init__(clamp, bitslice_strategy, save)

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_raw = params['x_q'] - params['x_zp']
        w_raw = params['w_q'] - params['w_zp']
        xw_q = module.fwd_func(x_raw, w_raw, **module.fwd_kwargs)
        
        # 1. Prepare Input Data
        x_delta = self.get_Raw_input(module, name, x_raw)

        # 2. Apply bitslice reconstruction
        x_delta_reconstructed = self.apply_bitslice_reconstruction(x_delta)

        # 3. Computation
        xw_delta = self.fwd_func(module, name, x_delta_reconstructed, w_raw)
        
        # 4. Output Recovery
        xw_raw = self.get_Raw_output(module, name, xw_delta, xw_q.shape)

        self.tensors = {
            "act": x_delta,
            "output": xw_delta,
        }

        if self.save:
            self.save_tensors = {
                # "x_q"   : params["x_q" ],       # uint8: 0~255
                # "w_q"   : params["w_q" ].t(),   # uint8: 0~255
                # "x_zp"  : params["x_zp"],       # uint8: 0~255
                # "w_zp"  : params["w_zp"].t(),   # uint8: 0~255
                "x_raw" : x_delta,              # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": xw_delta,             # int24
            }
        
        # if "x_zp" in self.tensors: print(self.tensors["x_zp"])
        # if "w_zp" in self.tensors: print(self.tensors["w_zp"])
       
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_q, params['x_scale'], params['w_scale'])


class SpatialDifferenceComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, ref_first: bool = False, save=False):
        super().__init__(clamp, bitslice_strategy, save)
        self.ref_first = ref_first
        logger.info(f"ref_first: {self.ref_first}")

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_raw = params['x_q'] - params['x_zp']
        w_raw = params['w_q'] - params['w_zp']
        xw_q = module.fwd_func(x_raw, w_raw, **module.fwd_kwargs)

        # 1. Prepare Input Data
        x_delta = self.get_SD_input(module, name, x_raw, self.ref_first)

        # 2. Apply bitslice reconstruction
        x_delta_reconstructed = self.apply_bitslice_reconstruction(x_delta)
        
        # 3. Computation
        xw_delta = self.fwd_func(module, name, x_delta_reconstructed, w_raw)

        # 4. Output Recovery
        xw_raw = self.get_SD_output(module, name, xw_delta, xw_q.shape, self.ref_first)

        self.tensors = {
            "act": x_delta,
            "output": xw_delta
        }

        if self.save:
            self.save_tensors = {
                "x_raw" : x_delta,              # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": xw_delta,             # int24
            }
        
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class TemporalDifferenceComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, save=False):
        super().__init__(clamp, bitslice_strategy, save)

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_raw = params['x_q'] - params['x_zp']
        w_raw = params['w_q'] - params['w_zp']
        xw_q = module.fwd_func(x_raw, w_raw, **module.fwd_kwargs)

        if hasattr(module, 'in_prev'):      # 2nd~ step
            # 1. Prepare Input Data
            x_delta = self.get_TD_input(module, name, x_raw, module.in_prev)

            # 2. Apply bitslice reconstruction
            x_delta_reconstructed = self.apply_bitslice_reconstruction(x_delta)

            # 3. Computation
            xw_delta = self.fwd_func(module, name, x_delta_reconstructed, w_raw)

            # 4. Output Recovery
            xw_raw = self.get_TD_output(module, name, xw_delta, module.out_prev, xw_q.shape)
        else:                               # 1st step
            # 1. Prepare Input Data
            x_delta = self.get_Raw_input(module, name, x_raw)

            # 2. Apply bitslice reconstruction
            x_delta_reconstructed = self.apply_bitslice_reconstruction(x_delta)

            # 3. Computation
            xw_delta = self.fwd_func(module, name, x_delta_reconstructed, w_raw)
            
            # 4. Output Recovery
            xw_raw = self.get_Raw_output(module, name, xw_delta, xw_q.shape)
        
        module.in_prev  = x_raw
        module.out_prev = xw_raw

        self.tensors = {
            "act": x_delta,
            "output": xw_delta
        }

        if self.save:
            self.save_tensors = {
                "x_raw" : x_delta,              # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": xw_delta,             # int24
            }
        
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class CFGDifferenceComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, save=False):
        super().__init__(clamp, bitslice_strategy, save)

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']    # target
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        w_raw    = params['w_q']    - params['w_zp']

        xw_cond_q   = module.fwd_func(x_cond  , w_raw, **module.fwd_kwargs)
        xw_uncond_q = module.fwd_func(x_uncond, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_q, xw_uncond_q], dim=0)

        # 1. Prepare Input Data
        x_cond   = self.get_CUD_input(module, name, x_cond, x_uncond)
        x_uncond = self.get_Raw_input(module, name, x_uncond)
        tensor_in = torch.stack([x_cond, x_uncond], dim=0)

        # 2. Apply bitslice reconstruction
        x_cond_reconstructed   = self.apply_bitslice_reconstruction(x_cond)
        x_uncond_reconstructed = self.apply_bitslice_reconstruction(x_uncond)
        
        # 2. Computation
        xw_cond   = self.fwd_func(module, name, x_cond_reconstructed  , w_raw)
        xw_uncond = self.fwd_func(module, name, x_uncond_reconstructed, w_raw)
        tensor_out = torch.stack([xw_cond, xw_uncond], dim=0)

        # 3. Output Recovery
        xw_uncond = self.get_Raw_output(module, name, xw_uncond, xw_cond_q.shape)
        xw_cond   = self.get_CUD_output(module, name, xw_cond, xw_uncond, xw_uncond_q.shape)
        xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)

        self.tensors = {
            "act": tensor_in,
            "output": tensor_out
        }

        if self.save:
            self.save_tensors = {
                "x_raw" : tensor_in,            # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": tensor_out,           # int24
            }
        
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class SpatialCFGDifferenceComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, ref_first: bool = False, save=False):
        super().__init__(clamp, bitslice_strategy, save)
        self.ref_first = ref_first
        logger.info(f"ref_first: {self.ref_first}")
    
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']    # target
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        w_raw    = params['w_q']    - params['w_zp']

        xw_cond_q   = module.fwd_func(x_cond  , w_raw, **module.fwd_kwargs)
        xw_uncond_q = module.fwd_func(x_uncond, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_q, xw_uncond_q], dim=0)

        # 1. Prepare Input Data
        x_cond   = self.get_CUD_input(module, name, x_cond, x_uncond)
        x_uncond = self.get_SD_input(module, name, x_uncond, self.ref_first)
        tensor_in = torch.stack([x_cond, x_uncond], dim=0)

        # 2. Apply bitslice reconstruction
        x_cond_reconstructed   = self.apply_bitslice_reconstruction(x_cond)
        x_uncond_reconstructed = self.apply_bitslice_reconstruction(x_uncond)
        
        # 3. Computation
        xw_cond   = self.fwd_func(module, name, x_cond_reconstructed  , w_raw)
        xw_uncond = self.fwd_func(module, name, x_uncond_reconstructed, w_raw)
        tensor_out = torch.stack([xw_cond, xw_uncond], dim=0)

        # 4. Output Recovery
        xw_uncond = self.get_SD_output(module, name, xw_uncond, xw_cond_q.shape, self.ref_first)
        xw_cond   = self.get_CUD_output(module, name, xw_cond, xw_uncond, xw_uncond_q.shape)
        xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)

        self.tensors = {
            "act": tensor_in,
            "output": tensor_out
        }

        if self.save:
            self.save_tensors = {
                "x_raw" : tensor_in,            # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": tensor_out,           # int24
            }
        
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class LargeNumbersCFGDifferenceComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, bit_th: int = 4):
        super().__init__(clamp, bitslice_strategy)
        self.bit_th = bit_th        
        logger.info(f"bit_th: {self.bit_th}")

    def _get_mask(self, tensor_tar):
        lower_bound = -1*2**(self.bit_th-1)
        upper_bound =    2**(self.bit_th-1)-1
        # logger.info("lower_bound, upper_bound:", lower_bound, upper_bound)

        small_mask = (tensor_tar >= lower_bound) & (tensor_tar <= upper_bound)

        del lower_bound
        del upper_bound

        return small_mask
    
    def _get_masked_tensor(self, tensor, mask):
        return torch.where(mask, tensor, torch.zeros_like(tensor))

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']    # target
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        w_raw    = params['w_q']    - params['w_zp']

        small_mask = self._get_mask(x_cond)
        x_cond_small   = self._get_masked_tensor(x_cond  ,  small_mask)
        x_cond_large   = self._get_masked_tensor(x_cond  , ~small_mask)
        x_uncond_small = self._get_masked_tensor(x_uncond,  small_mask)
        x_uncond_large = self._get_masked_tensor(x_uncond, ~small_mask)

        xw_cond_small_q   = module.fwd_func(x_cond_small  , w_raw, **module.fwd_kwargs)
        xw_cond_large_q   = module.fwd_func(x_cond_large  , w_raw, **module.fwd_kwargs)
        xw_uncond_small_q = module.fwd_func(x_uncond_small, w_raw, **module.fwd_kwargs)
        xw_uncond_large_q = module.fwd_func(x_uncond_large, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_small_q + xw_cond_large_q, xw_uncond_small_q + xw_uncond_large_q], dim=0)

        # 1. Prepare Input Data
        x_cond_small   = self.get_Raw_input(module, name, x_cond_small)
        x_cond_large   = self.get_CUD_input(module, name, x_cond_large, x_uncond_large)
        x_uncond_small = self.get_Raw_input(module, name, x_uncond_small)
        x_uncond_large = self.get_Raw_input(module, name, x_uncond_large)
        tensor_in = torch.stack([x_cond_small + x_cond_large, x_uncond_small + x_uncond_large], dim=0)

        # 2. Apply bitslice reconstruction
        x_cond_small_reconstructed   = self.apply_bitslice_reconstruction(x_cond_small)
        x_cond_large_reconstructed   = self.apply_bitslice_reconstruction(x_cond_large)
        x_uncond_small_reconstructed = self.apply_bitslice_reconstruction(x_uncond_small)
        x_uncond_large_reconstructed = self.apply_bitslice_reconstruction(x_uncond_large)

        # 3. Computation
        xw_cond_small   = self.fwd_func(module, name, x_cond_small_reconstructed  , w_raw)
        xw_cond_large   = self.fwd_func(module, name, x_cond_large_reconstructed  , w_raw)
        xw_uncond_small = self.fwd_func(module, name, x_uncond_small_reconstructed, w_raw)
        xw_uncond_large = self.fwd_func(module, name, x_uncond_large_reconstructed, w_raw)
        tensor_out = torch.stack([xw_cond_small + xw_cond_large, xw_uncond_small + xw_uncond_large], dim=0)

        # 4. Output Recovery
        xw_uncond_small = self.get_Raw_output(module, name, xw_uncond_small, xw_uncond_small_q.shape)
        xw_uncond_large = self.get_Raw_output(module, name, xw_uncond_large, xw_uncond_large_q.shape)
        xw_cond_small   = self.get_Raw_output(module, name, xw_cond_small  , xw_cond_small_q.shape)
        xw_cond_large   = self.get_CUD_output(module, name, xw_cond_large, xw_uncond_large, xw_uncond_large_q.shape)
        xw_raw = torch.stack([xw_cond_small + xw_cond_large, xw_uncond_small + xw_uncond_large], dim=0)

        self.tensors = {
            "act": tensor_in,
            "output": tensor_out
        }

        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class OptimalCFGDifferenceComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, bit_th: int = 4):
        super().__init__(clamp, bitslice_strategy)
        self.bit_th = bit_th        
        logger.info(f"bit_th: {self.bit_th}")
    
    def _get_bitwidth_type(self, tensor) -> torch.Tensor:
        lower_bound = -1*2**(self.bit_th-1)
        upper_bound =    2**(self.bit_th-1)-1
        # logger.info("bit_th, lower_bound, upper_bound:", bit_th, lower_bound, upper_bound)

        small_mask = (tensor >= lower_bound) & (tensor <= upper_bound) & (tensor != 0)
        large_mask = (tensor < lower_bound) | (tensor > upper_bound)

        tensor_type = torch.zeros_like(tensor)
        tensor_type[small_mask] = 1
        tensor_type[large_mask] = 2

        del small_mask
        del large_mask

        return tensor_type

    def _get_mask(self, tensor_raw, tensor_delta):
        bitwidth_type_raw   = self._get_bitwidth_type(tensor_raw)
        bitwidth_type_delta = self._get_bitwidth_type(tensor_delta)

        mask = bitwidth_type_delta < bitwidth_type_raw

        del bitwidth_type_raw
        del bitwidth_type_delta

        return mask
    
    def _get_masked_tensor(self, tensor, mask):
        return torch.where(mask, tensor, torch.zeros_like(tensor))
    
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']    # target
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        w_raw    = params['w_q']    - params['w_zp']
        x_cond_delta = x_cond - x_uncond

        diff_mask = self._get_mask(x_cond, x_cond_delta)
        x_cond_raw     = self._get_masked_tensor(x_cond  , ~diff_mask)
        x_cond_delta   = self._get_masked_tensor(x_cond  ,  diff_mask)
        x_uncond_raw   = self._get_masked_tensor(x_uncond, ~diff_mask)
        x_uncond_delta = self._get_masked_tensor(x_uncond,  diff_mask)

        xw_cond_raw_q     = module.fwd_func(x_cond_raw    , w_raw, **module.fwd_kwargs)
        xw_cond_delta_q   = module.fwd_func(x_cond_delta  , w_raw, **module.fwd_kwargs)
        xw_uncond_raw_q   = module.fwd_func(x_uncond_raw  , w_raw, **module.fwd_kwargs)
        xw_uncond_delta_q = module.fwd_func(x_uncond_delta, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_raw_q + xw_cond_delta_q, xw_uncond_raw_q + xw_uncond_delta_q], dim=0)

        # 1. Prepare Input Data
        x_cond_raw     = self.get_Raw_input(module, name, x_cond_raw)
        x_cond_delta   = self.get_CUD_input(module, name, x_cond_delta, x_uncond_delta)
        x_uncond_raw   = self.get_Raw_input(module, name, x_uncond_raw)
        x_uncond_delta = self.get_Raw_input(module, name, x_uncond_delta)
        tensor_in = torch.stack([x_cond_raw + x_cond_delta, x_uncond_raw + x_uncond_delta], dim=0)

        # 2. Apply bitslice reconstruction
        x_cond_raw_reconstructed     = self.apply_bitslice_reconstruction(x_cond_raw)
        x_cond_delta_reconstructed   = self.apply_bitslice_reconstruction(x_cond_delta)
        x_uncond_raw_reconstructed   = self.apply_bitslice_reconstruction(x_uncond_raw)
        x_uncond_delta_reconstructed = self.apply_bitslice_reconstruction(x_uncond_delta)

        # 3. Computation
        xw_cond_raw     = self.fwd_func(module, name, x_cond_raw_reconstructed    , w_raw)
        xw_cond_delta   = self.fwd_func(module, name, x_cond_delta_reconstructed  , w_raw)
        xw_uncond_raw   = self.fwd_func(module, name, x_uncond_raw_reconstructed  , w_raw)
        xw_uncond_delta = self.fwd_func(module, name, x_uncond_delta_reconstructed, w_raw)
        tensor_out = torch.stack([xw_cond_raw + xw_cond_delta, xw_uncond_raw + xw_uncond_delta], dim=0)

        # 4. Output Recovery
        xw_uncond_raw   = self.get_Raw_output(module, name, xw_uncond_raw  , xw_uncond_raw_q.shape)
        xw_uncond_delta = self.get_Raw_output(module, name, xw_uncond_delta, xw_uncond_delta_q.shape)
        xw_cond_raw     = self.get_Raw_output(module, name, xw_cond_raw    , xw_cond_raw_q.shape)
        xw_cond_delta   = self.get_CUD_output(module, name, xw_cond_delta, xw_uncond_delta, xw_cond_delta_q.shape)
        xw_raw = torch.stack([xw_cond_raw + xw_cond_delta, xw_uncond_raw + xw_uncond_delta], dim=0)

        self.tensors = {
            "act": tensor_in,
            "output": tensor_out
        }

        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class SpatialOptimalCFGDifferenceComputation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, ref_first: bool = False, bit_th: int = 4):
        super().__init__(clamp, bitslice_strategy)
        self.ref_first = ref_first
        self.bit_th = bit_th
        logger.info(f"ref_first: {self.ref_first}")
        logger.info(f"bit_th: {self.bit_th}")
    
    def _get_bitwidth_type(self, tensor) -> torch.Tensor:
        lower_bound = -1*2**(self.bit_th-1)
        upper_bound =    2**(self.bit_th-1)-1
        # logger.info("bit_th, lower_bound, upper_bound:", bit_th, lower_bound, upper_bound)

        small_mask = (tensor >= lower_bound) & (tensor <= upper_bound) & (tensor != 0)
        large_mask = (tensor < lower_bound) | (tensor > upper_bound)

        tensor_type = torch.zeros_like(tensor)
        tensor_type[small_mask] = 1
        tensor_type[large_mask] = 2

        del small_mask
        del large_mask

        return tensor_type

    def _get_mask(self, tensor_raw, tensor_delta):
        bitwidth_type_raw   = self._get_bitwidth_type(tensor_raw)
        bitwidth_type_delta = self._get_bitwidth_type(tensor_delta)

        mask = bitwidth_type_delta < bitwidth_type_raw

        del bitwidth_type_raw
        del bitwidth_type_delta

        return mask
    
    def _get_masked_tensor(self, tensor, mask):
        return torch.where(mask, tensor, torch.zeros_like(tensor))
    
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']    # target
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        w_raw    = params['w_q']    - params['w_zp']
        x_cond_delta = x_cond - x_uncond

        diff_mask = self._get_mask(x_cond, x_cond_delta)
        x_cond_raw     = self._get_masked_tensor(x_cond  , ~diff_mask)
        x_cond_delta   = self._get_masked_tensor(x_cond  ,  diff_mask)
        x_uncond_raw   = self._get_masked_tensor(x_uncond, ~diff_mask)
        x_uncond_delta = self._get_masked_tensor(x_uncond,  diff_mask)

        xw_cond_raw_q     = module.fwd_func(x_cond_raw    , w_raw, **module.fwd_kwargs)
        xw_cond_delta_q   = module.fwd_func(x_cond_delta  , w_raw, **module.fwd_kwargs)
        xw_uncond_raw_q   = module.fwd_func(x_uncond_raw  , w_raw, **module.fwd_kwargs)
        xw_uncond_delta_q = module.fwd_func(x_uncond_delta, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_raw_q + xw_cond_delta_q, xw_uncond_raw_q + xw_uncond_delta_q], dim=0)

        # 1. Prepare Input Data
        x_cond_raw     = self.get_Raw_input(module, name, x_cond_raw)
        x_cond_delta   = self.get_CUD_input(module, name, x_cond_delta, x_uncond_delta)
        x_uncond_raw   = self.get_SD_input(module, name, x_uncond_raw  , self.ref_first)
        x_uncond_delta = self.get_SD_input(module, name, x_uncond_delta, self.ref_first)
        tensor_in = torch.stack([x_cond_raw + x_cond_delta, x_uncond_raw + x_uncond_delta], dim=0)

        # 2. Apply bitslice reconstruction
        x_cond_raw_reconstructed     = self.apply_bitslice_reconstruction(x_cond_raw)
        x_cond_delta_reconstructed   = self.apply_bitslice_reconstruction(x_cond_delta)
        x_uncond_raw_reconstructed   = self.apply_bitslice_reconstruction(x_uncond_raw)
        x_uncond_delta_reconstructed = self.apply_bitslice_reconstruction(x_uncond_delta)

        # 2. Computation
        xw_cond_raw     = self.fwd_func(module, name, x_cond_raw_reconstructed    , w_raw)
        xw_cond_delta   = self.fwd_func(module, name, x_cond_delta_reconstructed  , w_raw)
        xw_uncond_raw   = self.fwd_func(module, name, x_uncond_raw_reconstructed  , w_raw)
        xw_uncond_delta = self.fwd_func(module, name, x_uncond_delta_reconstructed, w_raw)
        tensor_out = torch.stack([xw_cond_raw + xw_cond_delta, xw_uncond_raw + xw_uncond_delta], dim=0)

        # 3. Output Recovery
        xw_uncond_raw   = self.get_SD_output(module, name, xw_uncond_raw  , xw_uncond_raw_q.shape  , self.ref_first)
        xw_uncond_delta = self.get_SD_output(module, name, xw_uncond_delta, xw_uncond_delta_q.shape, self.ref_first)
        xw_cond_raw     = self.get_Raw_output(module, name, xw_cond_raw   , xw_cond_raw_q.shape)
        xw_cond_delta   = self.get_CUD_output(module, name, xw_cond_delta, xw_uncond_delta, xw_cond_delta_q.shape)
        xw_raw = torch.stack([xw_cond_raw + xw_cond_delta, xw_uncond_raw + xw_uncond_delta], dim=0)

        self.tensors = {
            "act": tensor_in,
            "output": tensor_out
        }

        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class Raw_SD_Computation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, ref_first: bool = False, save=False):
        super().__init__(clamp, bitslice_strategy, save)
        self.ref_first = ref_first
        logger.info(f"ref_first: {self.ref_first}")

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']    # target
        x_uncond = params['x_q'][1] - params['x_zp']    # reference
        w_raw    = params['w_q']    - params['w_zp']

        xw_cond_q   = module.fwd_func(x_cond  , w_raw, **module.fwd_kwargs)
        xw_uncond_q = module.fwd_func(x_uncond, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_q, xw_uncond_q], dim=0)

        # 1. Prepare Input Data
        x_cond   = self.get_SD_input (module, name, x_cond  , self.ref_first)
        x_uncond = self.get_Raw_input(module, name, x_uncond)
        tensor_in = torch.stack([x_cond, x_uncond], dim=0)

        # 2. Apply bitslice reconstruction
        x_cond_reconstructed   = self.apply_bitslice_reconstruction(x_cond)
        x_uncond_reconstructed = self.apply_bitslice_reconstruction(x_uncond)
        # tensor_in_reconstructed = torch.stack([x_cond_reconstructed, x_uncond_reconstructed], dim=0)

        # 3. Computation
        xw_cond   = self.fwd_func(module, name, x_cond_reconstructed  , w_raw)
        xw_uncond = self.fwd_func(module, name, x_uncond_reconstructed, w_raw)
        tensor_out = torch.stack([xw_cond, xw_uncond], dim=0)

        # 4. Output Recovery
        xw_cond   = self.get_SD_output (module, name, xw_cond  , xw_cond_q.shape  , self.ref_first)
        xw_uncond = self.get_Raw_output(module, name, xw_uncond, xw_uncond_q.shape)
        xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)

        self.tensors = {
            "act": tensor_in,
            # "act": tensor_in_reconstructed,
            "output": tensor_out
        }

        if self.save:
            self.save_tensors = {
                "x_raw" : tensor_in,            # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": tensor_out,           # int24
            }
        
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class Raw_TD_Computation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, save=False):
        super().__init__(clamp, bitslice_strategy, save)
        
    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']
        x_uncond = params['x_q'][1] - params['x_zp']
        _x_cond  = params['x_q'][0] - params['x_zp']
        w_raw    = params['w_q']    - params['w_zp']

        xw_cond_q   = module.fwd_func(x_cond  , w_raw, **module.fwd_kwargs)
        xw_uncond_q = module.fwd_func(x_uncond, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_q, xw_uncond_q], dim=0)

        if hasattr(module, 'in_prev'):      # 2nd~ step
            # 1. Prepare Input Data
            x_cond   = self.get_TD_input (module, name, x_cond  , module.in_prev)
            x_uncond = self.get_Raw_input(module, name, x_uncond)
            tensor_in = torch.stack([x_cond, x_uncond], dim=0)

            # 2. Apply bitslice reconstruction
            x_cond_reconstructed   = self.apply_bitslice_reconstruction(x_cond)
            x_uncond_reconstructed = self.apply_bitslice_reconstruction(x_uncond)
            
            # 3. Computation
            xw_cond   = self.fwd_func(module, name, x_cond_reconstructed  , w_raw)
            xw_uncond = self.fwd_func(module, name, x_uncond_reconstructed, w_raw)
            tensor_out = torch.stack([xw_cond, xw_uncond], dim=0)

            # 4. Output Recovery
            xw_cond   = self.get_TD_output (module, name, xw_cond  , module.out_prev, xw_cond_q.shape  )
            xw_uncond = self.get_Raw_output(module, name, xw_uncond,                  xw_uncond_q.shape)
            xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)
        else:                               # 1st step
            # 1. Prepare Input Data
            x_cond   = self.get_Raw_input(module, name, x_cond)
            x_uncond = self.get_Raw_input(module, name, x_uncond)
            tensor_in = torch.stack([x_cond, x_uncond], dim=0)

            # 2. Apply bitslice reconstruction
            x_cond_reconstructed   = self.apply_bitslice_reconstruction(x_cond)
            x_uncond_reconstructed = self.apply_bitslice_reconstruction(x_uncond)
            
            # 3. Computation
            xw_cond   = self.fwd_func(module, name, x_cond_reconstructed  , w_raw)
            xw_uncond = self.fwd_func(module, name, x_uncond_reconstructed, w_raw)
            tensor_out = torch.stack([xw_cond, xw_uncond], dim=0)
            
            # 4. Output Recovery
            xw_cond   = self.get_Raw_output(module, name, xw_cond  , xw_cond_q.shape)
            xw_uncond = self.get_Raw_output(module, name, xw_uncond, xw_uncond_q.shape)
            xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)
        
        module.in_prev  = _x_cond
        module.out_prev = xw_cond

        self.tensors = {
            "act": tensor_in,
            "output": tensor_out
        }

        if self.save:
            self.save_tensors = {
                "x_raw" : tensor_in,            # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": tensor_out,           # int24
            }
        
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


class SD_TD_Computation(ComputationStrategy):
    def __init__(self, clamp=False, bitslice_strategy=None, ref_first: bool = False, save=False):
        super().__init__(clamp, bitslice_strategy, save)
        self.ref_first = ref_first
        logger.info(f"ref_first: {self.ref_first}")

    def compute(self, module, input, output, name) -> torch.Tensor:
        params = self._get_quantization_params(module, input)

        # 0. Golden Output
        x_cond   = params['x_q'][0] - params['x_zp']
        x_uncond = params['x_q'][1] - params['x_zp']
        _x_cond  = params['x_q'][0] - params['x_zp']
        w_raw    = params['w_q']    - params['w_zp']

        xw_cond_q   = module.fwd_func(x_cond  , w_raw, **module.fwd_kwargs)
        xw_uncond_q = module.fwd_func(x_uncond, w_raw, **module.fwd_kwargs)
        xw_q = torch.stack([xw_cond_q, xw_uncond_q], dim=0)

        if hasattr(module, 'in_prev'):      # 2nd~ step
            # 1. Prepare Input Data
            x_cond   = self.get_TD_input(module, name, x_cond  , module.in_prev)
            x_uncond = self.get_SD_input(module, name, x_uncond, self.ref_first)
            tensor_in = torch.stack([x_cond, x_uncond], dim=0)

            # 2. Apply bitslice reconstruction
            x_cond_reconstructed   = self.apply_bitslice_reconstruction(x_cond)
            x_uncond_reconstructed = self.apply_bitslice_reconstruction(x_uncond)
            
            # 3. Computation
            xw_cond   = self.fwd_func(module, name, x_cond_reconstructed  , w_raw)
            xw_uncond = self.fwd_func(module, name, x_uncond_reconstructed, w_raw)
            tensor_out = torch.stack([xw_cond, xw_uncond], dim=0)

            # 4. Output Recovery
            xw_cond   = self.get_TD_output(module, name, xw_cond  , module.out_prev, xw_cond_q.shape  )
            xw_uncond = self.get_SD_output(module, name, xw_uncond,                  xw_uncond_q.shape)
            xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)
        else:                               # 1st step
            # 1. Prepare Input Data
            x_cond   = self.get_SD_input(module, name, x_cond  , self.ref_first)
            x_uncond = self.get_SD_input(module, name, x_uncond, self.ref_first)
            tensor_in = torch.stack([x_cond, x_uncond], dim=0)

            # 2. Apply bitslice reconstruction
            x_cond_reconstructed   = self.apply_bitslice_reconstruction(x_cond)
            x_uncond_reconstructed = self.apply_bitslice_reconstruction(x_uncond)
            
            # 3. Computation
            xw_cond   = self.fwd_func(module, name, x_cond_reconstructed  , w_raw)
            xw_uncond = self.fwd_func(module, name, x_uncond_reconstructed, w_raw)
            tensor_out = torch.stack([xw_cond, xw_uncond], dim=0)
            
            # 4. Output Recovery
            xw_cond   = self.get_SD_output(module, name, xw_cond  , xw_cond_q.shape)
            xw_uncond = self.get_SD_output(module, name, xw_uncond, xw_uncond_q.shape)
            xw_raw = torch.stack([xw_cond, xw_uncond], dim=0)
        
        module.in_prev  = _x_cond
        module.out_prev = xw_cond

        self.tensors = {
            "act": tensor_in,
            "output": tensor_out
        }

        if self.save:
            self.save_tensors = {
                "x_raw" : tensor_in,            # int9: -256~255
                "w_raw" : w_raw.t(),            # int9: -256~255
                "xw_raw": tensor_out,           # int24
            }
        
        if self.should_assert:
            assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        return self._output_rescaling(module, xw_raw, params['x_scale'], params['w_scale'])


def create_strategy(strategy_type: str, bitslice_method=None, bitslice_kwargs=None, analysis_manager=None, **kwargs) -> ComputationStrategy:
    # Create bitslice strategy
    bitslice_strategy = None
    if bitslice_method and bitslice_method != "none":
        bitslice_kwargs = bitslice_kwargs or {}
        bitslice_strategy = create_bitslice_strategy(bitslice_method, **bitslice_kwargs)

    # Add to analysis manager if provided
    if analysis_manager and bitslice_strategy:
        from analyses.bitslice_analyzer import BitsliceAnalyzer
        bitslice_analyzer = BitsliceAnalyzer(bitslice_strategy)
        analysis_manager.add_analyzer("bitslice", bitslice_analyzer)

    strategies = {
        'original': OriginalComputation,
        'spatial': SpatialDifferenceComputation,
        'temporal': TemporalDifferenceComputation,
        'cfg': CFGDifferenceComputation,
        'spatial_cfg': SpatialCFGDifferenceComputation,
        'cfg_large': LargeNumbersCFGDifferenceComputation,
        'cfg_opt': OptimalCFGDifferenceComputation,
        'spatial_cfg_opt': SpatialOptimalCFGDifferenceComputation,
        'Raw_SD': Raw_SD_Computation,
        'Raw_TD': Raw_TD_Computation,
        'SD_TD': SD_TD_Computation,
    }
    
    if strategy_type not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {available}")
    
    strategy_class = strategies[strategy_type]

    # Define parameter mappings for each strategy
    strategy_params = {
        'original'       : ['clamp',                        'save'],
        'spatial'        : ['clamp', 'ref_first',           'save'],
        'temporal'       : ['clamp',                        'save'],
        'cfg'            : ['clamp',                        'save'],
        'spatial_cfg'    : ['clamp', 'ref_first',           'save'],
        'cfg_large'      : ['clamp',              'bit_th'        ],
        'cfg_opt'        : ['clamp',              'bit_th'        ],
        'spatial_cfg_opt': ['clamp', 'ref_first', 'bit_th'        ],
        'Raw_SD'         : ['clamp', 'ref_first',           'save'],
        'Raw_TD'         : ['clamp',                        'save'],
        'SD_TD'          : ['clamp', 'ref_first',           'save'],
    }

    # Get allowed parameters for this strategy (default to just 'clamp')
    allowed_params = strategy_params.get(strategy_type, ['clamp'])
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
    return strategy_class(bitslice_strategy=bitslice_strategy, **filtered_kwargs)

