from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn.functional as F


class ComputationStrategy(ABC):
    def __init__(self):
        self.tensors = {}

    def fwd_func(self, module, name, input, weight):
        if module.fwd_func == F.conv2d:
            w_col = weight.clone()
            w_col = w_col.view(w_col.shape[0], -1)
            output = torch.matmul(w_col, input)
        elif module.fwd_func == F.linear:
            output = module.fwd_func(input, weight, **module.fwd_kwargs)
        else:
            raise Exception(f"Unsupported fwd_func in {name}")
        
        return output
    
    def get_Raw_input(self, module, name, tensor):
        tensor = tensor.clone()

        if module.fwd_func == F.conv2d:
            fold_params = self._get_fold_params(module)
            tensor = F.unfold(tensor, **fold_params)
        elif module.fwd_func == F.linear:
            pass
        else:
            raise Exception(f"Unsupported fwd_func in {name}")

        return tensor

    def get_Raw_output(self, module, name, tensor, shape):
        tensor = tensor.clone()

        if module.fwd_func == F.conv2d:
            tensor = tensor.view(shape)
        elif module.fwd_func == F.linear:
            pass
        else:
            raise Exception(f"Unsupported fwd_func in {name}")

        return tensor
    
    def get_SD_input(self, module, name, tensor, ref_first=False):
        tensor = tensor.clone()

        if module.fwd_func == F.conv2d:
            fold_params = self._get_fold_params(module)
            tensor = F.unfold(tensor, **fold_params)

            if ref_first:
                tensor[..., 1:] -= tensor[..., 0:1]
            else:
                tensor_tmp = torch.roll(tensor, 1, dims=-1)
                tensor_tmp[..., 0:1] = 0
                tensor -= tensor_tmp
                del tensor_tmp
        elif module.fwd_func == F.linear:
            if tensor.dim() in [2, 3]:
                if ref_first:
                    tensor[..., 1:, :] -= tensor[..., 0:1, :]
                else:
                    tensor_tmp = torch.roll(tensor, 1, dims=-2)
                    tensor_tmp[..., 0:1, :] = 0
                    tensor -= tensor_tmp
                    del tensor_tmp
            else:
                raise Exception(f"Unsupported input dimension {tensor.dim()} in {name}")
        else:
            raise Exception(f"Unsupported fwd_func in {name}")

        return tensor

    def get_SD_output(self, module, name, tensor, shape, ref_first=False):
        tensor = tensor.clone()

        if module.fwd_func == F.conv2d:
            if ref_first:
                tensor[..., 1:]  += tensor[..., 0:1]
            else:
                tensor = torch.cumsum(tensor, dim=-1)
            tensor = tensor.view(shape)
        elif module.fwd_func == F.linear:
            if tensor.dim() in [2, 3]:
                if ref_first:
                    tensor[..., 1:, :] += tensor[..., 0:1, :]
                else:
                    tensor = torch.cumsum(tensor, dim=-2)
            else:
                raise Exception(f"Unsupported input dimension {tensor.dim()} in {name}")
        else:
            raise Exception(f"Unsupported fwd_func in {name}")
        
        return tensor
    
    def get_TD_input(self, module, name, tensor, tensor_prev):
        tensor = tensor.clone()
        tensor -= tensor_prev

        if module.fwd_func == F.conv2d:
            fold_params = self._get_fold_params(module)
            tensor = F.unfold(tensor, **fold_params)
        elif module.fwd_func == F.linear:
            pass
        else:
            raise Exception(f"Unsupported fwd_func in {name}")
        
        return tensor
    
    def get_TD_output(self, module, name, tensor, tensor_prev, shape):
        tensor = tensor.clone()

        if module.fwd_func == F.conv2d:
            tensor = tensor.view(shape)
        elif module.fwd_func == F.linear:
            pass
        else:
            raise Exception(f"Unsupported fwd_func in {name}")

        tensor += tensor_prev        
        return tensor
    
    def get_CUD_input(self, module, name, tensor, tensor_ref):
        tensor = tensor.clone()
        tensor -= tensor_ref

        if module.fwd_func == F.conv2d:
            fold_params = self._get_fold_params(module)
            tensor = F.unfold(tensor, **fold_params)
        elif module.fwd_func == F.linear:
            pass
        else:
            raise Exception(f"Unsupported fwd_func in {name}")
        
        return tensor

    def get_CUD_output(self, module, name, tensor, tensor_ref, shape):
        tensor = tensor.clone()

        if module.fwd_func == F.conv2d:
            tensor = tensor.view(shape)
        elif module.fwd_func == F.linear:
            pass
        else:
            raise Exception(f"Unsupported fwd_func in {name}")

        tensor += tensor_ref        
        return tensor
    
    def _get_quantization_params(self, module, input):
        x_scale = module.act_quantizer.delta
        w_scale = module.weight_quantizer.delta
        x_zp = module.act_quantizer.zero_point
        w_zp = module.weight_quantizer.zero_point

        x_dq = module.act_quantizer(input[0])
        w_dq = module.weight_quantizer(module.weight)

        x_q = torch.round(x_dq / x_scale) + x_zp
        w_q = torch.round(w_dq / w_scale) + w_zp
        
        return {
            'x_scale': x_scale,
            'w_scale': w_scale,
            'x_zp': x_zp,
            'w_zp': w_zp,
            'x_q': x_q,
            'w_q': w_q
        }

    def _get_fold_params(self, module):
        if module.fwd_func != F.conv2d:
            raise ValueError("fold_params only applicable for conv2d layers")
        
        kernel_h, kernel_w = module.weight.shape[2], module.weight.shape[3]
        
        stride   = module.fwd_kwargs.get('stride'  , 1)
        padding  = module.fwd_kwargs.get('padding' , 0)
        dilation = module.fwd_kwargs.get('dilation', 1)
        
        # Handle tuple arguments
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
            
        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding
            
        if isinstance(dilation, int):
            dil_h = dil_w = dilation
        else:
            dil_h, dil_w = dilation
        
        return dict(kernel_size=(kernel_h, kernel_w), 
                    stride=(stride_h, stride_w), 
                    padding=(pad_h, pad_w), 
                    dilation=(dil_h, dil_w))
    
    def _output_rescaling(self, module, xw_q, x_scale, w_scale):
        if module.fwd_func == F.conv2d:
            y_dq = xw_q * x_scale * w_scale.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif module.fwd_func == F.linear:
            y_dq = xw_q * x_scale * w_scale.permute(1, 0) + module.bias
        else:
            raise Exception("Unsupported fwd_func")
            
        return module.activation_function(y_dq)

    @abstractmethod
    def compute(self, module, input, output, name) -> torch.Tensor:
        pass

    @abstractmethod
    def get_tensors_to_analyze(self) -> Dict[str, torch.Tensor]:
        pass