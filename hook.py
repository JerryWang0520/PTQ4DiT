import torch
import torch.nn.functional as F
from collections import defaultdict

from analysis import count_minimum_bitwidth

def hook_original(module, input, output, name, dequant=False, quant=True, bias_folding=False):
    # add step indicator
    if hasattr(module, 'step'):
        module.step += 1
    else:
        module.step = 1
    
    if not hasattr(module, 'counts'):
        module.counts = defaultdict(dict)
    
    x_scale = module.act_quantizer.delta
    w_scale = module.weight_quantizer.delta
    x_zp = module.act_quantizer.zero_point
    w_zp = module.weight_quantizer.zero_point

    x_dq = module.act_quantizer(input[0])
    w_dq = module.weight_quantizer(module.weight)

    x_q = torch.round(x_dq / x_scale) + x_zp
    w_q = torch.round(w_dq / w_scale) + w_zp

    # Naive dequant computation (no error)
    if dequant:
        y_dq = module.fwd_func(x_dq, w_dq, module.bias, **module.fwd_kwargs)
        out_dq = module.activation_function(y_dq)

    # Quant computation (with errors)
    if quant:
        xw_q = module.fwd_func(x_q - x_zp, w_q - w_zp, **module.fwd_kwargs)
        if module.fwd_func == F.conv2d:
            y_dq = xw_q * x_scale * w_scale.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif module.fwd_func == F.linear:
            y_dq = xw_q * x_scale * w_scale.permute(1, 0) + module.bias
        else:
            raise Exception("Unsupported fwd_func")
        out_dq = module.activation_function(y_dq)

    # assert torch.equal(out_dq, output), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"
    # assert torch.allclose(out_dq, output, atol=1e-1, rtol=1e-1), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"
    assert torch.allclose(out_dq, output, atol=1e-3, rtol=1e-3), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"

    return output   # pass golden
    # return out_dq   # pass raw

def hook_SD(module, input, output, name, ref_first=False):
    # add step indicator
    if hasattr(module, 'step'):
        module.step += 1
    else:
        module.step = 1
    
    if not hasattr(module, 'counts'):
        module.counts = defaultdict(dict)

    x_scale = module.act_quantizer.delta
    w_scale = module.weight_quantizer.delta
    x_zp = module.act_quantizer.zero_point
    w_zp = module.weight_quantizer.zero_point

    x_dq = module.act_quantizer(input[0])
    w_dq = module.weight_quantizer(module.weight)

    x_q = torch.round(x_dq / x_scale) + x_zp
    w_q = torch.round(w_dq / w_scale) + w_zp

    xw_q = module.fwd_func(x_q - x_zp, w_q - w_zp, **module.fwd_kwargs)

    if module.fwd_func == F.conv2d:
        fold_params = get_fold_params(module)
        x_delta = F.unfold(x_q - x_zp, **fold_params)
        if ref_first:
            x_delta[:, :, 1:] -= x_delta[:, :, 0:1]
        else:
            x_prev = torch.roll(x_delta, 1, dims=-1)
            x_prev[:, :, 0:1] = 0
            x_delta -= x_prev
        
        w_col = w_q - w_zp
        w_col = w_col.view(xw_q.shape[1], -1)
        xw_raw = torch.matmul(w_col, x_delta)   # differential computing

        if ref_first:
            xw_raw[:, :, 1:]  += xw_raw[:, :, 0:1]
        else:
            xw_raw = torch.cumsum(xw_raw, dim=-1)
        
        xw_raw = xw_raw.view(xw_q.shape[0], xw_q.shape[1], xw_q.shape[2], xw_q.shape[3])

        # module.counts["act"   ][module.step] = count_minimum_bitwidth(col       , signed=True, max_bits=10)
        # module.counts["weight"][module.step] = count_minimum_bitwidth(weight_col, signed=True, max_bits=10)

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        y_dq = xw_q * x_scale * w_scale.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif module.fwd_func == F.linear:
        x_delta = x_q
        if x_delta.dim() in [2, 3]:
            if ref_first:
                x_delta[..., 1:, :] -= x_delta[..., 0:1, :]
                x_delta[..., 0:1, :] -= x_zp

                xw_raw = module.fwd_func(x_delta, w_q - w_zp, **module.fwd_kwargs)
                xw_raw[..., 1:, :] += xw_raw[..., 0:1, :]
            else:
                x_prev = torch.roll(x_delta, 1, dims=-2)
                x_prev[..., 0:1, :] = x_zp
                x_delta -= x_prev

                xw_raw = module.fwd_func(x_delta, w_q - w_zp, **module.fwd_kwargs)
                xw_raw = torch.cumsum(xw_raw, dim=-2)
        else:
            raise Exception(f"Unsupported input dimension {x_delta.dim()}")

        # module.counts["act"   ][module.step] = count_minimum_bitwidth(x_delta   , signed=True, max_bits=10)
        # module.counts["weight"][module.step] = count_minimum_bitwidth(w_q - w_zp, signed=True, max_bits=10)

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        y_dq = xw_q * x_scale * w_scale.permute(1, 0) + module.bias
    else:
        raise Exception("Unsupported fwd_func")
    
    out_dq = module.activation_function(y_dq)

    # assert torch.equal(out_dq, output), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"
    assert torch.allclose(out_dq, output, atol=1e-3, rtol=1e-3), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"

    return output   # pass golden
    # return out_dq   # pass raw

def hook_TD(module, input, output, name):
    # add step indicator
    if hasattr(module, 'step'):
        module.step += 1
    else:
        module.step = 1

    x_scale = module.act_quantizer.delta
    w_scale = module.weight_quantizer.delta
    x_zp = module.act_quantizer.zero_point
    w_zp = module.weight_quantizer.zero_point

    x_dq = module.act_quantizer(input[0])
    w_dq = module.weight_quantizer(module.weight)

    x_q = torch.round(x_dq / x_scale) + x_zp
    w_q = torch.round(w_dq / w_scale) + w_zp

    xw_q = module.fwd_func(x_q - x_zp, w_q - w_zp, **module.fwd_kwargs)
    if hasattr(module, 'x_prev'):  # 2nd~ step
        x_delta = x_q - module.x_prev
        xw_delta = module.fwd_func(x_delta, w_q - w_zp, **module.fwd_kwargs)
        xw_raw   = xw_delta + module.output_prev
    else:  # 1st step
        xw_raw = xw_q
    
    module.x_prev  = x_q
    module.output_prev = xw_raw

    assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
    if module.fwd_func == F.conv2d:
        y_dq = xw_q * x_scale * w_scale.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif module.fwd_func == F.linear:
        y_dq = xw_q * x_scale * w_scale.permute(1, 0) + module.bias
    else:
        raise Exception("Unsupported fwd_func")
    
    out_dq = module.activation_function(y_dq)

    # assert torch.equal(out_dq, output), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"
    assert torch.allclose(out_dq, output, atol=1e-3, rtol=1e-3), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"

    return output   # pass golden
    # return out_dq   # pass raw


def hook_CUD(module, input, output, name):
    # add step indicator
    if hasattr(module, 'step'):
        module.step += 1
    else:
        module.step = 1

    x_scale = module.act_quantizer.delta
    w_scale = module.weight_quantizer.delta
    x_zp = module.act_quantizer.zero_point
    w_zp = module.weight_quantizer.zero_point

    x_dq = module.act_quantizer(input[0])
    w_dq = module.weight_quantizer(module.weight)

    x_q = torch.round(x_dq / x_scale) + x_zp
    w_q = torch.round(w_dq / w_scale) + w_zp

    xw_q = module.fwd_func(x_q - x_zp, w_q - w_zp, **module.fwd_kwargs)

    x_cond   = x_q[0] - x_zp
    x_uncond = x_q[1] - x_zp
    x_delta  = x_cond - x_uncond

    xw_cond   = module.fwd_func(x_cond  , w_q - w_zp, **module.fwd_kwargs)
    xw_uncond = module.fwd_func(x_uncond, w_q - w_zp, **module.fwd_kwargs)
    xw_delta  = module.fwd_func(x_delta , w_q - w_zp, **module.fwd_kwargs)

    xw_raw = xw_delta + xw_uncond    
    xw_raw = torch.stack([xw_raw, xw_uncond], dim=0)  # pass raw

    assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
    if module.fwd_func == F.conv2d:
        y_dq = xw_q * x_scale * w_scale.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif module.fwd_func == F.linear:
        y_dq = xw_q * x_scale * w_scale.permute(1, 0) + module.bias
    else:
        raise Exception("Unsupported fwd_func")

    out_dq = module.activation_function(y_dq)

    # assert torch.equal(out_dq, output), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"
    assert torch.allclose(out_dq, output, atol=1e-3, rtol=1e-3), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"

    return output   # pass golden
    # return out_dq   # pass raw

def hook_quant_info(module, input, output, name, output_dir="output"):
    import os
    import csv

    os.makedirs(output_dir, exist_ok=True)
    
    # add step indicator
    if hasattr(module, 'step'):
        module.step += 1
    else:
        module.step = 1
    
    x_dq = module.act_quantizer(input[0])
    weight_qfp = module.weight_quantizer(module.weight)

    input_int = torch.round(input_qfp / module.act_quantizer.delta) + module.act_quantizer.zero_point
    weight_int = torch.round(weight_qfp / module.weight_quantizer.delta) + module.weight_quantizer.zero_point

    def collect_and_save_quant_stats(quant, quantizer, quant_type):
        quant_info = {}
        delta = quantizer.delta
        zp = quantizer.zero_point
        quant_centered = quant - zp

        # Common information
        quant_info['step'] = module.step
        quant_info['module_name'] = name
        quant_info['quant_type'] = quant_type

        # Quantized values information
        quant_info['quant_min'] = float(quant.min())
        quant_info['quant_max'] = float(quant.max())
        quant_info['quant_shape'] = list(quant.shape)
            
        # Centered values information
        quant_info['quant_centered_min'] = float(quant_centered.min())
        quant_info['quant_centered_max'] = float(quant_centered.max())
        quant_info['quant_centered_shape'] = list(quant_centered.shape)

        # Delta information
        if isinstance(delta, torch.Tensor):
            if delta.dim() > 0:
                quant_info['delta_min'] = float(delta.min())
                quant_info['delta_max'] = float(delta.max())
                quant_info['delta_shape'] = list(delta.shape)
            else:
                quant_info['delta_value'] = float(delta)
                quant_info['delta_shape'] = "scalar_tensor"
        else:
            quant_info['delta_value'] = float(delta)
            quant_info['delta_shape'] = "scalar"

        # Zero point information
        if isinstance(zp, torch.Tensor):
            if zp.dim() > 0:
                quant_info['zp_min'] = float(zp.min())
                quant_info['zp_max'] = float(zp.max())
                quant_info['zp_shape'] = list(zp.shape)
            else:
                quant_info['zp_value'] = float(zp)
                quant_info['zp_shape'] = "scalar_tensor"
        else:
            quant_info['zp_value'] = float(zp)
            quant_info['zp_shape'] = "scalar"

        # Quantizer properties
        quant_info['sym'] = quantizer.sym
        
        # Save to CSV
        csv_filename = os.path.join(output_dir, f"quant_info_{quant_type}.csv")
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=quant_info.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(quant_info)

    # Process activations and weights
    collect_and_save_quant_stats(input_int, module.act_quantizer, "act")
    if module.step == 1:
        collect_and_save_quant_stats(weight_int, module.weight_quantizer, "weight")

def get_fold_params(module):
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