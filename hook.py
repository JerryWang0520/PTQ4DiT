import torch
import torch.nn.functional as F

def hook_original(module, input, output, name, dequant=False, quant=False, bias_folding=False):
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

"""
    # Bias folding (with errors)
    if bias_folding:
        N = module.weight.shape[1]                                               # in_features
        sum_wq = torch.sum(w_q, dim=1 if module.fwd_func == F.linear else (1, 2, 3))  # per-output-channel sum
        # print("w_q shape:", w_q.shape)
        # print("sum_wq shape:", sum_wq.shape)
        scale = (x_scale * w_scale).squeeze()                                    # [out_features]
        # print("w_scale shape:", w_scale.shape)
        # print("scale shape:", scale.shape)
        w_zp = w_zp.squeeze()                                                    # [out_features]
        # print("w_zp shape:", w_zp.shape)
        bias_folded = module.bias - scale * (x_zp * sum_wq - x_zp * w_zp * N)    # [out_features]
        # print("bias shape:", module.bias.shape)
        # print("bias_folded shape:", bias_folded.shape)

        xw_q = module.fwd_func(x_q, w_q, **module.fwd_kwargs)
        print("xw_q shape:", xw_q.shape)
        if module.fwd_func == F.conv2d:
            # scale = (x_scale * w_scale).view(1, -1, 1, 1)   # [1, out_channels, 1, 1]
            # print(scale.shape)
            # bias_folded = bias_folded.view(1, -1, 1, 1)     # [1, out_channels, 1, 1]
            # print(bias_folded.shape)

            batch_size, in_channels, height, width = x_q.shape
            out_channels, _, kernel_h, kernel_w = w_q.shape
            
            sum_xq = torch.sum(x_q, dim=1, keepdim=True)
            ones_kernel = torch.ones(out_channels, 1, kernel_h, kernel_w, device=x_q.device)
            
            # Convolve sum_xq with the ones kernel to get the summed values that affect each output
            zp_summed = F.conv2d(
                sum_xq,  # [2, 1, 32, 32]
                ones_kernel,  # [1152, 1, 2, 2]
                stride=module.fwd_kwargs.get('stride', 1),
                padding=module.fwd_kwargs.get('padding', 0),
                dilation=module.fwd_kwargs.get('dilation', 1),
                groups=1
            )
            
            # Step 5: Apply weight zero point
            w_zp_reshaped = w_zp.reshape(1, -1, 1, 1)  # [1, 1152, 1, 1]
            zp_compensation = zp_summed * w_zp_reshaped  # [2, 1152, H_out, W_out]

            # Step 6: Apply scale and bias
            scale_reshaped = scale.reshape(1, -1, 1, 1)  # [1, 1152, 1, 1]
            bias_folded_reshaped = bias_folded.reshape(1, -1, 1, 1)  # [1, 1152, 1, 1]

            # Final computation
            y_dq = scale_reshaped * (xw_q - zp_compensation) + bias_folded_reshaped
        elif module.fwd_func == F.linear:   # dim = 3
            if x_q.dim() in [2, 3]:
                x_q_orgin = x_q
                if x_q_orgin.dim() == 2:
                    x_q = x_q.unsqueeze(1)      # Add seq_len=1 dimension
                    xw_q = xw_q.unsqueeze(1)    # Add seq_len=1 dimension
                    # print("x_q shape:", x_q.shape)
                    # print("xw_q shape:", xw_q.shape)
                
                batch_size = x_q.shape[0]
                seq_len = x_q.shape[1]
                
                scale = (x_scale * w_scale).reshape(1, 1, -1)                       # [1, 1, out_features]
                bias_folded = bias_folded.reshape(1, 1, -1)                         # [1, 1, out_features]
                # print("scale shape:", scale.shape)
                # print("bias_folded shape:", bias_folded.shape)
                
                sum_xq = torch.sum(x_q, dim=-1).reshape(batch_size, seq_len, 1)     # [batch, seq_len, 1]
                # print("sum_xq shape:", sum_xq.shape)
                w_zp = w_zp.reshape(1, 1, -1)                                       # [1, 1, out_features]
                # print("w_zp shape:", w_zp.shape)
                zp_compensation = sum_xq * w_zp                                     # [batch, seq_len, out_features]
                # print("zp_compensation shape:", zp_compensation.shape)
                
                y_dq = scale * (xw_q - zp_compensation) + bias_folded               # [batch, seq_len, out_features]
                # print("y_dq shape:", y_dq.shape)


                if x_q_orgin.dim() == 2:
                    y_dq = y_dq.squeeze(1)  # shape: [batch_size, out_features]
                    # print("y_dq shape:", y_dq.shape)
            else:
                raise Exception(f"Unsupported input dimension {x_q.dim()}")
        else:
            raise Exception("Unsupported fwd_func")

        out_dq = module.activation_function(y_dq)
"""

    # assert torch.equal(out_dq, output), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"
    assert torch.allclose(out_dq, output, atol=1e-1, rtol=1e-1), f"out_dq != output in {name}: max diff = {torch.max(torch.abs(out_dq - output))}"

    return output   # pass golden
    # return out_dq   # pass raw

def hook_SD(module, input, output, name, ref_first=False):
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
        y_dq = xw_q * x_scale * w_scale.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif module.fwd_func == F.linear:
        x_delta = x_q
        if x_delta.dim() in [2, 3]:
            # slice(start, limit, step)
            slice_all   = slice(None)                       # slice(None, None, None),
            slice_first = slice(0, 1)                       # slice(0, 1, None)
            slice_rest  = slice(1, None)                    # slice(1, None, None)

            prefix = (slice_all,) * (x_delta.dim() - 2)     # ()                                              / (slice(None, None, None),)
            first_row = prefix + (slice_first, slice_all)   # (slice(0, 1, None), slice(None, None, None))    / (slice(None, None, None), slice(0, 1, None), slice(None, None, None))
            other_row = prefix + (slice_rest, slice_all)    # (slice(1, None, None), slice(None, None, None)) / (slice(None, None, None), slice(1, None, None), slice(None, None, None))

            if ref_first:   # all other rows are substracted by the first row
                x_delta[other_row] -= x_delta[first_row]
                x_delta[first_row] -= x_zp

                xw_raw = module.fwd_func(x_delta, w_q - w_zp, **module.fwd_kwargs)
                xw_raw[other_row] += xw_raw[first_row]
            else:           # every row is substracted by the previous row expect the first row
                x_prev = torch.roll(x_delta, 1, dims=-2)
                x_prev[first_row] = x_zp
                x_delta -= x_prev

                xw_raw = module.fwd_func(x_delta, w_q - w_zp, **module.fwd_kwargs)
                xw_raw = torch.cumsum(xw_raw, dim=-2)
        else:
            raise Exception(f"Unsupported input dimension {x_delta.dim()}")

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

    if hasattr(module, 'x_prev'):  # 2nd~ step
        x_delta = x_q - module.x_prev

        xw_q     = module.fwd_func(x_q - x_zp, w_q - w_zp, **module.fwd_kwargs)
        xw_delta = module.fwd_func(x_delta   , w_q - w_zp, **module.fwd_kwargs)
        xw_raw   = xw_delta + module.output_prev

        assert torch.equal(xw_raw, xw_q), f"xw_raw != xw_q in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_q))}"
        
        module.x_prev  = x_q
        module.output_prev = xw_raw
    else:  # 1st step
        xw_q = module.fwd_func(x_q - x_zp, w_q - w_zp, **module.fwd_kwargs)

        module.x_prev  = x_q
        module.output_prev = xw_q

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

    x_cond   = x_q[0]
    x_uncond = x_q[1]
    x_delta  = x_cond - x_uncond

    xw_cond   = module.fwd_func(x_cond   - x_zp, w_q - w_zp, **module.fwd_kwargs)
    xw_uncond = module.fwd_func(x_uncond - x_zp, w_q - w_zp, **module.fwd_kwargs)
    xw_delta  = module.fwd_func(x_delta        , w_q - w_zp, **module.fwd_kwargs)
    xw_raw    = xw_delta + xw_uncond

    assert torch.equal(xw_raw, xw_cond), f"xw_raw != xw_cond in {name}: max diff = {torch.max(torch.abs(xw_raw - xw_cond))}"
    
    xw_q = torch.stack([xw_cond, xw_uncond], dim=0)  # pass golden
    # xw_q = torch.stack([xw_raw,  xw_uncond], dim=0)  # pass raw

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