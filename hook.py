import torch
import torch.nn as nn

def hook_temporal_reuse(module, input, output):
    # add timestep indicator
    if hasattr(module, 'timestep'):
        module.timestep += 1
    else:
        module.timestep = 0

    input_qfp = module.act_quantizer(input[0])
    weight_qfp = module.weight_quantizer(module.weight)

    input_int = torch.round(input_qfp / module.act_quantizer.delta) + module.act_quantizer.zero_point
    weight_int = torch.round(weight_qfp / module.weight_quantizer.delta) + module.weight_quantizer.zero_point

    da = module.act_quantizer.delta
    za = module.act_quantizer.zero_point
    dw = module.weight_quantizer.delta
    zw = module.weight_quantizer.zero_point

    if hasattr(module, 'input_prev'):  # 2nd~ timestep
        input_delta = input_int - module.input_prev
        module.input_prev = input_int

        out_delta = module.fwd_func(input_delta, weight_int - zw, **module.fwd_kwargs)

        out_int = out_delta + module.output_prev
        module.output_prev = out_int
    else:  # 1st timestep
        module.input_prev = input_int
        out_int = module.fwd_func(input_int - za, weight_int - zw, **module.fwd_kwargs)
        module.output_prev = out_int

    if module.fwd_func == nn.functional.conv2d:
        out = out_int * da * dw.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif module.fwd_func == nn.functional.linear:
        out = out_int * da * dw.permute(1, 0) + module.bias
    else:
        raise Exception("Unsupported fwd_func")

    out = module.activation_function(out)

    return out


def hook_cfg_reuse(module, input, output):
    # add timestep indicator
    if hasattr(module, 'timestep'):
        module.timestep += 1
    else:
        module.timestep = 0

    da = module.act_quantizer.delta
    za = module.act_quantizer.zero_point
    dw = module.weight_quantizer.delta
    zw = module.weight_quantizer.zero_point

    input_qfp = module.act_quantizer(input[0])
    weight_qfp = module.weight_quantizer(module.weight)

    input_int = torch.round(input_qfp / module.act_quantizer.delta) + module.act_quantizer.zero_point
    weight_int = torch.round(weight_qfp / module.weight_quantizer.delta) + module.weight_quantizer.zero_point

    ###### integer computing
    # out_int = module.fwd_func(input_int - za, weight_int - zw, **module.fwd_kwargs)

    ###### diff computing
    input_int_org = input_int[0]
    input_int_cfg = input_int[1] - input_int[0]

    out_int_org = module.fwd_func(input_int_org - za, weight_int - zw, **module.fwd_kwargs)
    out_int_cfg = module.fwd_func(input_int_cfg, weight_int - zw, **module.fwd_kwargs)
    out_int_cfg = out_int_cfg + out_int_org
    
    out_int = torch.stack([out_int_org, out_int_cfg], dim=0)
    #######################################

    if module.fwd_func == nn.functional.conv2d:
        out = out_int * da * dw.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    elif module.fwd_func == nn.functional.linear:
        out = out_int * da * dw.permute(1, 0) + module.bias
    else:
        raise Exception("Unsupported fwd_func")

    out = module.activation_function(out)

    return out