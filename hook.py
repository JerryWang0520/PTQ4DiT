import torch
import torch.nn as nn

def hook_temporal_reuse(module, input, output, name):
        # add step indicator
        if hasattr(module, 'step'):
            module.step += 1
        else:
            module.step = 1

        input_qfp = module.act_quantizer(input[0])
        weight_qfp = module.weight_quantizer(module.weight)

        input_int = torch.round(input_qfp / module.act_quantizer.delta) + module.act_quantizer.zero_point
        weight_int = torch.round(weight_qfp / module.weight_quantizer.delta) + module.weight_quantizer.zero_point

        da = module.act_quantizer.delta
        za = module.act_quantizer.zero_point
        dw = module.weight_quantizer.delta
        zw = module.weight_quantizer.zero_point

        if hasattr(module, 'input_prev'):  # 2nd~ step
            input_delta = input_int - module.input_prev

            out_int   = module.fwd_func(input_int - za, weight_int - zw, **module.fwd_kwargs)
            out_delta = module.fwd_func(input_delta   , weight_int - zw, **module.fwd_kwargs)
            out_raw   = out_delta + module.output_prev

            assert torch.equal(out_raw, out_int), f"raw != org in {name}"
            
            module.input_prev  = input_int
            module.output_prev = out_raw
        else:  # 1st step
            out_int = module.fwd_func(input_int - za, weight_int - zw, **module.fwd_kwargs)

            module.input_prev  = input_int
            module.output_prev = out_int

        if module.fwd_func == nn.functional.conv2d:
            out = out_int * da * dw.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif module.fwd_func == nn.functional.linear:
            out = out_int * da * dw.permute(1, 0) + module.bias
        else:
            raise Exception("Unsupported fwd_func")

        out = module.activation_function(out)

        return out


def hook_cfg_reuse(module, input, output, name):
        # add step indicator
        if hasattr(module, 'step'):
            module.step += 1
        else:
            module.step = 1

        input_qfp = module.act_quantizer(input[0])
        weight_qfp = module.weight_quantizer(module.weight)

        input_int = torch.round(input_qfp / module.act_quantizer.delta) + module.act_quantizer.zero_point
        weight_int = torch.round(weight_qfp / module.weight_quantizer.delta) + module.weight_quantizer.zero_point

        da = module.act_quantizer.delta
        za = module.act_quantizer.zero_point
        dw = module.weight_quantizer.delta
        zw = module.weight_quantizer.zero_point

        ###### integer computing
        # out_int = module.fwd_func(input_int - za, weight_int - zw, **module.fwd_kwargs)

        ###### diff computing
        input_int_cond   = input_int[0]
        input_int_uncond = input_int[1]
        input_int_diff   = input_int_cond - input_int_uncond

        out_int_cond   = module.fwd_func(input_int_cond   - za, weight_int - zw, **module.fwd_kwargs)
        out_int_uncond = module.fwd_func(input_int_uncond - za, weight_int - zw, **module.fwd_kwargs)
        out_int_delta  = module.fwd_func(input_int_diff       , weight_int - zw, **module.fwd_kwargs)
        out_int_raw    = out_int_delta + out_int_uncond

        assert torch.equal(out_int_raw, out_int_cond), f"raw != org in {name}"
        
        out_int = torch.stack([out_int_cond, out_int_uncond], dim=0)
        #######################################

        if module.fwd_func == nn.functional.conv2d:
            out = out_int * da * dw.permute(1, 0, 2, 3) + module.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif module.fwd_func == nn.functional.linear:
            out = out_int * da * dw.permute(1, 0) + module.bias
        else:
            raise Exception("Unsupported fwd_func")

        out = module.activation_function(out)

        return out