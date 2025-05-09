### This is a test script to check the correctness of spatial difference computing

import torch
import torch.nn as nn
import torch.nn.functional as f

torch.manual_seed(0)

input_fp = torch.randn(2, 256, 1024)
weight_fp = torch.randn(512, 1024)

def get_quant_param(x):
    range_i = torch.max(x) - torch.min(x)
    d = range_i / 255
    z = torch.min(torch.round(x / d)) * -1
    return d, z

da, za = get_quant_param(input_fp)
dw, zw = get_quant_param(weight_fp)

input_int = torch.round(input_fp / da) + za
input_qfp = (input_int - za) * da

weight_int = torch.round(weight_fp / dw) + zw
weight_qfp = (weight_int - zw) * dw

### 1. qfp computing
output_golden = f.linear(input_qfp, weight_qfp)


### 2. int computing
# out_int = f.linear(input_int - za, weight_int - zw)
# out = out_int * da * dw

### 3. int computing with diff
out_int = torch.zeros(2, input_int.shape[1], weight_int.shape[0])

out_int[:, 0, :] = f.linear(input_int[:, 0, :] - za, weight_int - zw)

# for i in range(1, input_int.shape[1]):
#     input_delta = input_int[:, i, :] - input_int[:, i-1, :]
#     out_delta = f.linear(input_delta, weight_int - zw)
#     out_int[:, i, :] = out_delta + out_int[:, i-1, :]
input_delta = input_int[:, 1:, :] - input_int[:, :-1, :]
out_delta = f.linear(input_delta, weight_int - zw)
out_int[:, 1:, :] = torch.cumsum(out_delta, dim=1)
out_int[:, 1:, :] += out_int[:, 0:1, :]


out = out_int * da * dw

### Check computation correctness
print(output_golden[0, 0:5, 0:5])
print(out[0, 0:5, 0:5])
if torch.allclose(output_golden, out, atol=1e-4):
    print("Identical")
else:
    print("Different")

out_diff = output_golden - out
print(f"Max difference: {torch.max(torch.abs(out_diff)).item()}")