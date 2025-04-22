# fp8_layer.py — Simulated FP8 Linear Layer (Quantized MLP)

import torch
import torch.nn as nn

class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(FP8Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def quantize_to_fp8(self, tensor):
        scale = 127.0 / tensor.abs().max().clamp(min=1e-6)
        tensor_fp8 = torch.clamp((tensor * scale).round(), -127, 127)
        return tensor_fp8, scale

    def dequantize_from_fp8(self, tensor_fp8, scale):
        return tensor_fp8 / scale

    def forward(self, x):
        x_fp8, scale = self.quantize_to_fp8(x)
        x_dequant = self.dequantize_from_fp8(x_fp8, scale)
        return self.linear(x_dequant)
