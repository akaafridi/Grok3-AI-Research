import torch
from src.moe_layer import MoELayer

def test_moe_layer_shape():
    batch_size = 4
    input_dim = 32
    hidden_dim = 64
    x = torch.randn(batch_size, input_dim)
    
    model = MoELayer(input_dim=input_dim, hidden_dim=hidden_dim, num_experts=4, top_k=2)
    output = model(x)

    assert output.shape == x.shape, "Output shape should match input shape"

def test_moe_layer_grad():
    input_dim = 32
    hidden_dim = 64
    x = torch.randn(4, input_dim, requires_grad=True)
    model = MoELayer(input_dim, hidden_dim)
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradient should be calculated"
