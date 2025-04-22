# train_grok3.py — Simulated Training Loop for Grok‑3 MoE

import torch
import torch.nn as nn
from moe_layer import MoELayer

# Hyperparameters
input_dim = 32
hidden_dim = 64
batch_size = 8
epochs = 5

# Dummy Dataset
def generate_dummy_data(batch_size, input_dim):
    return torch.randn(batch_size, input_dim), torch.randn(batch_size, input_dim)

# Model
model = MoELayer(input_dim=input_dim, hidden_dim=hidden_dim, num_experts=4, top_k=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training Loop
print("🚀 Starting training loop for Grok‑3 MoE...")
for epoch in range(epochs):
    inputs, targets = generate_dummy_data(batch_size, input_dim)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

print("✅ Training completed.")
