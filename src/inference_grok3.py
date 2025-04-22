# inference_grok3.py — Real Text Inference using Hugging Face GPT-2

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2
print("🚀 Loading model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inference function
def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_k=40, temperature=0.8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompt
if __name__ == "__main__":
    prompt = "Grok-3 is an AI system designed to"
    generated = generate_text(prompt)
    print("\n🧠 Generated Text:\n", generated)
