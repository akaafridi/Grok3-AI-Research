import argparse  # To handle --prompt from command line
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Function to run inference
def generate_text(prompt, max_length=50, temperature=0.8, top_k=40):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.eval()

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, temperature=temperature, top_k=top_k)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# This part runs when you use the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grok-3 CLI Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate from")
    parser.add_argument("--max_length", type=int, default=50, help="Max token length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")

    args = parser.parse_args()  # Read all command line inputs
    output = generate_text(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("\n🧠 Grok-3 Output:\n" + output)
