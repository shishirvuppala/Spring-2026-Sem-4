import torch
import glob
import os
import matplotlib.pyplot as plt
from model import OneHotMultiHeadAttention

def helper_plot(attn, N, case_num):
    """Helper provided to visualize their attention pattern."""
    plt.figure(figsize=(6, 5))
    plt.imshow(attn[0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.title(f"Attention Map (Case {case_num}, N={N})")
    plt.colorbar()
    plt.savefig(f"attn_N{N}.png")
    plt.close()

def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(217) # Ensure reproducibility
    
    try:
        model = OneHotMultiHeadAttention(d_model=128, num_heads=4).to(device)
        model.eval()
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    test_files = sorted(glob.glob("test_cases/case_*.pt"))
    if not test_files:
        print("No test cases found in test_cases/ directory!")
        return

    print(f"Checking Task 2: One-hot Attention Implementation...")
    print("-" * 60)

    for i, file in enumerate(test_files):
        data = torch.load(file)
        x = data["input"].to(device)
        target_out = data["target_output"].to(device)
        N = data["N"]

        with torch.no_grad():
            try:
                out, attn = model(x)
            except Exception as e:
                print(f"Error during forward pass for Case {i+1}: {e}")
                continue

        # We use a small epsilon for float precision
        is_correct_out = torch.allclose(out, target_out, atol=1e-3)

        status = "PASSED" if (is_correct_out) else "FAILED"
        
        print(f"Test Case {i+1} (N={N}): {status}")
        if not is_correct_out:
            print(f"   > Numerical Error: Output values do not match ground truth.")
        
        # Generate plot to debug
        helper_plot(attn, N, i+1)

    print("-" * 60)
    print("Verification complete. Check your directory for PNG debug plots.")

if __name__ == "__main__":
    evaluate()