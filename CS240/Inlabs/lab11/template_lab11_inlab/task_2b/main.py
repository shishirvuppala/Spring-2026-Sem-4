import torch
import glob
import os
import matplotlib.pyplot as plt
from model import WindowedMultiHeadAttention

def helper_plot(attn, N, K, case_num):
    """Visualizes the attention pattern for debugging."""
    plt.figure(figsize=(6, 5))
    # We plot the first head of the first batch element
    plt.imshow(attn[0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.title(f"Windowed Attention (Case {case_num}: N={N}, K={K})")
    plt.xlabel("Key Position (j)")
    plt.ylabel("Query Position (i)")
    plt.colorbar()
    
    filename = f"attn_N{N}_K{K}.png"
    plt.savefig(filename)
    plt.close()
    print(f"   [Plot] Saved attention map to {filename}")

def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Locate all generated test cases
    test_files = sorted(glob.glob("test_cases/case_windowed_*.pt"))
    if not test_files:
        print("No test cases found in test_cases/ directory!")
        return

    print(f"Checking Task 2: Windowed Attention Implementation...")
    print("-" * 60)

    for i, file in enumerate(test_files):
        torch.manual_seed(217) # Ensure reproducibility
        # Load instructor-generated data
        data = torch.load(file, map_location=device)
        x = data["input"]
        target_out = data["target_output"]
        N = data["N"]
        K = data["K"]

        # Initialize model with the specific K for this test case
        try:
            model = WindowedMultiHeadAttention(d_model=128, num_heads=4, window_size=K).to(device)
            model.eval()
        except Exception as e:
            print(f"Error initializing model for Case {i+1}: {e}")
            continue

        # Forward pass
        with torch.no_grad():
            try:
                out, attn = model(x)
            except Exception as e:
                print(f"Error during forward pass for Case {i+1}: {e}")
                continue

        # We use a small epsilon for float precision
        is_correct_out = torch.allclose(out, target_out, atol=1e-3)

        status = "PASSED" if (is_correct_out) else "FAILED"
        
        print(f"Test Case {i+1} (N={N}, K={K}): {status}")
        
        if not is_correct_out:
            print(f"   > Numerical Error: Output values do not match ground truth.")
        
        # Generate plot to help visualize the band-diagonal pattern
        helper_plot(attn, N, K, i+1)

    print("-" * 60)
    print("Verification complete. Check your directory for PNG debug plots.")

if __name__ == "__main__":
    evaluate()