import random
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from dataset import CustomDataset
from model import RNNAttentionModel

def set_seed(seed=42):
    """Locks down all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def visualize_attention(model, x_example):
#     model.eval()
    
#     with torch.no_grad():
#         _, attn_weights = model(x_example, return_attention=True)
    
#     # attn_weights shape: (1, T_out, T_in)
#     # Remove batch dimension and move to CPU
#     attn_matrix = attn_weights[0].cpu().numpy()
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(attn_matrix, annot=False, cmap='viridis', 
#                 xticklabels=x_example[0].cpu().tolist(), 
#                 yticklabels=range(len(x_example[0])))
    
#     plt.xlabel("Input Sequence (Encoder Timesteps)")
#     plt.ylabel("Output Sequence (Decoder Timesteps)")
#     plt.title("Attention Heatmap")
#     plt.savefig('attention.png')

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=-1)

            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / total

def print_examples(model, test_loader, n, device):
    model.eval() # Always set to eval mode for inference
    
    # ANSI Color codes for better visibility
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= n: break

            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)

            # visualize_attention(model, x, device="cpu")

            # Take the first example in the batch
            inp = x[0].cpu().tolist()
            tgt = y[0].cpu().tolist()
            prd = preds[0].cpu().tolist()

            is_correct = tgt == prd
            status = f"{GREEN}✔ CORRECT{RESET}" if is_correct else f"{RED}✘ INCORRECT{RESET}"

            print(f"{BOLD}--- Example {i+1} | {status} ---{RESET}")
            
            # Use join for a cleaner horizontal list look
            print(f"{'Input:':<12} {' '.join(map(str, inp))}")
            print(f"{'Target:':<12} {' '.join(map(str, tgt))}")
            
            # Construct the prediction string token by token
            colored_preds = []
            for p, t in zip(prd, tgt):
                color = GREEN if p == t else RED
                colored_preds.append(f"{color}{p}{RESET}")
            
            pred_str = " ".join(colored_preds)
            print(f"{'Prediction:':<12} {pred_str}")
            print("-" * 40 + "\n")

def train(K, device="cpu"):
    model = RNNAttentionModel(K=K).to(device)

    print(f"Number of trainable parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad==True]):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    acc_max = -np.inf

    for epoch in range(10):
        dataset = CustomDataset(K=K)

        # Train/Test split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64)

        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        test_acc = evaluate(model, test_loader, device)
        acc_max = max(acc_max, test_acc)

        print(f"Epochs: {epoch+1}/10")
        print(f"  Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
    
    print(f"Best Test Accuracy: {acc_max*100:.2f}%")
    print()
    print_examples(model, test_loader, n=2, device=device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    set_seed(217)
    K = 2
    train(K, device=device)
