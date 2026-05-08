import matplotlib.pyplot as plt
import torch

def visualize_attention_grid(model, sample, device, save_path="attention_grid.png"):
    model.eval()

    x, _ = sample
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn_maps = model(x, return_attn=True)
        preds = logits.argmax(dim=-1)

    num_layers = len(attn_maps)
    
    # Handle both MHA (heads) and Conv (channels)
    first_attn = attn_maps[0]
    num_heads = first_attn.shape[1]

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(3*num_heads, 3*num_layers))
    vmin = min(attn_maps[l][0].min().item() for l in range(num_layers))
    vmax = max(attn_maps[l][0].max().item() for l in range(num_layers))

    # If only 1 layer or 1 head, make axes 2D
    if num_layers == 1:
        axes = axes[None, :]
    if num_heads == 1:
        axes = axes[:, None]

    for l in range(num_layers):
        attn = attn_maps[l][0]  # (heads, T, T)

        for h in range(num_heads):
            ax = axes[l, h]
            attn_map = attn[h].cpu()

            # im = ax.imshow(attn_map, cmap='viridis')
            im = ax.imshow(attn_map, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f"L{l+1} H{h+1}")

            ax.set_xlabel("Key")
            ax.set_ylabel("Query")

    plt.tight_layout()
    plt.savefig(save_path)
    print("Attention plotted !!!")