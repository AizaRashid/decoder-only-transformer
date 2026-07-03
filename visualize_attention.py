"""
visualize_attention.py

Renders the attention weights of the last decoder block as a heatmap for a
single example, one panel per head. This is the actual evidence that the
model is doing something other than memorizing: if it were just memorizing
positions, attention wouldn't need to vary based on which country or
capital appears in the sentence.

Run: python3 visualize_attention.py
Output: assets/attention_heatmap.png
"""

import matplotlib.pyplot as plt
import torch

from dataset import EOS
from generate import load_model

EXAMPLE_TOKENS = ["capital", "of", "kenya", "?", EOS, "nairobi"]


@torch.no_grad()
def get_attention(model, token_to_id, tokens):
    input_ids = torch.tensor([[token_to_id[t] for t in tokens]], dtype=torch.long)
    _, all_layers_attn = model(input_ids, return_attention=True)
    # Use the last block: by then attention has had every earlier layer's
    # output to work with, so its patterns are the most "settled".
    last_layer_attn = all_layers_attn[-1][0]  # (n_heads, seq, seq)
    return last_layer_attn


def plot_attention(attn_weights, tokens, out_path="assets/attention_heatmap.png"):
    n_heads = attn_weights.size(0)
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for head_idx, ax in enumerate(axes):
        weights = attn_weights[head_idx].numpy()
        im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)
        ax.set_title(f"head {head_idx}")
        ax.set_xlabel("attending to (key)")
        if head_idx == 0:
            ax.set_ylabel("query token")

    fig.suptitle(f'Last-block attention for: "{" ".join(tokens)}"')
    fig.colorbar(im, ax=axes, shrink=0.7, label="attention weight")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    model, token_to_id, id_to_token = load_model()
    attn_weights = get_attention(model, token_to_id, EXAMPLE_TOKENS)
    plot_attention(attn_weights, EXAMPLE_TOKENS)
