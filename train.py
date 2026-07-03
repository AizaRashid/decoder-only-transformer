"""
train.py

Trains the DecoderOnlyTransformer on the country/capital dataset.

We use a plain PyTorch loop instead of a training framework (the original
tutorial this project is inspired by uses PyTorch Lightning). Writing the
loop by hand -- zero_grad, forward, loss, backward, step -- keeps every
optimization step visible, which matters more for a learning-focused
portfolio project than the convenience Lightning would add on a dataset
this small.
"""

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import build_examples, build_vocab, make_training_tensors, PAD
from model import DecoderOnlyTransformer

SEED = 42
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.1
LEARNING_RATE = 3e-3
EPOCHS = 400
LOG_EVERY = 50
CHECKPOINT_PATH = "checkpoints/model.pt"


def train():
    torch.manual_seed(SEED)

    examples = build_examples()
    token_to_id, id_to_token = build_vocab(examples)
    pad_id = token_to_id[PAD]

    input_ids, label_ids, lengths = make_training_tensors(examples, token_to_id, pad_id)
    max_len = input_ids.size(1) + 1  # +1 to give position encoding a little headroom

    model = DecoderOnlyTransformer(
        vocab_size=len(token_to_id),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=max_len,
        dropout=DROPOUT,
        pad_id=pad_id,
    )

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # ignore_index=pad_id: padded positions contribute nothing to the loss,
    # since they aren't real tokens the model should be scored on predicting.
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        logits = model(input_ids)  # (batch, seq, vocab)
        # CrossEntropyLoss wants (N, C) predictions vs (N,) targets, so we
        # flatten the batch and sequence dimensions together.
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), label_ids.reshape(-1))

        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            accuracy = token_accuracy(logits, label_ids, pad_id)
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  token_acc {accuracy:.3f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "config": dict(
                vocab_size=len(token_to_id), d_model=D_MODEL, n_heads=N_HEADS,
                n_layers=N_LAYERS, d_ff=D_FF, max_len=max_len, dropout=DROPOUT, pad_id=pad_id,
            ),
        },
        CHECKPOINT_PATH,
    )
    print(f"\nsaved checkpoint to {CHECKPOINT_PATH}")


def token_accuracy(logits, label_ids, pad_id):
    """Fraction of non-pad positions where argmax prediction matches the label."""
    predictions = logits.argmax(dim=-1)
    mask = label_ids != pad_id
    correct = ((predictions == label_ids) & mask).sum().item()
    total = mask.sum().item()
    return correct / total


if __name__ == "__main__":
    train()
