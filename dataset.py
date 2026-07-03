"""
dataset.py

Builds the training data for our decoder-only transformer.

Design choice: instead of one fact phrased two ways (which a model can solve
by memorizing position -> token, no attention required), we use a small set
of country/capital facts, each phrased as a *forward* question ("capital of
X ?") and a *backward* question ("X is the capital of ?"). Two consequences:

1. The model has to use the QUESTION WORD ORDER to figure out which entity
   is being asked about, and then use the CONTENT of the question (which
   country or capital appears) to pick the right answer. That's exactly the
   job self-attention does: relate the answer position back to the relevant
   token in the question, not just to a fixed position.
2. Because forward and backward questions share the same vocabulary but ask
   for different targets, a model that only memorized "token at position 2
   -> answer" would fail on one direction while passing the other. Solving
   both directions is a much stronger correctness signal than a single
   fact repeated twice.

Everything here is plain Python + a public list of countries/capitals we
typed in ourselves -- no scraped or copyrighted text.
"""

import torch

# A small, easily hand-checkable set of country -> capital facts.
# Kept to single-word capitals/countries so word-level tokenization stays simple.
FACTS = [
    ("france", "paris"),
    ("japan", "tokyo"),
    ("italy", "rome"),
    ("egypt", "cairo"),
    ("canada", "ottawa"),
    ("brazil", "brasilia"),
    ("germany", "berlin"),
    ("india", "delhi"),
    ("spain", "madrid"),
    ("kenya", "nairobi"),
    ("norway", "oslo"),
    ("chile", "santiago"),
]

EOS = "<eos>"
PAD = "<pad>"


def build_examples():
    """
    Turn each (country, capital) fact into two training examples:
      forward:  "capital of <country> ?" -> "<capital>"
      backward: "<capital> is the capital of ?" -> "<country>"

    Each example is returned as a single list of tokens that includes an
    <eos> token *between* the question and the answer, and another <eos>
    at the very end. That inner <eos> is what tells the model "the question
    is finished, start answering" -- without it, there's no signal marking
    where the prompt ends and generation should begin.
    """
    examples = []
    for country, capital in FACTS:
        forward = ["capital", "of", country, "?", EOS, capital, EOS]
        backward = [capital, "is", "the", "capital", "of", "?", EOS, country, EOS]
        examples.append(forward)
        examples.append(backward)
    return examples


def build_vocab(examples):
    """Word-level vocab built directly from the data (plus a pad token for batching)."""
    vocab = {PAD: 0}
    for tokens in examples:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    id_to_token = {i: t for t, i in vocab.items()}
    return vocab, id_to_token


def encode(tokens, token_to_id):
    return [token_to_id[t] for t in tokens]


def make_training_tensors(examples, token_to_id, pad_id):
    """
    Standard "teacher forcing" setup for a decoder-only LM: the label at
    position i is just the input token at position i+1 (predict the next
    token). We right-pad every sequence to the longest one in the batch so
    they can be stacked into a single tensor, and we remember each real
    sequence length so training can build a mask that ignores the padding.
    """
    encoded = [encode(tokens, token_to_id) for tokens in examples]
    max_len = max(len(seq) for seq in encoded)

    input_ids = torch.full((len(encoded), max_len - 1), pad_id, dtype=torch.long)
    label_ids = torch.full((len(encoded), max_len - 1), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(encoded), dtype=torch.long)

    for i, seq in enumerate(encoded):
        seq_t = torch.tensor(seq, dtype=torch.long)
        n = len(seq) - 1  # number of (input, label) pairs this example contributes
        input_ids[i, :n] = seq_t[:-1]
        label_ids[i, :n] = seq_t[1:]
        lengths[i] = n

    return input_ids, label_ids, lengths


if __name__ == "__main__":
    examples = build_examples()
    vocab, id_to_token = build_vocab(examples)
    print(f"{len(examples)} training examples, vocab size {len(vocab)}")
    for tokens in examples[:4]:
        print(" ", " ".join(tokens))
