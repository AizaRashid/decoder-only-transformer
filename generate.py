"""
generate.py

Autoregressive text generation from the trained model, plus a small eval
harness that asks every fact in the dataset and checks the answer.

Unlike greedy argmax decoding, we sample from the (temperature-scaled,
optionally top-k-filtered) output distribution. Greedy decoding always
picks the single most likely next token, so if a tiny model is even
slightly correct it will look perfect every single time -- that's not
strong evidence of a real generalizing solution, just of a lucky arg-max
path. Sampling with temperature > 0 lets us re-run the same prompt many
times and check whether the model is *reliably* right, not just right
on the one path argmax happens to take.
"""

import argparse
import torch
import torch.nn.functional as F

from dataset import build_examples, EOS
from model import DecoderOnlyTransformer

CHECKPOINT_PATH = "checkpoints/model.pt"
MAX_NEW_TOKENS = 10


def load_model(checkpoint_path=CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DecoderOnlyTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["token_to_id"], checkpoint["id_to_token"]


def sample_next_token(logits, temperature, top_k):
    logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        top_values, top_indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, top_indices, top_values)
        logits = filtered
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate(model, token_to_id, id_to_token, prompt_tokens, temperature=0.8, top_k=3):
    """
    prompt_tokens: list of strings, e.g. ["capital", "of", "france", "?"]
    Feeds <eos> after the prompt (matching training, where <eos> marks
    "question is done, start answering"), then samples tokens one at a
    time, feeding each prediction back in until <eos> or MAX_NEW_TOKENS.
    """
    input_ids = [token_to_id[t] for t in prompt_tokens] + [token_to_id[EOS]]
    generated = []

    for _ in range(MAX_NEW_TOKENS):
        x = torch.tensor([input_ids], dtype=torch.long)
        logits = model(x)
        next_logits = logits[0, -1, :]
        next_id = sample_next_token(next_logits, temperature, top_k)

        if id_to_token[next_id] == EOS:
            break
        generated.append(id_to_token[next_id])
        input_ids.append(next_id)

    return generated


def check_all_facts(model, token_to_id, id_to_token, temperature=0.8, top_k=3, trials=5):
    """
    Re-derives the forward/backward question for every fact in the dataset
    (rather than importing the answer key directly) and checks the model's
    generated answer against it, repeated `trials` times per question since
    we're sampling rather than doing greedy decoding.
    """
    from dataset import FACTS

    total, correct = 0, 0
    for country, capital in FACTS:
        for _ in range(trials):
            answer = generate(model, token_to_id, id_to_token,
                               ["capital", "of", country, "?"], temperature, top_k)
            total += 1
            correct += int(answer == [capital])

            answer = generate(model, token_to_id, id_to_token,
                               [capital, "is", "the", "capital", "of", "?"], temperature, top_k)
            total += 1
            correct += int(answer == [country])

    print(f"{correct}/{total} correct across {trials} sampling runs per question")
    return correct, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", nargs="+", help="e.g. --prompt capital of japan ?")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--check_all", action="store_true", help="run the full-dataset accuracy check")
    args = parser.parse_args()

    model, token_to_id, id_to_token = load_model()

    if args.check_all:
        check_all_facts(model, token_to_id, id_to_token, args.temperature, args.top_k)
    elif args.prompt:
        answer = generate(model, token_to_id, id_to_token, args.prompt, args.temperature, args.top_k)
        print(" ".join(args.prompt), "->", " ".join(answer))
    else:
        print("Pass --prompt capital of france ?  or  --check_all")
