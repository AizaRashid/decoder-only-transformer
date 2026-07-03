# Decoder-Only Transformer, From Scratch

A small decoder-only transformer (the architecture family behind GPT-style
models) implemented from first principles in PyTorch, no `transformers`
library, no pre-built attention layers. Multi-head causal self-attention,
sinusoidal position encoding, feed-forward blocks, residual connections,
and layer norm, stacked into a mini language model that learns to answer
country/capital questions from a handful of training examples.

## Why this exists

I started from a tutorial ([StatQuest: Coding Transformers from Scratch](https://www.youtube.com/@statquest)
by Josh Starmer) to understand the mechanics of a decoder-only transformer,
his walkthrough is a genuinely good explanation of position encoding and
masked self-attention. But the tutorial's model is intentionally minimal:
one attention head, `d_model=2`, a single decoder block, trained on one
fact phrased two ways. That's the right size for a first explanation, and
the wrong size to prove you understand anything about how the pieces
scale or where they go wrong.

So this project keeps the same starting concepts, attention, position
encoding, a decoder block, but rebuilds them independently: my own code,
own comments explaining the *why* rather than a line-by-line description
of the *what*, multi-head attention instead of single-head, stacked
layers instead of one, and a dataset built to actually need attention
rather than let the model shortcut to memorizing a position.

## What the model has to do

The training data is a set of country/capital facts, each asked two ways:

```
capital of france ?          -> paris
paris is the capital of ?    -> france
```

A model that just memorizes "the 3rd token predicts the 6th token" would
fail the moment the country changes. To get every fact right in both
directions, the model has to attend back to whichever token actually
carries the answer, which is the entire point of self-attention, and
why this dataset is a better test of it than a single repeated fact.

## Architecture

| Component | What it does |
|---|---|
| `SinusoidalPositionEncoding` | Fixed sin/cos signal added to embeddings so the model can tell token order apart |
| `MultiHeadSelfAttention` | 4 attention heads running in parallel, each learning its own notion of "what's relevant to me" |
| `FeedForward` | Per-token MLP (expand → GELU → contract) applied after attention |
| `DecoderBlock` | Attention + residual + LayerNorm, then feed-forward + residual + LayerNorm |
| `DecoderOnlyTransformer` | Embedding → position encoding → 2 stacked `DecoderBlock`s → final norm → output projection (weight-tied to the embedding) |

Causal masking and padding masking are combined into a single boolean
mask per batch, so training can process multiple variable-length examples
at once instead of one at a time.

Training uses a hand-written PyTorch loop (`train.py`) rather than a
training framework, every `zero_grad → forward → loss → backward → step`
is visible, which was the point: I wanted to actually see the mechanics,
not have a framework hide them.

## Results

- 400 training steps, full-batch gradient descent, Adam, lr=3e-3.
- Evaluated with **temperature sampling (not greedy argmax)**, repeated
  5 times per question, so a lucky argmax path can't hide a shaky model:
  **120/120 correct** across all 12 facts, both question directions.
- `assets/attention_heatmap.png` shows the last block's attention for
  `capital of kenya ? <eos> nairobi`: the answer token attends back to
  `kenya`, not to a fixed position, visual confirmation that the model
  is routing information through attention rather than memorizing slots.

## Files

```
model.py                # architecture: attention, feed-forward, decoder blocks
dataset.py               # country/capital facts + word-level tokenizer
train.py                  # training loop
generate.py                # autoregressive sampling + full accuracy check
visualize_attention.py       # renders the attention heatmap
checkpoints/model.pt          # trained weights (produced by train.py)
assets/attention_heatmap.png    # generated visualization
```

## Running it

```bash
pip install -r requirements.txt

python3 train.py                                  # trains, saves checkpoints/model.pt
python3 generate.py --prompt capital of japan ?    # -> tokyo
python3 generate.py --check_all                    # runs the 120-question accuracy check
python3 visualize_attention.py                     # regenerates the attention heatmap
```

## What I'd do next

- Swap the toy dataset for a real small corpus (e.g. a public-domain
  short story) and see how the same architecture handles open-ended
  text generation instead of closed-form Q&A.
- Add KV-caching to `generate.py` so decoding doesn't recompute
  attention over the whole prefix at every step.
- Try ablating heads/layers on the current dataset to see how much
  capacity 12 facts actually need, a cheap way to build intuition
  about capacity vs. task complexity before touching anything bigger.

---

