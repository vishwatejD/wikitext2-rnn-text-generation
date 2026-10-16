# Word-Level Text Generation on WikiText-2: RNN vs. LSTM vs. GRU

A PyTorch study of autoregressive, word-level text generation on
[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext). It compares four
recurrent architectures under identical training conditions:

- Simple RNN
- Bidirectional RNN
- LSTM
- GRU

Each model reads a 32-token context and predicts the next word. The notebook evaluates
the models quantitatively (perplexity, top-1/top-5 accuracy) and qualitatively (generated
text at sampling temperatures 0.5, 1.0 and 1.5, with raw and top-k/top-p sampling).

Everything lives in one notebook: [`wikitext2_text_generation.ipynb`](wikitext2_text_generation.ipynb).

## Results

Test set, 239,233 next-word targets whose 32-token contexts stay inside one article.
Best checkpoint chosen by validation loss.

| Model | Loss | Perplexity | Top-1 | Top-5 | Known-word top-1 | Parameters | Optimization time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LSTM** | 4.7200 | **112.17** | 25.57% | 46.16% | 25.20% | 4,720,893 | 317 s |
| GRU | 4.7509 | 115.69 | 25.22% | 45.96% | 24.69% | 4,622,077 | 307 s |
| Simple RNN | 5.0847 | 161.53 | 22.15% | 42.64% | 21.73% | 4,424,445 | 249 s |
| Bidirectional RNN | 5.0895 | 162.31 | 21.98% | 42.49% | 21.61% | 4,391,677 | 212 s |
| Unigram baseline | 6.8506 | 944.44 | 5.83% | n/a | n/a | n/a | n/a |

Timings are from an RTX 4070 Laptop GPU.

### Findings

- **LSTM is best, GRU is close.** The LSTM–GRU gap is small and comes from a single
  training seed, so treat them as comparable. Both clearly beat the Simple and Bidirectional
  RNNs, and all four are far better than the unigram baseline.
- **Bidirectionality doesn't help here.** The bidirectional model only encodes the observed
  context (never the target), and it did not outperform the unidirectional Simple RNN.
- **Temperature trades coherence for variety.** At T=0.5 the models loop on Wikipedia
  templates ("The film was released…") and, for the Simple/Bidirectional RNNs, on `<unk>`.
  At T=1.0 text is locally fluent but drifts between topics. At T=1.5 lexical diversity is
  near 1 but the output is word salad. High diversity is not coherence.
- **Filtering is not a free fix.** Top-k 50 / top-p 0.9 with `<unk>` suppressed reads better
  but can increase repetition (LSTM at T=0.5: repeated-trigram rate 0.168 raw → 0.352
  filtered).
- **Generated text is fluent in fragments, not factual.** Models invent details such as
  impossible dates.

## Method summary

- **Data:** official WikiText-2 train/validation/test splits. Vocabulary built from the
  training split only (33,277 words, `<unk>` included). Whitespace tokenization.
- **Windows:** 32-token context → next word. No window crosses an article boundary. Windows
  are built lazily on the GPU.
- **Training:** one *sweep* is a quarter of the training targets, sampled without
  replacement. Maximum 24 sweeps (6 full passes), batch size 256, Adam (lr 0.002),
  gradient clipping at 1, ReduceLROnPlateau, early stopping (patience 5).
- **Architecture:** 128-d embeddings, 256-d recurrent representation (128 per direction
  for the bidirectional model), dropout 0.15, learned 256→128 projection, embedding/output
  weight tying, output bias initialised from training unigram frequencies.
- **Evaluation:** cross-entropy, perplexity, top-1/top-5 accuracy, and known-word accuracy
  (so `<unk>` targets don't inflate scores).
- **Generation:** 3 fixed test-set seed contexts × 3 temperatures × 3 draws × raw and
  filtered sampling × 4 models = 216 continuations of 80 tokens, with distinct-1/2,
  repeated-trigram and unknown-token metrics.

## Getting started

```bash
git clone https://github.com/vishwatejD/wikitext2-rnn-text-generation.git
cd wikitext2-rnn-text-generation
pip install -r requirements.txt
jupyter notebook wikitext2_text_generation.ipynb
```

Run all cells in order. The first cell installs any missing packages. The dataset is
downloaded automatically from the Hugging Face Hub.

- **Hardware:** a CUDA GPU is strongly recommended. Training all four models took about
  20 minutes on an RTX 4070 Laptop GPU. CPU-only runs work but are much slower.


## Repository contents

| File | Purpose |
|---|---|
| `wikitext2_text_generation.ipynb` | Full experiment: data, models, training, evaluation, generation, discussion |
| `requirements.txt` | Python dependencies |
| `results/` (generated) | `test_results.csv`, `generations.csv`, plots |

## Limitations

- One training seed, so no variance estimate across training runs. Small differences
  (LSTM vs. GRU, timings) should not be over-interpreted.
- All four models' best checkpoint was the final sweep, so they were still improving at the
  six-pass budget. The ranking is for a fixed budget, not for converged models.
- A 32-token context limits long-range coherence. Generation statistics use nine draws per
  model/mode/temperature and measure sampling variation only.
- Generation metrics (distinct-n, repetition) do not measure semantic quality.

## Data and licence

WikiText-2 is distributed under CC BY-SA 3.0 and is downloaded at run time, not stored in
this repository.

> Merity, S., Xiong, C., Bradbury, J., Socher, R. (2016). *Pointer Sentinel Mixture Models.*
> [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)

Add a `LICENSE` file for your own code (MIT is a common choice).
