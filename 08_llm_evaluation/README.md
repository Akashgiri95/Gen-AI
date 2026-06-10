# 08 · LLM Evaluation

How to *measure* LLM output quality quantitatively, using the Hugging Face
[`evaluate`](https://huggingface.co/docs/evaluate) library — moving beyond "does
this look right" to metrics that can be tracked across model versions and prompts.

## Notebook

### `llm_evaluation_metrics.ipynb`

Walks through the major categories of LLM evaluation metrics, each with a worked
example:

- **Classification metrics** — `accuracy`, `f1`, `pearsonr`, used for tasks where
  the LLM's output can be checked against a ground-truth label.
- **Perplexity** — measures how "surprised" GPT-2 is by a piece of generated text
  (`evaluate.load('perplexity').compute(model_id='gpt2', predictions=...)`); lower
  perplexity = more fluent/predictable text under the model.
- **ROUGE** — n-gram overlap between a generated summary and a reference summary,
  including a worked example showing how naive repetition can artificially inflate
  ROUGE scores ("gaming" the metric).
- **BLEU & METEOR** — translation/generation quality metrics comparing candidate
  text against one or more reference texts at the n-gram and synonym/stem level.
- **Exact Match** — strict string-equality scoring, useful for QA tasks with a
  single correct short answer.
- **Toxicity** — scores generated text for toxic content (`toxicity_ratio`),
  comparing two sets of predictions about the same subject to check for biased or
  harmful framing.
- **Regard** — measures the *social perception* (positive/neutral/negative)
  conveyed about a demographic group in generated text — e.g. comparing how an LLM
  describes "abc are loyal employees" vs. negatively-framed alternatives, surfacing
  potential bias.

## Key Takeaways

- Different tasks need different metrics: perplexity for fluency, ROUGE/BLEU/METEOR
  for generation-vs-reference overlap, exact match for QA, toxicity/regard for
  safety and fairness.
- Overlap-based metrics (ROUGE, BLEU) can be gamed by repetition or near-duplicate
  phrasing — they're a useful signal, not a substitute for human review.
- Toxicity and regard metrics are essential for catching subtle bias in generated
  text about people/groups — something accuracy-style metrics can't detect at all.

## Tech Stack

Hugging Face `evaluate` · `transformers` (GPT2) · PyTorch
