# 01 · LLM Fundamentals

How a transformer-based LLM actually turns text into numbers, processes it, and
generates the next token — built from the ground up with Hugging Face `transformers`
and PyTorch, using GPT-2 as the working example.

## Notebook

### `llm_fundamentals_tokenizers.ipynb`

- Loads the GPT-2 tokenizer and causal language model (`AutoTokenizer`,
  `AutoModelForCausalLM`).
- **Tokenization**: converts raw text to input IDs and back, showing how GPT-2's
  byte-pair encoding splits words/sub-words into tokens (e.g. `"It was a dark and
  windy"` → token ID sequence `[1026, 373, 257, 3223, 290, 6388, 88]`).
- **Forward pass**: runs the tokenized input through the model and inspects the
  raw `CausalLMOutputWithCrossAttentions` — logits over the vocabulary for the
  next-token prediction.
- **Greedy generation**: extends a prompt token-by-token using `model.generate`,
  showing how the model autoregressively continues a sentence.
- **Token frequency analysis**: tallies the probability distribution over candidate
  next tokens for a prompt — e.g. for a "time of day" continuation, the model
  assigns `night` 46.2%, `day` 23.5%, `evening` 5.9%, `morning` 4.4%, demonstrating
  how next-token probabilities encode real-world likelihood.

## Key Takeaways

- Text → tokens → embeddings → logits → next-token sampling is the core loop
  behind every LLM, including ChatGPT, Gemini, and Llama.
- Tokenizers don't split on words — they split on sub-word units learned from a
  training corpus, which is why uncommon words can split into multiple tokens.
- The model's output is a probability distribution over the entire vocabulary at
  every step; "generation" is just repeated sampling/argmax from that distribution.

## Tech Stack

`transformers` · PyTorch · GPT-2
