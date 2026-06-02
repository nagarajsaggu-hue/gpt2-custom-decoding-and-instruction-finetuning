# Natural Language Processing - Programming Task 2

**Project:** Custom Text Generation and GPT-2 Fine-Tuning  
**Author:** Saggu Nagaraju

## Project Overview

This project implements two core Natural Language Processing tasks related to language modeling and text generation:

1. **Custom text generation using decoding strategies** with GPT-2.
2. **Fine-tuning GPT-2 on an instruction-following dataset** and evaluating the generated responses.

The project focuses on understanding how different decoding methods influence generated text quality and how fine-tuning affects instruction-following behavior. The implementation uses PyTorch, Hugging Face Transformers, and the Alpaca instruction dataset.

## Main Objectives

- Implement a custom decoder for GPT-2 text generation.
- Compare greedy decoding, temperature sampling, top-k sampling, top-p sampling, and hybrid sampling.
- Fine-tune GPT-2 on instruction-style prompts.
- Evaluate generated text using quantitative metrics.
- Analyze output quality, repetition, diversity, and instruction alignment.

## Project Structure

```text
.
├── decoder.py
├── finetuning.py
├── helper.py
├── requirements.txt
├── custom_decoder_results_GROUP_16.txt
├── generated_outputs_gpt2_16.txt
├── evaluation_results_gpt2_16.txt
├── log_GROUP_16.log
├── config.json
├── generation_config.json
├── merges.txt
├── vocab.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## File Description

| File | Description |
|---|---|
| `decoder.py` | Implements the custom GPT-2 decoder and sampling strategies. |
| `finetuning.py` | Implements GPT-2 tokenization, training, generation, and evaluation. |
| `helper.py` | Provides dataset loading, train-test splitting, logging, and reproducibility utilities. |
| `requirements.txt` | Lists the required Python packages. |
| `custom_decoder_results_GROUP_16.txt` | Stores generated outputs for different decoding strategies. |
| `generated_outputs_gpt2_16.txt` | Stores generated responses after model evaluation. |
| `evaluation_results_gpt2_16.txt` | Stores perplexity, repetition, diversity, and Jaccard similarity results. |
| `log_GROUP_16.log` | Contains training, decoding, and evaluation logs. |
| `config.json`, `vocab.json`, `merges.txt`, etc. | GPT-2 model and tokenizer configuration files. |

## Task 1 - Custom Decoder

The custom decoder loads GPT-2 using Hugging Face Transformers and generates text token by token. The `_sample_token()` function implements several decoding strategies:

- **Greedy decoding:** selects the token with the highest probability.
- **Temperature sampling:** scales logits to control randomness.
- **Top-k sampling:** restricts sampling to the k most likely tokens.
- **Top-p sampling:** samples from the smallest probability mass above a cumulative threshold.
- **Hybrid sampling:** combines temperature, top-k, and top-p settings.

The tested configurations include:

| Strategy | Configuration |
|---|---|
| Temperature sampling | `temperature = 0.5` |
| Greedy decoding | `do_sample = False` |
| Top-k sampling | `top_k = 10`, `top_k = 50` |
| Top-p sampling | `top_p = 0.5`, `top_p = 0.95` |
| Hybrid sampling | `temperature = 0.7`, `top_k = 50`, `top_p = 0.9` |
| Hybrid sampling | `temperature = 1.0`, `top_k = 40`, `top_p = 0.8` |

## Task 2 - GPT-2 Fine-Tuning

The fine-tuning pipeline uses GPT-2 for causal language modeling. The model is trained on instruction-style text data from the Alpaca dataset.

### Fine-Tuning Setup

| Parameter | Value |
|---|---:|
| Model | GPT-2 |
| Dataset | Alpaca instruction-following dataset |
| Small dataset mode | Enabled |
| Dataset size used | 10 examples |
| Train/test split | 8 train / 2 test |
| Epochs | 1 |
| Batch size | 4 |
| Learning rate | 5e-5 |
| Warmup steps | 50 |
| Max sequence length | 128 |

## Evaluation Metrics

The fine-tuned model was evaluated using the following metrics:

| Metric | Value |
|---|---:|
| Perplexity | 277.3959 |
| Average repetition score | 0.2801 |
| Average diversity score | 0.4888 |
| Average Jaccard similarity | 0.5177 |

## Result Interpretation

- The model generated instruction-aligned outputs for simple prompts.
- Some responses were relevant but repeated phrases or prompt sections.
- The high perplexity indicates that the model still had difficulty learning strongly from the very small training subset.
- The diversity score shows moderate lexical variation.
- The Jaccard similarity indicates partial overlap between generated and expected outputs.

## How to Run the Project

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the custom decoder

```bash
python decoder.py
```

This generates text using different decoding strategies and saves the results to:

```text
custom_decoder_results_GROUP_16.txt
log_GROUP_16.log
```

### 4. Run GPT-2 fine-tuning and evaluation

```bash
python finetuning.py
```

This fine-tunes GPT-2, saves the model, generates outputs, and writes evaluation results to:

```text
generated_outputs_gpt2_16.txt
evaluation_results_gpt2_16.txt
log_GROUP_16.log
```

## Example Output Behavior

For simple instruction prompts such as asking for health tips, the model generated relevant content but also repeated some lines. For grammar correction and primary color questions, the output showed partial correctness but also produced repeated or incorrect continuation patterns.

## Limitations

- Only a very small subset of the Alpaca dataset was used.
- Training was performed for only one epoch.
- The generated responses still show repetition and prompt continuation artifacts.
- More training data, longer training, and stronger generation constraints would improve output quality.

## Future Improvements

- Train on a larger Alpaca subset or the complete dataset.
- Compare pre-trained GPT-2 and fine-tuned GPT-2 results more deeply.
- Add BLEU, ROUGE, and BERTScore evaluation.
- Improve prompt formatting and response extraction.
- Add early stopping and validation tracking.
- Use GPU training for larger experiments.

## Skills Demonstrated

- Natural Language Processing
- Language modeling
- GPT-2 text generation
- Custom decoding strategies
- Fine-tuning transformer models
- PyTorch training loop implementation
- Hugging Face Transformers
- Text generation evaluation
- Experiment logging and result analysis
