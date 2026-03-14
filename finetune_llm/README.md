# Finetune LLM module

This folder contains scripts to fine-tune and evaluate a Vietnamese legal QA LLM based on the base model `1TuanPham/T-VisStar-7B-v0.1`.

## Files overview

- `requirements.txt`: Python dependencies for finetuning and evaluation (transformers, peft, bitsandbytes, accelerate, datasets, wandb, trl, pandas, etc.).
- `download_model.py`: Minimal helper to download the base model and tokenizer from Hugging Face to the local cache.
- `gen_data.py`: Uses `gpt-4o-mini` via the OpenAI API to generate conversational training data from an initial CSV (`train.csv`). It produces `gen_data/train.csv` with columns `question`, `context`, and `answer`.
- `finetune.py`: Main script to fine-tune `1TuanPham/T-VisStar-7B-v0.1` using QLoRA + `SFTTrainer` on the conversational data. Logs to Weights & Biases and saves LoRA checkpoints into `output_ckp/`.
- `merge_with_base.py`: Merges a trained LoRA checkpoint with the base model to produce a standalone merged model directory ready for inference.
- `test_model.py`: Runs inference with the merged model using `vLLM` on a test CSV and writes predictions to a results CSV.
- `evaluate_finetuned_model.py`: Uses LlamaIndex `CorrectnessEvaluator` with `gpt-4o-mini` to score the model answers vs ground truth from the results CSV.

## Requirements

- Python 3.10+
- GPU with enough VRAM for a 7B model and QLoRA (recommended)
- Installed dependencies:
  - `pip install -r requirements.txt`
- Access keys / environment variables:
  - `OPENAI_API_KEY` for `gen_data.py` and `evaluate_finetuned_model.py`
  - `WANDB_API_KEY` (or `wandb login`) for logging in `finetune.py`

## Step-by-step finetuning workflow

1. **Prepare raw supervised data**
   - Place your base CSV (e.g. `train.csv`) in this folder with columns such as `question` and `context` (list or text of retrieved documents).

2. **Generate conversational training data**
   - Ensure `OPENAI_API_KEY` is set in your environment or `.env` file.
   - Adjust index ranges and file paths in `gen_data.py` to cover your desired subset.
   - Run:
     - `cd finetune_llm`
     - `python gen_data.py`
   - Output: `gen_data/train.csv` with columns `question`, `context`, `answer`.

3. **Install dependencies and login to W&B**
   - From the project root or this folder:
     - `pip install -r finetune_llm/requirements.txt`
   - Login to Weights & Biases (or update the key in `finetune.py`):
     - `wandb login`  (or edit `wandb.login(key=...)`).

4. **Fine-tune the base model (QLoRA)**
   - In this folder, verify in `finetune.py` that:
     - `base_model_id` is set to your base model (default `1TuanPham/T-VisStar-7B-v0.1`).
     - `train_csv` / `test_csv` paths point to your generated data (e.g. `gen_data/train.csv`, `gen_data/test.csv`).
   - Run:
     - `python finetune.py`
   - Checkpoints will be saved under `output_ckp/`.

5. **Merge LoRA weights into base model**
   - Edit `merge_with_base.py` to set:
     - `--peft_model_path` to your chosen LoRA checkpoint inside `output_ckp/`.
     - `--output_path` to a new directory where you want the merged model stored.
   - Run, for example:
     - `python merge_with_base.py --base_model_path 1TuanPham/T-VisStar-7B-v0.1 --peft_model_path output_ckp/checkpoint-XXXX --output_path merged_model`

6. **Test the merged model**
   - Update paths in `test_model.py`:
     - `model=` argument to point to your `merged_model` directory.
     - CSV paths (`test.csv`, `results.csv`) to your data locations.
   - Run:
     - `python test_model.py`
   - This will produce a results file (e.g. `gen_data/results.csv`) containing model answers vs ground truth.

7. **Evaluate correctness**
   - Ensure `OPENAI_API_KEY` is set.
   - Update file paths in `evaluate_finetuned_model.py` to point to your results CSV.
   - Run:
     - `python evaluate_finetuned_model.py`
   - The script prints average correctness score for the finetuned model.

This README is a high-level guide; always double-check hard-coded paths inside each script and adapt them to your environment (local directories, dataset locations, and model checkpoints).