# LLMZip
1. conda env create -f environment.yml
2. huggingface-cli login --token {token}
3. conda activate venv
4. Run scripts like: CUDA_VISIBLE_DEVICES="{number corresponding to available gpu}" python3 -m scripts.eval

eval.py and finetune.py in the scripts directory.

Use memory_eval function in eval.py