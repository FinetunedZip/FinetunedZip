from huggingface_hub import notebook_login
import torch

def setup():
    notebook_login()
    torch.cuda.set_device(0)
