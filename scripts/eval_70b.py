import pprint
from typing import List
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import array
import zlib
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from huggingface_hub import login
from itertools import cycle
# from utils.setup import setup
from pyfastpfor import getCodec
from accelerate import Accelerator

def tensor_to_numpy(tensor):
    # Ensure the tensor is on CPU and convert it to NumPy
    conv = np.array(tensor.cpu().numpy(), dtype=np.uint32, order='C')
    # ensure that eaach value is uint32
    for i in range(len(conv)):
        conv[i] = np.uint32(conv[i])
    return conv

def numpy_to_tensor(array: np.ndarray, device):
    # Convert the NumPy array back to a PyTorch tensor
    # convert the type of the np array to float32
    array = array.astype(np.float32)
    return torch.tensor(array, device=device)

class ZipModel:
    def __init__(self, model_name, tokenizer_name, model, tokenizer, finetuned, context_size, batch_size):
        self.accelerator = Accelerator()
        self.CONTEXT_SIZE = context_size
        self.BATCH_SIZE = batch_size
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.finetuned = finetuned

        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)
        self.device = self.accelerator.device

        self.ranks = []

    def text_to_tokens(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens["input_ids"].squeeze().to(self.device)
        return tokens

    def tokens_to_text(self, tokens):
        tokens = tokens.reshape((1, -1))
        text = self.tokenizer.batch_decode(tokens)
        return text[0]

    def pad(self, tokens, padding_val):
        pad_len = self.CONTEXT_SIZE - tokens.shape[0] % self.CONTEXT_SIZE
        if pad_len != self.CONTEXT_SIZE:
            padding = torch.tensor([padding_val] * pad_len).to(self.device)
            tokens = torch.cat((tokens, padding))
        else:
            pad_len = 0
        return tokens, pad_len

    @torch.no_grad()
    def get_logits(self, tokens, token_index, past=None):
        my_inputs = {'input_ids': tokens[:, token_index].reshape(-1, 1)}
        output = self.model(**my_inputs, past_key_values=past)
        logits = output.logits
        if len(logits.shape) > 2:
            logits = logits.reshape((logits.shape[0], -1))
        return logits, output.past_key_values

    def encode_one_batch(self, tokens, token_index, past=None):
        assert len(tokens.shape) == 2
        logits, past = self.get_logits(tokens, token_index, past)
        assert len(logits.shape) == 2
        logits, sorted_tokens = torch.sort(logits, descending=True)
        assert len(sorted_tokens.shape) == 2
        next_tokens = tokens[:, token_index + 1]
        next_tokens_expanded = next_tokens.view(-1, 1).expand_as(sorted_tokens)
        scores = (sorted_tokens == next_tokens_expanded).nonzero(as_tuple=True)
        scores = scores[1]
        self.ranks.extend(scores.tolist())
        return scores, past

    def decode_one_batch(self, input_tokens, scores, score_index, past=None):
        assert len(scores.shape) == 2
        logits, past = self.get_logits(input_tokens, score_index, past)
        logits, sorted_tokens = torch.sort(logits, descending=True)
        assert len(sorted_tokens.shape) == 2
        indexes = scores[:, score_index].flatten()
        decoded_tokens = sorted_tokens[torch.arange(indexes.shape[0]), indexes]
        return decoded_tokens.int(), past

    def encode(self, text):
        tokens = self.text_to_tokens(text)
        return self.encode_tokens(tokens)

    def encode_tokens(self, tokens):
        tokens, pad_len = self.pad(tokens, self.tokenizer.eos_token_id)
        tokens = tokens.view(-1, self.CONTEXT_SIZE)
        output_scores = torch.zeros(tokens.shape).to(self.device)
        eos = torch.tensor([self.tokenizer.eos_token_id] * tokens.shape[0]).unsqueeze(1).to(self.device)
        tokens = torch.cat((eos, tokens), 1)
        tokens = tokens.to(self.device)
        batches = tokens.shape[0] // self.BATCH_SIZE
        if tokens.shape[0] % self.BATCH_SIZE != 0:
            batches += 1
        print("Encoding")
        for i in range(batches):
            cur_tokens = tokens[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]
            cur_output_scores = torch.zeros((cur_tokens.shape[0], cur_tokens.shape[1] - 1)).to(self.device)
            past = None
            print(i, "out of", batches)
            for j in range(cur_tokens.shape[1] - 1):
                cur_output_scores[:, j], past = self.encode_one_batch(cur_tokens, j, past)
            output_scores[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE] = cur_output_scores
            del cur_tokens
        torch.cuda.empty_cache()
        output_scores = output_scores.flatten().int()
        if pad_len > 0:
            output_scores = output_scores[:-pad_len]
        return output_scores

    def decode(self, scores):
        output_tokens = self.decode_tokens(scores)
        text = self.tokenizer.batch_decode(output_tokens)
        text = "".join(text)
        return text

    def decode_tokens(self, scores):
        scores, pad_len = self.pad(scores, self.tokenizer.eos_token_id)
        scores = scores.view(-1, self.CONTEXT_SIZE)
        output_tokens = torch.zeros(scores.shape, dtype=int).to(self.device)
        eos = torch.tensor([self.tokenizer.eos_token_id] * output_tokens.shape[0]).unsqueeze(1).to(self.device)
        output_tokens = torch.cat((eos, output_tokens), 1)
        output_tokens = output_tokens.to(self.device)
        batches = scores.shape[0] // self.BATCH_SIZE
        if scores.shape[0] % self.BATCH_SIZE != 0:
            batches += 1
        print("Decoding")
        for i in range(batches):
            print(i, "out of", batches)
            cur_scores = scores[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]
            cur_output_tokens = output_tokens[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE]
            cur_output_tokens = cur_output_tokens.to(self.device)
            past = None
            for j in tqdm(range(scores.shape[1])):
                cur_output_tokens[:, j + 1], past = self.decode_one_batch(cur_output_tokens, cur_scores, j, past)
            output_tokens[i * self.BATCH_SIZE:(i + 1) * self.BATCH_SIZE] = cur_output_tokens
        output_tokens = output_tokens[:, 1:].int()
        output_tokens = output_tokens.flatten()
        if pad_len != 0:
            output_tokens = output_tokens[:-pad_len]
        return output_tokens

    def encode_and_zip(self, text):
        encoded = self.encode(text)
        codec = getCodec('optpfor')
        inpComp = np.zeros(encoded.size(0), dtype=np.uint32, order='C')
        compSize = codec.encodeArray(tensor_to_numpy(encoded), encoded.size(0), inpComp, len(inpComp))
        compressed_array = inpComp[:compSize]
        compressed_bytes = compressed_array.tobytes()
        return compressed_bytes

    def unzip_and_decode(self, zipped):
        unzipped = zlib.decompress(zipped)
        unzipped = array.array("H", unzipped)
        decoded = self.decode(torch.tensor(unzipped).to(self.device))
        return decoded

    def zip_file(self, text_file, zip_file):
        with open(text_file, encoding="utf-8") as f:
            text = f.read()
        zipped = self.encode_and_zip(text)
        with open(zip_file, "wb") as f:
            f.write(zipped)

    def unzip_file(self, zip_file, text_file):
        with open(zip_file, "rb") as f:
            zipped = f.read()
        text = self.unzip_and_decode(zipped)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)

    def plot_rank_distribution(self, plot_type='histogram'):
        if plot_type == 'histogram':
            plt.figure(figsize=(10, 6))
            plt.hist(self.ranks, bins=100, log=True)
            plt.title('Distribution of Token Ranks')
            plt.xlabel('Rank')
            plt.ylabel('Frequency (log scale)')
        elif plot_type == 'scatter':
            plt.figure(figsize=(10, 6))
            x = list(range(len(self.ranks)))
            y = self.ranks
            plt.scatter(x, y, alpha=0.5)
            plt.title('Scatter Plot of Token Ranks')
            plt.xlabel('Sequence Position')
            plt.ylabel('Rank')
            plt.yscale('log')
        else:
            print("Invalid plot type specified. Please choose 'histogram' or 'scatter'.")
        plt.show()


def memory_eval(finetuned_models=None, original_model_names=None, context_sizes=[32], batch_size=16, file_path="data/data.txt", save_name="eval_plots_test"):
    accelerator = Accelerator()

    models = set()
    tokenizers = set()
    if finetuned_models is None and original_model_names is None:
        raise ValueError("model_dirs and original_model_names cannot both be None")

    if finetuned_models is None:
        finetuned_models = []
    if original_model_names is None:
        original_model_names = []

    for model in finetuned_models:
        models.add(model)
        tokenizers.add(finetuned_models[model])
    for model in original_model_names:
        models.add(model)
        tokenizers.add(model)

    models = list(models)
    tokenizers = list(tokenizers)

    print(models)
    print(tokenizers)

    zip_models: List[ZipModel] = []

    for i in range(len(models)):
        model = models[i]
        if model in finetuned_models:
            loaded_model = AutoModelForCausalLM.from_pretrained(f"/LLMZip/finetuned_models/{model}", torch_dtype=torch.float16 , device_map="auto")
            loaded_tokenizer = AutoTokenizer.from_pretrained(finetuned_models[model])
            tokenizer_name = finetuned_models[model]
        else:
            loaded_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)
            loaded_tokenizer = AutoTokenizer.from_pretrained(model)
            tokenizer_name = model

        loaded_model, loaded_tokenizer = accelerator.prepare(loaded_model, loaded_tokenizer)

        for context_size in context_sizes:
            zip_model = ZipModel(model, tokenizer_name, loaded_model, loaded_tokenizer, model in finetuned_models, context_size, batch_size)
            zip_models.append(zip_model)
            start = time.time()
            zip_model.zip_file(file_path, f"/LLMZip/zipped/{zip_model.model_name.replace('/', '-')}_{zip_model.CONTEXT_SIZE}_{zip_model.BATCH_SIZE}_8bit.gpz")
            end = time.time()

            with open(f"/LLMZip/zipped/{zip_model.model_name.replace('/', '-')}_{zip_model.CONTEXT_SIZE}_{zip_model.BATCH_SIZE}_8bit.txt", "w") as f:
                for rank in zip_model.ranks:
                    f.write(f"{rank}\n")
                f.write(f"{end-start}")

        del loaded_model
        torch.cuda.empty_cache()

    ranks_0_15 = {}
    for zip_model in zip_models:
        name = f"{zip_model.model_name}_{zip_model.CONTEXT_SIZE}"
        ranks_0_15[name] = []
        for rank in zip_model.ranks:
            if rank >= 0 and rank <= 15:
                ranks_0_15[name].append(rank)
        ranks_0_15[name] = (len(ranks_0_15[name])/len(zip_model.ranks))*100

    ranks_0 = {}
    for zip_model in zip_models:
        name = f"{zip_model.model_name}_{zip_model.CONTEXT_SIZE}"
        ranks_0[name] = []
        for rank in zip_model.ranks:
            if rank == 0:
                ranks_0[name].append(rank)
        ranks_0[name] = (len(ranks_0[name])/len(zip_model.ranks))*100

    compression_ratios = {}
    for zip_model in zip_models:
        name = f"{zip_model.model_name}_{zip_model.CONTEXT_SIZE}"
        compressed_size = os.path.getsize(f"/LLMZip/zipped/{zip_model.model_name.replace('/', '-')}_{zip_model.CONTEXT_SIZE}_{zip_model.BATCH_SIZE}_8bit.gpz")
        dataset_size = os.path.getsize(file_path)
        compression_ratios[name] = compressed_size/dataset_size

    pprint.pprint(ranks_0_15)
    pprint.pprint(ranks_0)
    pprint.pprint(compression_ratios)

    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
    color_mapping = {tokenizer: next(colors) for tokenizer in tokenizers}

    fig, axs = plt.subplots(1, 3, figsize=(30, 6), sharex=True)
    axs[0].set_title('Percentage of Ranks Between 0-15')
    axs[1].set_title('Percentage of Ranks at 0')
    axs[2].set_title('Compression Ratio')

    for ax in axs:
        ax.set_xlabel('Context Size')
    axs[0].set_ylabel('Percentage')
    axs[1].set_ylabel('Percentage')
    axs[2].set_ylabel('Compression Ratio')

    context_sizes = [32, 64, 128, 256, 512]
    for ax in axs:
        ax.set_xticks(context_sizes)

    plot_data = {'ranks_0_15': ranks_0_15, 'ranks_0': ranks_0, 'compression_ratios': compression_ratios}

    for metric, ax in zip(['ranks_0_15', 'ranks_0', 'compression_ratios'], axs):
        model_data_points = {}
        for zip_model in zip_models:
            model_name = zip_model.model_name
            if model_name not in model_data_points:
                model_data_points[model_name] = {'x': [], 'y': [], 'finetuned': zip_model.finetuned, 'tokenizer': zip_model.tokenizer_name}
            percentage = plot_data[metric].get(f"{model_name}_{zip_model.CONTEXT_SIZE}", 0)
            model_data_points[model_name]['x'].append(zip_model.CONTEXT_SIZE)
            model_data_points[model_name]['y'].append(percentage)

        for model_name, data in model_data_points.items():
            linestyle = "--" if data['finetuned'] else "-"
            color = color_mapping[data['tokenizer']]
            ax.plot(data['x'], data['y'], label=model_name, linestyle=linestyle, color=color, marker='o')

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(f'/LLMZip/plots/{save_name}.png')


if __name__ == "__main__":

    login("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")

    # setup()
    finetuned_models = {
        # "meta-llama-Meta-Llama-3-8B-enwik8": "meta-llama/Meta-Llama-3-8B",
        # "gpt2-medium-finetune": "gpt2-medium",
        # "gpt2-large-finetune": "gpt2-large",
        # "data-models-Llama-2-70b-hf--enwik10mb_64_r16_8bit": "meta-llama/Llama-2-70b-hf",
        # "meta-llama-Meta-Llama-3-8B-enwik10mb_64_r16": "meta-llama/Meta-Llama-3-8B",
        # "mistralai-Mistral-7B-Instruct-v0.2-enwik10mb_16_r16": "mistralai/Mistral-7B-Instruct-v0.2"
        # "gpt2-100mb": "gpt2",
        # "gpt2-medium-100mb": "gpt2-medium",
        # "gpt2-large-100mb": "gpt2-large",
        # "gpt2-xl-100mb": "gpt2-xl"
    }
    # original_model_names = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    # original_model_names = ["tiiuae/falcon-7b"]
    original_model_names = ["meta-llama/Llama-2-13b-hf"]
    context_sizes = [512]

    # eval(finetuned_models=finetuned_models, original_model_names=original_model_names, context_sizes=context_sizes, batch_size=16, file_path="data.txt")
    memory_eval(finetuned_models=finetuned_models, original_model_names=original_model_names, context_sizes=context_sizes, batch_size=16, file_path="/LLMZip/datasets/enwik10mb.txt", save_name="llama2-13b-base-load-in-16bit")
    # memory_eval(finetuned_models=None, original_model_names=original_model_names, context_sizes=context_sizes, batch_size=16, file_path="datasets/enwik10mb.txt", save_name="llama3-enwik10mb-base")
    # memory_eval(finetuned_models=finetuned_models, original_model_names=None, context_sizes=[2048], batch_size=16, file_path="datasets/large_data.txt", save_name="eval_plots_new")