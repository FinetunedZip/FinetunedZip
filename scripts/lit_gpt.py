import pprint
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from huggingface_hub import login
from accelerate import Accelerator

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def finetune(model, save_path, dataset_path="datasets/enwik8.txt", block_size=128, epochs=10, r=8, learning_rate=1e-3, batch_size=8):
    accelerator = Accelerator()

    loaded_model = AutoModelForCausalLM.from_pretrained(
        model,
        load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    print("Model loaded")

    for param in loaded_model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    loaded_model.gradient_checkpointing_enable()  # reduce number of stored activations
    loaded_model.enable_input_require_grads()

    loaded_model.lm_head = CastOutputToFloat(loaded_model.lm_head)

    config = LoraConfig(
        r=r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    loaded_model = get_peft_model(loaded_model, config)
    print_trainable_parameters(loaded_model)

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    print("Dataset loaded")

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        max_steps=epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        output_dir="output",
        save_total_limit=2,
        save_steps=10,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir=f"/LLMZip/output/logs",
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=loaded_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    loaded_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # Prepare everything with `accelerate`
    trainer = accelerator.prepare(trainer)
    accelerator.wait_for_everyone()

    trainer.train()
    trainer.save_model(save_path + f"_{epochs}_r{r}")

    del loaded_model
    del trainer
    torch.cuda.empty_cache()

    print("Finished finetuning")

if __name__ == "__main__":

    login("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")  # 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

    finetune_list = [
        # "gpt2-xl",
        "data/models/Llama-2-70b-hf/",
        # "meta-llama/Llama-2-70b-hf"
        # "meta-llama/Meta-Llama-3-8B",
        # "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    r_list = [16]
    # dataset_paths = ["datasets/enwik64mb.txt", "datasets/enwik16mb.txt", "datasets/enwik4mb.txt"]

    for model in finetune_list:
        for r in r_list:
            finetune(
                model,
                save_path=f"/LLMZip/finetuned_models/{model.replace('/', '-')}-enwik10mb",
                dataset_path="/LLMZip/datasets/enwik10mb.txt",
                block_size=128,
                epochs=64,
                r=16,
                learning_rate=1e-3,
                batch_size=8
            )
