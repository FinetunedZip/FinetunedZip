import pprint
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from utils.finetune_utils import CastOutputToFloat, print_trainable_parameters
from huggingface_hub import login

def finetune(model, save_path, dataset_path="datasets/enwik8.txt", block_size=128, epochs=10, r=8, learning_rate=1e-3, batch_size=8):
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model, 
        # load_in_8bit=True, 
        device_map={"":torch.cuda.current_device()}
        # device_map="auto"
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

    loaded_model.to(torch.cuda.current_device())

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    print("Dataset loaded")

    trainer = Trainer(
        model=loaded_model, 
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=4, 
            gradient_accumulation_steps=4,
            max_steps=epochs, 
            learning_rate=learning_rate, 
            fp16=True,
            logging_steps=1,
            output_dir="output",
        ),
        data_collator=data_collator
    )
    loaded_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

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
        # "meta-llama/Llama-2-7b-hf",
        # "meta-llama/Llama-2-70b-hf"
        "meta-llama/Meta-Llama-3-8B",
        # "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    epoch_list = [1, 4, 16]
    # dataset_paths = ["datasets/enwik64mb.txt", "datasets/enwik16mb.txt", "datasets/enwik4mb.txt"]

    for model in finetune_list:
        for e in epoch_list:
            finetune(
                model, 
                save_path=f"finetuned_models/{model.replace('/', '-')}-enwik10mb", 
                dataset_path="datasets/enwik10mb.txt", 
                block_size=128, 
                epochs=e, 
                r=8,
                learning_rate=1e-3, 
                batch_size=8
            )

    # for model in finetune_list:
    #     finetune(
    #         model, 
    #         save_path=f"finetuned_models/{model.replace('/', '-')}-enwik10mb", 
    #         dataset_path="datasets/enwik10mb.txt", 
    #         block_size=128, 
    #         epochs=1, 
    #         learning_rate=1e-3, 
    #         batch_size=8
    #     )

    # finetune("gpt2", save_path="finetuned_models/gpt2-test", dataset_path="datasets/text8.txt", block_size=128, epochs=10, learning_rate=1e-3, batch_size=8)