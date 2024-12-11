import pandas as pd
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from prompts import prompt_training_1, prompt_eval_1
from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch


def load_sample(name):
    return pd.read_parquet(
        f"hf://datasets/cardiffnlp/databench/data/{name}/sample.parquet"
    )

def load_qa(**kwargs) -> Dataset:
    return load_dataset(
        "cardiffnlp/databench", **{"name": "qa", "split": "train", **kwargs}
    )

def dataframe_to_comma_separated_string(df):
    return df.to_csv(index=False, header=True).strip()

######################################################
######################################################
train_data = load_qa(name="semeval", split="train")
devtest_data = load_qa(name="semeval", split="dev")
split = devtest_data.train_test_split(test_size=0.5, shuffle=False)
dev_data = split["train"]
test_data = split["test"]
output_dir = ""
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
######################################################
######################################################

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name = "unsloth/Qwen2.5-0.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
        load_in_4bit = load_in_4bit,
        cache_dir="/data/datasets/hf_cache"
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func_train(examples):
    question      = examples["question"]
    dataset       = examples["dataset"]
    output        = examples["sample_answer"]
    # texts = []
    # for question, dataset, output in zip(questions, datasets, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
    # print("Dataset: ", dataset)
    # print("Question: ", question)
    # print("Output: ", output)
    dataset_here = load_sample(dataset)
    csv_dataset_here = dataframe_to_comma_separated_string(dataset_here)
    text = prompt_training_1.format(csv=csv_dataset_here, question=question, answer=output) + EOS_TOKEN
    # texts.append(text)
    return {"text": text,}

def formatting_prompts_func_eval(examples):
    question      = examples["question"]
    dataset       = examples["dataset"]
    # texts = []
    # for question, dataset, output in zip(questions, datasets, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
    # print("Dataset: ", dataset)
    # print("Question: ", question)
    # print("Output: ", output)
    dataset_here = load_sample(dataset)
    csv_dataset_here = dataframe_to_comma_separated_string(dataset_here)
    text = prompt_eval_1.format(csv=csv_dataset_here, question=question)
    return {"text": text,}


# dataset = load_dataset("cardiffnlp/databench", name="semeval", split="train")
# dataset = dataset.map(formatting_prompts_func)#, batched=True,)
# dataset.to_csv("data/training/train.csv")
dataset_valtest = load_dataset("cardiffnlp/databench", name="semeval", split="dev")
dataset_val = dataset_valtest.train_test_split(test_size=0.5, shuffle=False)["train"]
dataset_test = dataset_valtest.train_test_split(test_size=0.5, shuffle=False)["test"]
dataset_val = dataset_val.map(formatting_prompts_func_eval)#, batched=True,)
dataset_test = dataset_test.map(formatting_prompts_func_eval)#, batched=True,)
dataset_val.to_csv("data/training/val.csv")
dataset_test.to_csv("data/training/test.csv")
# dataset_train = load_dataset("csv", data_files="data/training/train.csv", split="train")
# dataset_val = load_dataset("csv", data_files="data/training/val.csv", split="train")
# dataset_test = load_dataset("csv", data_files="data/training/test.csv", split="train")


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 0,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "train_outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

#@title Show current memory stats
# gpu_stats = torch.cuda.get_device_properties(0)
# start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
# print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
# print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained("models/lora_model") # Local saving
tokenizer.save_pretrained("models/lora_model")