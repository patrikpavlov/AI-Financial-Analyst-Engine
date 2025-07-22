###In Google Colab, you can run the following commands to set up the environment for fine-tuning the Llama 3 8B model:
#!pip uninstall -y transformers accelerate peft trl bitsandbytes datasets fsspec sentence-transformers
#!pip install -q datasets==2.19.0
#!pip install -q transformers==4.41.2
#!pip install -q peft==0.11.1
#!pip install -q bitsandbytes==0.43.1
#!pip install -q accelerate==0.30.1
#!pip install -q trl==0.8.6
#!pip install -q triton==2.3.0

### This script is designed to fine-tune the Llama 3 8B model on financial news sentiment analysis using the Hugging Face Hub.

# Import Libraries and Authenticate
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login, HfApi

# Authenticate huggingface_hub
login()


# Configuration 
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  
DATASET_NAME = "zeroshot/twitter-financial-news-sentiment"

hub_model_id = "patrikpavlov/llama-finance-sentiment"

# Load Tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id 


# Pre-process the Dataset with Llama 3's Chat Template
def create_prompt(sample):
    label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    user_message = f"What is the sentiment of the following financial news?\n\nInput:\n{sample['text']}"
    assistant_response = label_map[sample['label']]

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

dataset = load_dataset(DATASET_NAME, split="train")
formatted_dataset = dataset.map(create_prompt)


# Setup Quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False


# Setup PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)



# Setup Training Arguments
training_args = TrainingArguments(
    output_dir=hub_model_id,  
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch", 
    logging_steps=10,
    num_train_epochs=1,
    report_to="none",
    fp16=True,
    push_to_hub=True, 
)

# Create the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# Train the Model
trainer.train()
print("Training finished!")