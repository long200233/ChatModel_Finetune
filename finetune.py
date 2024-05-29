from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os

# Project Config
project_name = 'try-finetune' # @param {type:"string"}
model_name = 'teknium/OpenHermes-2.5-Mistral-7B' # @param {type:"string"}

push_to_hub = True # @param ["False", "True"] {type:"raw"}
read_token = "YOUR_READ_TOKEN"
write_token = "YOUR_WRITE"
repo_id = "DrAgOn200233/OpenHermes-2.5-Mistral-7B-ArthurHayes" #@param {type:"string"}

#@markdown ---
#@markdown #### Hyperparameters
learning_rate = 2e-4 # @param {type:"number"}
num_epochs = 3 #@param {type:"number"}
batch_size = 12 # @param {type:"slider", min:1, max:32, step:1}
block_size = 1024 # @param {type:"number"}
trainer = "sft" # @param ["default", "sft"] {type:"raw"}
warmup_ratio = 0.1 # @param {type:"number"}
weight_decay = 0.01 # @param {type:"number"}
gradient_accumulation = 4 # @param {type:"number"}
mixed_precision = "fp16" # @param ["fp16", "bf16", "none"] {type:"raw"}
peft = True # @param ["False", "True"] {type:"raw"}
quantization = "int4" # @param ["int4", "int8", "none"] {type:"raw"}
lora_r = 16 #@param {type:"number"}
lora_alpha = 32 #@param {type:"number"}
lora_dropout = 0.05 #@param {type:"number"}

os.environ["PROJECT_NAME"] = project_name
os.environ["MODEL_NAME"] = model_name
os.environ["PUSH_TO_HUB"] = str(push_to_hub)
os.environ["READ_TOKEN"] = read_token
os.environ["WRITE_TOKEN"] = write_token
os.environ["REPO_ID"] = repo_id
os.environ["TRAINER"] = trainer
os.environ["LEARNING_RATE"] = str(learning_rate)
os.environ["NUM_EPOCHS"] = str(num_epochs)
os.environ["BATCH_SIZE"] = str(batch_size)
os.environ["BLOCK_SIZE"] = str(block_size)
os.environ["WARMUP_RATIO"] = str(warmup_ratio)
os.environ["WEIGHT_DECAY"] = str(weight_decay)
os.environ["GRADIENT_ACCUMULATION"] = str(gradient_accumulation)
os.environ["MIXED_PRECISION"] = str(mixed_precision)
os.environ["PEFT"] = str(peft)
os.environ["QUANTIZATION"] = str(quantization)
os.environ["LORA_R"] = str(lora_r)
os.environ["LORA_ALPHA"] = str(lora_alpha)
os.environ["LORA_DROPOUT"] = str(lora_dropout)


tokenizer = AutoTokenizer.from_pretrained("Gryphe/MythoMist-7b")
model = AutoModelForCausalLM.from_pretrained("Gryphe/MythoMist-7b")
    


def tokenize_function(example, tokenizer):
    return tokenizer(example['text'], truncation=True)

def main():
    data_files = {"train": "arthur_lora.csv"}
    dataset = load_dataset("DrAgOn200233/ArthurHayes", token=read_token)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    

if __name__  == '__main__':
    main()