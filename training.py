import os

# 0. **Set Environment Variables Early**
os.environ["CUDA_VISIBLE_DEVICES"] = "4,2,3"  # RTX A6000, RTX 3090, RTX 3090
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism to avoid warnings

import json
import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

# 1. **Data Preparation**

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

input_texts = []
target_texts = []

def mask_smiles(smiles, mask_token='[MASK]', mask_prob=0.15):
    tokens = list(smiles)
    num_tokens_to_mask = max(1, int(len(tokens) * mask_prob))
    mask_indices = random.sample(range(len(tokens)), num_tokens_to_mask)
    for idx in mask_indices:
        tokens[idx] = mask_token
    masked_smiles = ''.join(tokens)
    return masked_smiles

for molecule_id, molecule_data in data.items():
    smiles = molecule_data['smiles']
    description = molecule_data['Description']

    # Mask the SMILES string
    masked_smiles = mask_smiles(smiles)

    # Create input and target texts
    input_text = f"Description: {description}\nMasked SMILES: {masked_smiles}\nReconstruct the SMILES string."
    target_text = smiles

    input_texts.append(input_text)
    target_texts.append(target_text)

# Save the data in a format suitable for fine-tuning
with open('train.txt', 'w') as f:
    for input_text, target_text in zip(input_texts, target_texts):
        f.write(f"{input_text}\n{target_text}\n###\n")

# 2. **Model Selection and Loading with 4-Bit Quantization and PEFT (LoRA)**

# Define 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',  # 'nf4' is recommended for better performance
    bnb_4bit_use_double_quant=True,  # Optional: Improves accuracy
    bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 for better precision
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    quantization_config=bnb_config,
    device_map='auto',  # Automatically maps layers to available GPUs
)

# **Prepare the model for k-bit (4-bit) training**
model = prepare_model_for_kbit_training(model)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank of LoRA adapters
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    target_modules=["q_proj", "v_proj"],  # Targeted modules in GPT-J
    task_type=TaskType.CAUSAL_LM,  # Task type
)

# Integrate LoRA adapters with the model
model = get_peft_model(model, lora_config)

# Optional: Print trainable parameters to verify LoRA integration
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
print(f"Trainable parameters:\n{trainable_params}")
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 3. **Data Tokenization**

def read_data(file_path):
    data = {'text': []}
    with open(file_path, 'r') as f:
        sample = ''
        for line in f:
            if line.strip() == '###':
                data['text'].append(sample.strip())
                sample = ''
            else:
                sample += line
    return data

data_dict = read_data('train.txt')
dataset = Dataset.from_dict(data_dict)

# Split dataset into train and validation
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,  # Reduced sequence length to save memory
    )

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 4. **Custom Data Collator**

def data_collator(features):
    batch = tokenizer.pad(
        features,
        padding=True,  # Ensure padding is applied
        return_tensors='pt'
    )

    labels = batch['input_ids'].clone()

    # Identify the end of the input prompt to mask labels accordingly
    separator = "Reconstruct the SMILES string."
    separator_token_ids = tokenizer.encode(separator, add_special_tokens=False)
    if len(separator_token_ids) == 0:
        raise ValueError("Separator token not found in tokenizer.")
    separator_token_id = separator_token_ids[-1]

    for i in range(len(labels)):
        input_ids = batch['input_ids'][i]
        sep_indices = (input_ids == separator_token_id).nonzero(as_tuple=True)[0]
        if len(sep_indices) > 0:
            sep_index = sep_indices[0].item()
            label_start = sep_index + 1
            labels[i][:label_start] = -100  # Mask input tokens
        else:
            labels[i][:] = -100  # If separator not found, mask all labels

    batch['labels'] = labels
    return batch

# 5. **Training Arguments and Trainer Setup**

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Low batch size to reduce memory usage
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Increase to maintain effective batch size
    eval_strategy="steps",  # Updated from `evaluation_strategy`
    eval_steps=200,
    save_steps=400,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=True,  # Mixed precision for memory efficiency
    dataloader_num_workers=2,  # Adjust based on your CPU cores
    save_total_limit=2,  # Limit the number of saved checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

# 6. **Model Fine-Tuning**

trainer.train()
