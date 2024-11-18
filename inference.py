from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import random
import os

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
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token

import torch
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# Load the base quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    quantization_config=bnb_config,
    device_map='auto',
)

# Load the LoRA adapters
model = PeftModel.from_pretrained(model, "fine-tuned-gpt-j-6B-peft")

def mask_smiles(smiles, mask_token='[MASK]', mask_prob=0.15):
    tokens = list(smiles)
    num_tokens_to_mask = max(1, int(len(tokens) * mask_prob))
    mask_indices = random.sample(range(len(tokens)), num_tokens_to_mask)
    for idx in mask_indices:
        tokens[idx] = mask_token
    masked_smiles = ''.join(tokens)
    return masked_smiles


# Example input
description = "Molecular weight: 280.37 g/mol; Rotatable bonds: 2; Contains amide group; Contains nitrogen atom(s); Aromatic ring count: 2"
masked_smiles = "CC(C)c1cccc(c1)c2cccc3c2NC(=O)N(C3)C"  # Example SMILES string
masked_smiles = mask_smiles(masked_smiles)  # Apply masking as per your function

input_text = f"Description: {description}\nMasked SMILES: {masked_smiles}\nReconstruct the SMILES string.\n"

# Tokenize input
inputs = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

# Decode the generated SMILES string
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_smiles = generated_text[len(input_text):].strip()
print("Generated SMILES:", generated_smiles)
