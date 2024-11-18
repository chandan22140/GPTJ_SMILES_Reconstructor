#!/usr/bin/env python3
# eval.py

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned GPT-J (6B) model on molecular generation tasks.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model directory.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the test data file.')
    parser.add_argument('--output_path', type=str, default='generated_smiles.txt', help='Path to save generated SMILES.')
    args = parser.parse_args()
    return args

def load_test_data(file_path):
    test_data = []
    with open(file_path, 'r') as f:
        entry = {}
        for line in f:
            if line.startswith("Description:"):
                entry['description'] = line.strip()
            elif line.startswith("Masked SMILES:"):
                entry['masked_smiles'] = line.strip().split(": ")[1]
            elif line.startswith("Reconstruct the SMILES string."):
                continue
            elif line.strip().endswith("###"):
                entry['expected_smiles'] = line.strip().replace("###", "")
                test_data.append(entry)
                entry = {}
            else:
                entry['expected_smiles'] = line.strip()
    return test_data

def is_valid(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def calculate_fragment_similarity(generated, reference):
    gen_frags = set()
    ref_frags = set()
    
    for smiles in generated:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            frags = Chem.GetMolFrags(mol, asMols=True)
            for frag in frags:
                gen_frags.add(Chem.MolToSmiles(frag, isomericSmiles=True))
    
    for smiles in reference:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            frags = Chem.GetMolFrags(mol, asMols=True)
            for frag in frags:
                ref_frags.add(Chem.MolToSmiles(frag, isomericSmiles=True))
    
    intersection = gen_frags.intersection(ref_frags)
    union = gen_frags.union(ref_frags)
    similarity = len(intersection) / len(union) if len(union) > 0 else 0
    return similarity

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    print("Loading test data...")
    test_data = load_test_data(args.test_data)
    
    generated_smiles = []
    reference_smiles = []

    print("Generating SMILES...")
    for data in tqdm(test_data, desc="Generating"):
        description = data['description']
        masked_smiles = data['masked_smiles']
        expected = data['expected_smiles']

        input_text = f"{description}\nMasked SMILES: {masked_smiles}\nReconstruct the SMILES string."
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=512,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=1.0,
                top_p=0.95,
                do_sample=False
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated = output_text.replace(input_text, '').strip().split('\n')[0]
        
        generated_smiles.append(generated)
        reference_smiles.append(expected)

    print(f"Saving generated SMILES to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        for smiles in generated_smiles:
            f.write(f"{smiles}\n")

    print("Evaluating Validity...")
    valid_smiles = [s for s in generated_smiles if is_valid(s)]
    validity = len(valid_smiles) / len(generated_smiles) if len(generated_smiles) > 0 else 0
    print(f"Validity: {validity * 100:.2f}%")
    
    print("Evaluating Fragment Similarity...")
    fragment_similarity = calculate_fragment_similarity(generated_smiles, reference_smiles)
    print(f"Fragment Similarity (Jaccard Index): {fragment_similarity * 100:.2f}%")
    
    with open("evaluation_metrics.txt", "w") as f:
        f.write(f"Validity: {validity * 100:.2f}%\n")
        f.write(f"Fragment Similarity (Jaccard Index): {fragment_similarity * 100:.2f}%\n")
    
    print("Evaluation completed. Metrics saved to evaluation_metrics.txt.")

if __name__ == "__main__":
    main()

