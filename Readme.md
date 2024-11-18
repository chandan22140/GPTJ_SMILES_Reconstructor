# Molecular SMILES Generation and Evaluation Project

This project involves fine-tuning, inference, and evaluation of a GPT-J (6B) language model for molecular generation tasks. Specifically, it focuses on reconstructing SMILES (Simplified Molecular Input Line Entry System) strings from masked inputs based on molecular descriptions. 

## Project Structure

The repository contains the following scripts:
- **`eval.py`**: Evaluates the fine-tuned GPT-J model using molecular generation metrics like validity and fragment similarity.
- **`inference.py`**: Runs inference on new data to generate SMILES strings based on given molecular descriptions and masked inputs.
- **`training.py`**: Fine-tunes the GPT-J model with LoRA (Low-Rank Adaptation) using 4-bit quantization to improve training efficiency.

---

## Requirements

### Python Libraries
- `torch`
- `transformers`
- `datasets`
- `rdkit`
- `tqdm`
- `peft`

### Hardware Requirements
- A GPU-enabled system is highly recommended for fine-tuning and inference.
- Support for 4-bit quantization requires CUDA-compatible hardware.

---

## Setup

1. Clone the repository and navigate to the directory.
2. Install the required Python packages:
   ```bash
   pip install torch transformers datasets rdkit tqdm peft
   ```

3. Ensure CUDA is available:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## Usage

### 1. **Fine-Tuning the Model**
Run `training.py` to fine-tune the GPT-J model. Ensure the dataset file `data.json` is present in the working directory.
```bash
python training.py
```

**Key Outputs**:
- Fine-tuned model saved in the specified output directory.
- Training metrics.

### 2. **Inference**
Run `inference.py` to generate SMILES strings for new molecular descriptions and masked SMILES strings.
```bash
python inference.py
```

**Input**: Molecular descriptions and masked SMILES strings.  
**Output**: Reconstructed SMILES strings.

### 3. **Evaluation**
Run `eval.py` to evaluate the fine-tuned model on test data.
```bash
python eval.py --model_path path_to_finetuned_model --test_data test_data_file.txt --output_path generated_smiles.txt
```

**Key Metrics**:
- **Validity**: Percentage of chemically valid SMILES strings.
- **Fragment Similarity**: Jaccard Index for molecular fragment similarity.

**Outputs**:
- Generated SMILES in `generated_smiles.txt`.
- Evaluation metrics in `evaluation_metrics.txt`.

---

## Key Features

1. **4-Bit Quantization**: Efficient model fine-tuning with low memory overhead.
2. **LoRA Adaptation**: Enhances the model's ability to handle specific tasks with minimal parameter updates.
3. **Chemical Validation**: Ensures output SMILES strings are chemically valid using RDKit.
4. **Fragment Similarity**: Compares molecular fragments to evaluate the quality of generation.

---

## Dataset Format

The training and test datasets are expected to follow a specific format:

**Training Data (JSON)**:
```json
{
  "molecule_id1": {
    "smiles": "CC(C)c1cccc(c1)c2cccc3c2NC(=O)N(C3)C",
    "Description": "Molecular weight: ...; Rotatable bonds: ...; ..."
  },
  ...
}
```

**Test Data (Plain Text)**:
```
Description: Molecular weight: ...; Rotatable bonds: ...; ...
Masked SMILES: C[C](C)c1cccc...
Reconstruct the SMILES string.
###
```

---

## Output Example

**Input Description**:
```
Description: Molecular weight: 280.37 g/mol; Rotatable bonds: 2; Aromatic ring count: 2
Masked SMILES: CC(C)c1c[...]
```

**Generated SMILES**:
```
CC(C)c1cccc(c1)c2cccc3c2NC(=O)N(C3)C
```

---


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
