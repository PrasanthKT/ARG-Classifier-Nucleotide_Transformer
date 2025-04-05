# Antibiotic Resistant Gene Classifier

### Title
Classification of Antibiotic Resistance Gene Sequences using Nucleotide Transformer

### Objective
This project aims to fine-tune the Nucleotide Transformer model for the classification of antibiotic resistance genes (ARGs). By leveraging a pre-trained transformer architecture on labeled genomic data (ARGs, non-ARGs, and synthetic sequences), the model is optimized to accurately identify ARGs across diverse bacterial species, enabling faster and more reliable antimicrobial resistance prediction.

### Dependencies
1. transformers==4.38.1
2. datasets==2.18.0
3. scikit-learn==1.4.1
4. pandas==2.2.1
5. numpy==1.26.4
6. torch==2.2.1
7. matplotlib==3.8.3
8. seaborn==0.13.2

```
conda create -n transformer python=3.10
conda activate transformer
```
```
pip install transformers datasets scikit-learn pandas numpy torch matplotlib seaborn
```
### Download the model
The fine-tuned ARG Classifier model (based on the Nucleotide Transformer) is hosted via Google Drive due to its large file size (~1.9GB), which exceeds GitHub’s file upload limit.

Download the model folder (arg_classifier, https://drive.google.com/drive/folders/1CR0PMbnJ1DyLB7ej9gtPYOSTtkBRHn7A?usp=share_link )
After downloading, extract the folder and place it in the root directory of this repository.

```
arg_classifier/
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── vocab.txt
```

### Run the Model
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_path = "arg_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example sequence
sequence = "ATGCGTACGTAGCTAGCTAGCTAGCGTATCGTAGCTAGT"

# Tokenize input
inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

# Run prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted Label: {'Resistant' if prediction == 1 else 'Non-Resistant'}")
```
### Output
