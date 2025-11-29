# Configuration du Projet - Workshop LLM Coding

## Chemins des Fichiers

### Données
DATA_DIR = "data"
REASONING_DATASET = "data/python_reasoning_dataset.jsonl"

### Modèles
MODELS_DIR = "models"
PRE_TRAINING_DIR = "models/pre_training"
POST_TRAINING_DIR = "models/post_training"

# Modèle pré-entraîné
PRETRAINED_MODEL = "models/pre_training/mini_gpt_code_FINAL.pt"
PRETRAINED_TOKENIZER = "models/pre_training/tokenizer"

# Modèle post-entraîné
SFT_MODEL = "models/post_training/mini_gpt_sft_FINAL.pt"
SFT_TOKENIZER = "models/post_training/tokenizer"

### Sorties
OUTPUTS_DIR = "outputs"
CORPUS_FILE = "outputs/mini_corpus_mixed.txt"

## Hyperparamètres

### Pre-Training
PRE_TRAINING = {
    "sample_size": 100000,      # Nombre de documents
    "block_size": 256,          # Longueur des séquences
    "batch_size": 16,           # Taille des batches
    "n_epochs": 3,              # Nombre d'époques
    "learning_rate": 3e-4,      # Taux d'apprentissage
    "n_embd": 256,              # Dimension des embeddings
    "n_head": 4,                # Nombre de têtes d'attention
    "n_layer": 4,               # Nombre de couches
    "dropout": 0.1              # Taux de dropout
}

### Post-Training (SFT)
POST_TRAINING = {
    "batch_size": 8,            # Plus petit batch (données plus riches)
    "n_epochs": 5,              # Plus d'époques
    "learning_rate": 1e-4,      # Learning rate plus faible
    "max_length": 256,          # Longueur max des séquences
}

## Datasets HuggingFace

THE_STACK = {
    "name": "bigcode/the-stack-smol",
    "data_dir": "data/python",
    "split": "train"
}

COSMOPEDIA = {
    "name": "HuggingFaceTB/smollm-corpus",
    "subset": "cosmopedia-v2",
    "split": "train"
}

## Tokens Spéciaux (SFT)

SPECIAL_TOKENS = [
    "<instruction>",
    "<reasoning>",
    "<answer>"
]

## Génération

GENERATION = {
    "temperature": 0.7,         # Contrôle la créativité
    "top_k": 50,                # Top-k sampling
    "max_new_tokens": 150       # Nombre max de tokens à générer
}
