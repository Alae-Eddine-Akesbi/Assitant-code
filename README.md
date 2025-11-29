# 🧠 Mini-GPT: Coding LLM from Scratch

<div align="center">

**Un modèle de langage spécialisé pour la génération de code Python**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

[Installation](#-installation) • [Quick Start](#-quick-start) • [Dashboard](#-dashboard) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Structure du Projet](#-structure-du-projet)
- [Pipeline d'Entraînement](#-pipeline-dentraînement)
- [Dashboard Interactif](#-dashboard-interactif)
- [Architecture](#-architecture)
- [Résultats](#-résultats)
- [Utilisation Avancée](#-utilisation-avancée)
- [Troubleshooting](#-troubleshooting)
- [Contribution](#-contribution)
- [Licence](#-licence)

---

## 🎯 À Propos

Ce projet implémente un **modèle de langage transformer de type GPT** entraîné from scratch pour la génération de code Python. Le modèle suit un pipeline d'entraînement en 3 phases :

1. **Pre-Training** : Apprentissage non supervisé sur du code Python (CLM)
2. **Post-Training** : Fine-tuning supervisé avec des instructions (SFT)
3. **Alignment** : Optimisation par préférences humaines (RLHF) - *À venir*

### Caractéristiques Principales

- ✅ **Architecture GPT** : Decoder-only Transformer (4 layers, 256 dims)
- ✅ **~300K paramètres** : Petit mais performant
- ✅ **Training complet** : Pre-training + SFT implémentés
- ✅ **Dashboard Streamlit** : Interface de comparaison interactive
- ✅ **100% PyTorch** : Code clair et pédagogique

---

## ✨ Fonctionnalités

### Modèles Entraînés

| Modèle | Dataset | Taille | Capacité |
|--------|---------|--------|----------|
| **Pre-Training** | 100k+ fichiers Python (The Stack) | ~38 MB | Complétion de code |
| **Post-Training** | 10k paires instruction-code | ~38 MB | Génération à partir d'instructions |

### Dashboard Streamlit

- 🔄 Comparaison côte-à-côte de 3 modèles
- 📊 Graphiques interactifs (Plotly)
- 🎛️ Contrôle des paramètres de génération
- 📈 Métriques en temps réel (Loss, Perplexity, Temps)
- ✨ Interface moderne et animée

### Notebooks Complets

- 📓 `1_pre_training.ipynb` : Entraînement CLM détaillé
- 📓 `2_post_training.ipynb` : SFT avec instructions structurées
- 📚 Documentation extensive avec explications pédagogiques

---

## 🔧 Installation

### Prérequis

- Python 3.8+
- CUDA (optionnel, pour GPU)
- ~5 GB d'espace disque

### Installation Rapide

```bash
# Cloner le repo
git clone https://github.com/votre-repo/mini-gpt-coding.git
cd mini-gpt-coding

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Installer Streamlit pour le dashboard
pip install streamlit plotly
```

### Dépendances Principales

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
huggingface-hub>=0.15.0
streamlit>=1.28.0
plotly>=5.17.0
```

---

## 🚀 Quick Start

---

## 🚀 Quick Start

### 1️⃣ Entraînement Complet (Recommandé)

```bash
# 1. Pre-Training (CLM sur code Python)
jupyter notebook 1_pre_training.ipynb
# Exécuter toutes les cellules → Sortie: models/pre_training/

# 2. Post-Training (SFT avec instructions)
jupyter notebook 2_post_training.ipynb
# Exécuter toutes les cellules → Sortie: models/post_training/

# 3. Lancer le Dashboard
streamlit run dashboard.py
```

### 2️⃣ Utilisation Rapide (Modèles pré-entraînés)

```python
import torch
from transformers import GPT2Tokenizer

# Charger le modèle post-entraîné
checkpoint = torch.load('models/post_training/mini_gpt_sft_FINAL.pt')
tokenizer = GPT2Tokenizer.from_pretrained('models/post_training/tokenizer')

# Générer du code
prompt = "<instruction> Write a function to calculate factorial <reasoning>"
inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(inputs, max_new_tokens=150, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

---

## 📁 Structure du Projet

```
📦 mini-gpt-coding/
│
├── 📓 1_pre_training.ipynb          # Pre-Training (CLM)
├── 📓 2_post_training.ipynb         # Post-Training (SFT)
├── 🎨 dashboard.py                   # Dashboard Streamlit
│
├── ⚙️  config.py                      # Configuration centralisée
├── 📦 requirements.txt               # Dépendances Python
├── 📖 README.md                      # Ce fichier
│
├── 📂 data/                          # Datasets
│   └── python_reasoning_dataset.jsonl
│
├── 📂 models/                        # Modèles entraînés
│   ├── pre_training/
│   │   ├── mini_gpt_code_FINAL.pt   # ← Modèle pré-entraîné
│   │   └── tokenizer/
│   └── post_training/
│       ├── mini_gpt_sft_FINAL.pt    # ← Modèle post-entraîné
│       └── tokenizer/
│
└── 📂 outputs/                       # Artefacts temporaires
    └── mini_corpus_mixed.txt
```

---

## 🔄 Pipeline d'Entraînement

### Phase 1: Pre-Training (CLM)

**Objectif** : Apprendre la syntaxe Python et les patterns de code

```python
# Dataset: The Stack (100k+ fichiers Python)
# Méthode: Causal Language Modeling
# Durée: ~30 min (CPU) | ~5 min (GPU)

# Résultat:
# ✅ Validation Loss: ~2.3
# ✅ Perplexity: ~10.4
# ✅ Capable de compléter du code Python
```

**Notebook** : `1_pre_training.ipynb`

### Phase 2: Post-Training (SFT)

**Objectif** : Apprendre à suivre des instructions

```python
# Dataset: 10k paires instruction-reasoning-code
# Format: <instruction> X <reasoning> Y <answer> Z
# Durée: ~20 min (CPU) | ~3 min (GPU)

# Résultat:
# ✅ Validation Loss: ~1.8
# ✅ Perplexity: ~6.2
# ✅ Génération à partir d'instructions naturelles
```

**Notebook** : `2_post_training.ipynb`

### Phase 3: Alignment (RLHF)

**Status** : 🚧 En développement

Prochaines étapes :
- Collecte de préférences humaines
- Entraînement d'un reward model
- Optimisation PPO

---

## 🎨 Dashboard Interactif

### Lancement

```bash
streamlit run dashboard.py
```

Le dashboard s'ouvre automatiquement à **http://localhost:8501**

### Fonctionnalités

<div align="center">

| Feature | Description |
|---------|-------------|
| 🔄 **Comparaison Multi-Modèles** | Pre-Training vs Post-Training vs Alignment |
| 🎛️ **Contrôles Dynamiques** | Temperature, Top-K, Max Tokens |
| 📊 **Graphiques Plotly** | Temps, Loss, Perplexity, Longueur |
| ✨ **Interface Animée** | Cartes hover, fade-in, slide-in |
| 📝 **Exemples Pré-définis** | Fibonacci, Factorial, QuickSort, etc. |

</div>

### Captures d'Écran

**Interface Principale**
```
┌─────────────────────────────────────────────────────────┐
│  🧠 LLM Coding Assistant Dashboard                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  💬 Entrez votre Prompt                                │
│  ┌────────────────────────────────────────────────┐    │
│  │ <instruction> Write factorial function         │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  [ 🚀 Générer avec tous les modèles ]                  │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  🔵 Pre-Training  │ 🟣 Post-Training │ 🔷 Alignment   │
│  ⏱️ 0.45s          │ ⏱️ 0.52s         │ ⏱️ N/A         │
│  📉 Loss: 2.34    │ 📉 Loss: 1.82   │ 🚧 En dev.    │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### Mini-GPT Model

```python
Mini-GPT (Decoder-only Transformer)
│
├── Token Embedding (50,260 → 256)
├── Position Embedding (256 → 256)
├── Dropout (0.1)
│
├── Transformer Blocks (x4)
│   ├── Layer Norm
│   ├── Multi-Head Attention (4 heads)
│   │   ├── Query, Key, Value projections
│   │   ├── Causal masking
│   │   └── Attention dropout
│   ├── Layer Norm
│   └── Feed-Forward Network
│       ├── Linear (256 → 1024)
│       ├── GELU activation
│       ├── Linear (1024 → 256)
│       └── Dropout
│
├── Final Layer Norm
└── Language Model Head (256 → 50,260)
    └── Weight tying with Token Embedding
```

### Spécifications Techniques

| Paramètre | Valeur |
|-----------|--------|
| **Architecture** | Decoder-only Transformer |
| **Paramètres** | ~300,000 |
| **Dimensions** | 256 |
| **Attention Heads** | 4 |
| **Layers** | 4 |
| **Context Length** | 256 tokens |
| **Vocabulaire** | 50,260 (GPT-2 + tokens spéciaux) |
| **Activation** | GELU |
| **Dropout** | 0.1 |

### Tokenizer

- **Base** : GPT-2 BPE Tokenizer (50,257 tokens)
- **Tokens spéciaux** : `<instruction>`, `<reasoning>`, `<answer>`
- **Vocabulaire final** : 50,260 tokens

---

## 📊 Résultats

### Métriques d'Entraînement

| Modèle | Dataset | Epochs | Val Loss | Perplexity | Temps (GPU) |
|--------|---------|--------|----------|------------|-------------|
| **Pre-Training** | 100k files | 3 | 2.34 | 10.4 | ~5 min |
| **Post-Training** | 10k pairs | 5 | 1.82 | 6.2 | ~3 min |

### Exemples de Génération

#### Pre-Training (Code Completion)

**Input:**
```python
def fibonacci(n):
```

**Output:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

#### Post-Training (Instruction Following)

**Input:**
```
<instruction> Write a function to calculate factorial <reasoning>
```

**Output:**
```python
<instruction> Write a function to calculate factorial 
<reasoning> Use recursive approach with base case n=0 or n=1
<answer> 
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
```

### Comparaison des Modèles

| Critère | Pre-Training | Post-Training |
|---------|--------------|---------------|
| **Complète le code** | ✅ Excellent | ✅ Excellent |
| **Suit les instructions** | ❌ Non | ✅ Oui |
| **Ajoute du raisonnement** | ❌ Non | ✅ Oui |
| **Code structuré** | ⚠️ Variable | ✅ Cohérent |

---

## 🔬 Utilisation Avancée

### Charger un Modèle Spécifique

```python
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

# Définir l'architecture (voir notebooks pour le code complet)
from model import MiniGPT  # ou copier depuis les notebooks

# Charger le checkpoint
checkpoint = torch.load('models/post_training/mini_gpt_sft_FINAL.pt')
config = checkpoint['config']

# Créer le modèle
model = MiniGPT(
    vocab_size=config['vocab_size'],
    block_size=config['block_size'],
    n_embd=config['n_embd'],
    n_head=config['n_head'],
    n_layer=config['n_layer'],
    dropout=config['dropout']
)

# Charger les poids
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Charger le tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('models/post_training/tokenizer')
```

### Génération avec Contrôle Fin

```python
def generate_code(prompt, max_tokens=150, temperature=0.7, top_k=50):
    """Génère du code avec paramètres personnalisés"""
    
    # Encoder
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Générer
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,  # 0.1-2.0 (créativité)
            top_k=top_k,              # 10-100 (diversité)
            do_sample=True
        )
    
    # Décoder
    return tokenizer.decode(output_ids[0])

# Exemples
generate_code("def quicksort(arr):", temperature=0.5)  # Plus déterministe
generate_code("class BinaryTree:", temperature=1.2)     # Plus créatif
```

### Export pour Production

```python
# Exporter uniquement les poids (plus léger)
torch.save(
    model.state_dict(), 
    'models/mini_gpt_production.pt'
)

# Quantization (réduction de taille)
import torch.quantization as quant
model_quantized = quant.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

---

## 🛠️ Troubleshooting

### Problèmes Courants

#### "CUDA out of memory"

**Solution** : Réduire `BATCH_SIZE` dans les notebooks
```python
BATCH_SIZE = 8  # au lieu de 16
```

#### "FileNotFoundError: models/pre_training/..."

**Solution** : Exécuter d'abord `1_pre_training.ipynb` complètement

#### "Dataset not accessible"

**Solution** : S'authentifier sur HuggingFace
```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN")
```

#### Dashboard lent

**Solutions** :
- Réduire `max_tokens` (50-100)
- Utiliser `temperature=0.5` (plus rapide)
- Désactiver les modèles non nécessaires

### Performance

| Device | Pre-Training | Post-Training | Génération |
|--------|--------------|---------------|------------|
| **CPU** | ~30 min | ~20 min | ~2s/sample |
| **GPU (T4)** | ~5 min | ~3 min | ~0.3s/sample |
| **GPU (V100)** | ~2 min | ~1 min | ~0.1s/sample |

---

## 🤝 Contribution

Contributions bienvenues ! Voici comment participer :

1. **Fork** le projet
2. **Créer** une branche (`git checkout -b feature/AmazingFeature`)
3. **Commit** vos changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrir** une Pull Request

### Roadmap

- [ ] Implémentation RLHF complète
- [ ] Support pour d'autres langages (JavaScript, Java)
- [ ] API REST avec FastAPI
- [ ] Docker container
- [ ] Tests unitaires
- [ ] CI/CD avec GitHub Actions

---

## 📚 Ressources

### Documentation

- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Streamlit Docs](https://docs.streamlit.io/)

### Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformers)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
- [Training language models to follow instructions](https://arxiv.org/abs/2203.02155) (InstructGPT)

### Datasets

- [The Stack](https://huggingface.co/datasets/bigcode/the-stack) - Code source
- [CodeParrot](https://huggingface.co/codeparrot) - Python code

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 👥 Auteurs

**Équipe IRA**  
Workshop: Build a Coding LLM from Scratch  
Date: Décembre 2025

---

## 🙏 Remerciements

- [HuggingFace](https://huggingface.co/) pour les datasets et tokenizers
- [BigCode](https://www.bigcode-project.org/) pour The Stack
- [PyTorch Team](https://pytorch.org/) pour le framework
- [Streamlit](https://streamlit.io/) pour le dashboard

---

<div align="center">

**⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile !**

Made with ❤️ by Équipe IRA

</div>
