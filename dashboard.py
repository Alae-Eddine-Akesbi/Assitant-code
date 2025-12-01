"""
🎯 Dashboard Streamlit - Comparaison des Modèles LLM
Comparez les sorties des modèles Pre-Training, Post-Training et Alignement

Équipe IRA - Workshop LLM Coding
"""

import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import AutoTokenizer
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import math
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================
st.set_page_config(
    page_title="🧠 LLM Coding Assistant - Comparaison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLES CSS PERSONNALISÉS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
        animation: fadeIn 1s ease-in;
    }
    
    .model-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    
    .pre-training {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border-left: 4px solid #667eea;
    }
    
    .post-training {
        background: linear-gradient(135deg, #f093fb22 0% , #f5576c22 100%);
        border-left: 4px solid #f093fb;
    }
    
    .alignment {
        background: linear-gradient(135deg, #4facfe22 0%, #00f2fe22 100%);
        border-left: 4px solid #4facfe;
    }
    
    .metric-box {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    
    .code-output {
        background: #1e1e1e;
        color: #d4d4d4;
        border-radius: 8px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL ARCHITECTURE - Exact copy from notebooks
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    block_size: int = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.block_size)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============================================================================
# CUSTOM DECODER - Avoid TensorFlow/Tokenizer decode issues
# ============================================================================

def safe_decode_tokens(token_ids, tokenizer):
    """
    Decode token IDs to text without using tokenizer.decode() which triggers TensorFlow.
    Falls back to manual vocabulary lookup.
    """
    try:
        # Get the vocab dictionary directly
        vocab = tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        
        # Convert IDs to tokens using vocabulary
        tokens = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in id_to_token:
                token = id_to_token[token_id]
                tokens.append(token)
            else:
                # Fallback for unknown tokens
                tokens.append(f"<unk:{token_id}>")
        
        # Join tokens with special handling for BPE tokens
        text = ""
        for token in tokens:
            if token.startswith("Ġ"):  # GPT-2 BPE space token
                text += " " + token[1:]
            elif token.startswith("##"):  # WordPiece continuation
                text += token[2:]
            else:
                text += token
        
        return text.strip()
    except Exception as e:
        # Last resort: return token IDs as string
        return f"[Tokens: {token_ids[:50]}...]"

# ============================================================================
# MODEL LOADING FUNCTION - Updated for notebook format
# ============================================================================

@st.cache_resource
def load_model(model_path, tokenizer_path, model_name, device):
    """
    Charge un modèle et son tokenizer avec mise en cache
    Compatible avec les checkpoints des notebooks
    Gère à la fois Pre-Training et Post-Training (SFT)
    """
    try:
        # ====================================================================
        # 1. CHARGER LE TOKENIZER
        # ====================================================================
        st.write(f"📖 Chargement tokenizer depuis: {tokenizer_path}")
        
        # Vérifier si le chemin existe
        from pathlib import Path
        tok_path = Path(tokenizer_path)
        if not tok_path.exists():
            st.error(f"❌ Tokenizer introuvable: {tokenizer_path}")
            return None, None, None, None
        
        # Charger le tokenizer - supporte BPE et HuggingFace formats
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            st.warning(f"⚠️  AutoTokenizer échoué: {str(e)[:80]}...")
            st.info(f"💡 Essai avec EleutherAI/gpt-neox-20b comme source...")
            try:
                # The notebook loads from EleutherAI directly - use that as fallback
                tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
                st.info(f"✅ Tokenizer EleutherAI chargé avec succès")
            except:
                st.warning(f"⚠️  EleutherAI échoué, essai du pre-training tokenizer...")
                try:
                    # Use pre-training tokenizer as final fallback for SFT
                    default_tokenizer_path = "models/pre_training/tokenizer"
                    tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_path)
                    st.info(f"✅ Tokenizer pre-training utilisé comme fallback")
                except:
                    st.error(f"❌ Impossible de charger le tokenizer")
                    return None, None, None, None
        
        if tokenizer is None:
            st.error(f"❌ Tokenizer est None")
            return None, None, None, None
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        vocab_size = len(tokenizer)
        st.success(f"✅ Tokenizer chargé ({vocab_size:,} tokens)")
        
        # ====================================================================
        # 2. CHARGER LE CHECKPOINT
        # ====================================================================
        st.write(f"📦 Chargement checkpoint: {model_path}")
        
        # Vérifier si le checkpoint existe
        model_file = Path(model_path)
        if not model_file.exists():
            st.error(f"❌ Checkpoint introuvable: {model_path}")
            return None, None, None, None
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Afficher la structure du checkpoint pour debug
        checkpoint_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['<direct_tensor>']
        st.write(f"🔍 Clés du checkpoint: {checkpoint_keys[:10]}...")
        
        # Determine vocabulary size from checkpoint if available
        checkpoint_vocab_size = None
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            checkpoint_vocab_size = checkpoint['config'].get('vocab_size')
        
        # Use checkpoint vocab size if different from tokenizer
        # (this handles the SFT case where model was trained with special tokens)
        if checkpoint_vocab_size and checkpoint_vocab_size != vocab_size:
            original_vocab_size = vocab_size
            vocab_size = checkpoint_vocab_size
            st.info(f"💡 Vocab size ajusté: {original_vocab_size} -> {vocab_size} (depuis checkpoint)")
        
        # ====================================================================
        # 3. EXTRAIRE LA CONFIGURATION
        # ====================================================================
        # Le checkpoint peut être:
        # 1. Un dict avec 'model_state_dict' et 'config' (Post-Training)
        # 2. Un dict avec 'model_state_dict' directement (Pre-Training final) 
        # 3. Un dict de poids (weights only - Pre-Training raw)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            # Format 1: Checkpoint structuré avec config
            config_dict = checkpoint['config']
            model_state = checkpoint.get('model_state_dict', checkpoint)
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Format 2: Checkpoint avec model_state_dict mais sans config
            config_dict = checkpoint.get('config', {})
            model_state = checkpoint['model_state_dict']
        else:
            # Format 3: Weights only ou state_dict direct
            config_dict = {}
            model_state = checkpoint
        
        # Map config keys intelligemment (Pre-Training vs Post-Training)
        # Pre-Training: n_embd, n_head, n_layer, dropout
        # Post-Training: d_model, n_head, n_layer, d_ff
        d_model = config_dict.get('d_model') or config_dict.get('n_embd', 512)
        n_heads = config_dict.get('n_head', config_dict.get('n_heads', 8))
        n_layers = config_dict.get('n_layer', config_dict.get('n_layers', 8))
        d_ff = config_dict.get('d_ff', 2048)
        block_size = config_dict.get('block_size', 256)
        
        config = ModelConfig(
            vocab_size=vocab_size,  # Toujours utiliser vocab_size du tokenizer !
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            block_size=block_size,
        )
        
        st.write(f"📊 Config du modèle:")
        st.write(f"   • d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
        st.write(f"   • block_size={config.block_size}, d_ff={config.d_ff}")
        
        # ====================================================================
        # 4. CRÉER ET CHARGER LE MODÈLE
        # ====================================================================
        model = TinyDecoderLM(config).to(device)
        
        # Charger les poids depuis le checkpoint
        try:
            model.load_state_dict(model_state)
        except RuntimeError as e:
            if 'size mismatch' in str(e) and 'tok_emb' in str(e):
                # Vocab size mismatch - adjust embeddings (like notebook does)
                st.info(f"💡 Ajustement du vocabulaire détecté...")
                old_tok_emb = model_state.get('tok_emb.weight')
                if old_tok_emb is not None:
                    old_vocab_size = old_tok_emb.shape[0]
                    new_vocab_size = config.vocab_size
                    st.write(f"   • Ancien vocab: {old_vocab_size}, Nouveau vocab: {new_vocab_size}")
                    
                    # Expand embeddings with new tokens
                    new_tok_emb = model.tok_emb.weight.data.clone()
                    new_tok_emb[:old_vocab_size] = old_tok_emb
                    model.tok_emb.weight.data = new_tok_emb
                    
                    # Update head weights (weight tying)
                    model.head.weight.data = new_tok_emb
                    
                    # Update state dict for loading
                    model_state['tok_emb.weight'] = new_tok_emb
                    model_state['head.weight'] = new_tok_emb
                    
                    model.load_state_dict(model_state, strict=False)
                    st.success(f"✅ Vocabulaire ajusté")
                else:
                    raise e
            else:
                raise e
        
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        st.success(f"✅ Modèle chargé ({total_params:,} paramètres)")
        
        # ====================================================================
        # 5. EXTRAIRE LES MÉTRIQUES
        # ====================================================================
        history = checkpoint.get('history', {})
        
        # Extraire val_loss et train_loss de manière robuste
        val_loss = 'N/A'
        if history and isinstance(history.get('val_loss'), list) and len(history['val_loss']) > 0:
            val_loss = float(history['val_loss'][-1])
        elif 'best_val_loss' in checkpoint:
            val_loss = float(checkpoint['best_val_loss'])
        
        train_loss = 'N/A'
        if history and isinstance(history.get('train_loss'), list) and len(history['train_loss']) > 0:
            train_loss = float(history['train_loss'][-1])
        
        metrics = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'params': total_params,
            'training_stage': checkpoint.get('training_stage', 'unknown'),
            'selected_from_epoch': checkpoint.get('selected_from_epoch', 'N/A'),
            'best_val_loss': checkpoint.get('best_val_loss', 'N/A'),
        }
        
        # Calculer perplexity
        if isinstance(metrics['val_loss'], float):
            metrics['perplexity'] = np.exp(metrics['val_loss'])
        else:
            metrics['perplexity'] = 'N/A'
        
        st.success(f"✅ Métriques: val_loss={metrics['val_loss']}, epoch={metrics['epoch']}")
        
        return model, tokenizer, device, metrics
        
    except FileNotFoundError as e:
        st.error(f"❌ Fichier non trouvé: {str(e)}")
        st.error(f"📂 Vérifiez les chemins: {model_path}, {tokenizer_path}")
        return None, None, None, None
    except RuntimeError as e:
        st.error(f"❌ Erreur à la charge du modèle: {str(e)}")
        st.info(f"💡 Conseil: Assurez-vous que la config correspond aux poids sauvegardés")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de {model_name}: {str(e)}")
        st.error(f"📂 Vérifiez le chemin: {model_path}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None


@st.cache_data
def generate_code(_model, _tokenizer, _device, prompt, max_tokens=150, temperature=0.7, top_k=50):
    """
    Génère du code avec un modèle
    Accepte les tokens comme premier argument pour bypass du cache
    """
    try:
        # Encoder le prompt
        encoded = _tokenizer.encode(prompt)
        if not encoded:
            return "❌ Erreur: Le prompt n'a pas pu être encodé", 0.0
        
        input_ids = torch.tensor([encoded], device=_device)
        
        # Vérifier que le tensor n'est pas vide
        if input_ids.numel() == 0:
            return "❌ Erreur: Prompt vide après encodage", 0.0
        
        with torch.no_grad():
            start_time = time.time()
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
            generation_time = time.time() - start_time
        
        
        # Decode using custom function that avoids TensorFlow
        generated_text = safe_decode_tokens(output_ids[0].tolist(), _tokenizer)
        
        return generated_text, generation_time
    except Exception as e:
        import traceback
        error_msg = f"❌ Erreur: {str(e)[:100]}"
        return error_msg, 0.0


# ============================================================================
# MAIN UI
# ============================================================================

st.markdown('<h1 class="main-header">🧠 LLM Coding Assistant Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Détection device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.info(f"🚀 Device: {device.upper()}")

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 🎛️ Paramètres de Génération")
    max_tokens = st.slider("Max Tokens", 50, 300, 150, 10)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    top_k = st.slider("Top-K", 10, 100, 50, 5)
    
    st.markdown("---")
    st.markdown("### 📊 Modèles à Charger")
    
    show_pretrained = st.checkbox("✅ Pre-Training", value=True)
    show_sft = st.checkbox("✅ Post-Training (SFT)", value=True)
    
    st.markdown("---")
    st.markdown("### 📁 Chemins des Modèles")
    
    # Chemins relatifs par défaut (depuis le répertoire courant)
    pretrained_path = st.text_input(
        "Pre-Training Model",
        "models/pre_training/model_final.pt",
        help="Chemin relatif ou absolu au checkpoint pre-training"
    )
    
    pretrained_tokenizer = st.text_input(
        "Pre-Training Tokenizer",
        "models/pre_training/tokenizer",
        help="Dossier contenant le tokenizer pre-training"
    )
    
    sft_path = st.text_input(
        "Post-Training Model",
        "models/post_training/model_sft_FINAL.pt",
        help="Chemin relatif ou absolu au checkpoint SFT final"
    )
    
    sft_tokenizer = st.text_input(
        "Post-Training Tokenizer",
        "models/post_training/tokenizer",
        help="Dossier contenant le tokenizer post-training"
    )

# ============================================================================
# MAIN CONTENT - INPUT
# ============================================================================
st.markdown("## 💬 Entrez votre Prompt")

col1, col2 = st.columns([3, 1])

with col1:
    prompt_input = st.text_area(
        "Prompt",
        placeholder="Ex: def fibonacci(n):\n\nOu pour SFT: <instruction> Write a function to calculate factorial <reasoning>",
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 📝 Exemples")
    
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("Fibonacci"):
            st.session_state.prompt = "def fibonacci(n):"
    with col_ex2:
        if st.button("Factorial (SFT)"):
            st.session_state.prompt = "<instruction> Write a function to calculate factorial <reasoning>"
    
    col_ex3, col_ex4 = st.columns(2)
    with col_ex3:
        if st.button("Binary Search"):
            st.session_state.prompt = "<instruction> Implement binary search algorithm <reasoning>"
    with col_ex4:
        if st.button("Reverse List"):
            st.session_state.prompt = "<instruction> Create a function to reverse a list <reasoning>"
    
    if 'prompt' in st.session_state:
        prompt_input = st.session_state.prompt

generate_button = st.button("🚀 Générer", type="primary", use_container_width=True)

# ============================================================================
# GÉNÉRATION ET COMPARAISON
# ============================================================================
if generate_button and prompt_input:
    
    st.markdown("---")
    st.markdown("## 🔬 Comparaison des Modèles")
    
    results = {}
    
    # Container for status
    status_container = st.container()
    progress_container = st.container()
    
    # ========================================================================
    # MODÈLE 1: PRE-TRAINING
    # ========================================================================
    if show_pretrained:
        with status_container:
            st.info("⏳ Chargement Pre-Training...")
        
        model_pre, tokenizer_pre, device_pre, metrics_pre = load_model(
            pretrained_path,
            pretrained_tokenizer,
            "Pre-Training",
            device
        )
        
        if model_pre is not None:
            with status_container:
                st.info("🔄 Génération avec Pre-Training...")
            
            generated_pre, time_pre = generate_code(
                model_pre, tokenizer_pre, device_pre,
                prompt_input, max_tokens, temperature, top_k
            )
            
            results['pre'] = {
                'output': generated_pre,
                'time': time_pre,
                'metrics': metrics_pre
            }
    
    # ========================================================================
    # MODÈLE 2: POST-TRAINING (SFT)
    # ========================================================================
    if show_sft:
        with status_container:
            st.info("⏳ Chargement Post-Training (SFT)...")
        
        model_sft, tokenizer_sft, device_sft, metrics_sft = load_model(
            sft_path,
            sft_tokenizer,
            "Post-Training (SFT)",
            device
        )
        
        if model_sft is not None:
            with status_container:
                st.info("🔄 Génération avec Post-Training...")
            
            generated_sft, time_sft = generate_code(
                model_sft, tokenizer_sft, device_sft,
                prompt_input, max_tokens, temperature, top_k
            )
            
            results['sft'] = {
                'output': generated_sft,
                'time': time_sft,
                'metrics': metrics_sft
            }
    
    # Clear status
    status_container.empty()
    progress_container.empty()
    
    if results:
        st.success("✅ Génération terminée !")
        
        # ====================================================================
        # AFFICHAGE DES RÉSULTATS
        # ====================================================================
        st.markdown("### 📊 Résultats")
        
        model_styles = {
            'pre': ('pre-training', '🔵 Pre-Training', '#667eea'),
            'sft': ('post-training', '🟣 Post-Training (SFT)', '#f093fb'),
        }
        
        cols = st.columns(len(results))
        
        for idx, (model_key, result) in enumerate(results.items()):
            with cols[idx]:
                style_class, title, color = model_styles[model_key]
                
                st.markdown(f'<div class="model-card {style_class}">', unsafe_allow_html=True)
                st.markdown(f"### {title}")
                
                # Métriques
                metrics = result['metrics']
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("⏱️ Temps", f"{result['time']:.3f}s")
                    st.metric("📅 Epoch", str(metrics['epoch']))
                
                with col_m2:
                    val_loss = metrics.get('val_loss')
                    if isinstance(val_loss, float):
                        st.metric("📉 Val Loss", f"{val_loss:.4f}")
                    else:
                        st.metric("📉 Val Loss", "N/A")
                    
                    perp = metrics.get('perplexity')
                    if isinstance(perp, float):
                        st.metric("🎯 Perplexity", f"{perp:.2f}")
                    else:
                        st.metric("🎯 Perplexity", "N/A")
                
                # Stage info
                st.caption(f"Stage: {metrics.get('training_stage', 'N/A')}")
                
                # Code généré
                st.markdown("**📝 Sortie générée:**")
                st.markdown(f'<div class="code-output">{result["output"]}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # ====================================================================
        # GRAPHIQUES COMPARATIFS
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📈 Analyses Comparatives")
        
        tab1, tab2, tab3 = st.tabs(["⏱️ Performance", "📊 Métriques", "📏 Longueur"])
        
        with tab1:
            fig_time = go.Figure()
            for model_key, result in results.items():
                _, title, color = model_styles[model_key]
                fig_time.add_trace(go.Bar(
                    name=title,
                    x=[title],
                    y=[result['time']],
                    marker_color=color,
                    text=[f"{result['time']:.3f}s"],
                    textposition='auto'
                ))
            
            fig_time.update_layout(
                title="⏱️ Temps de Génération",
                yaxis_title="Temps (secondes)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with tab2:
            metrics_data = []
            for model_key, result in results.items():
                _, title, _ = model_styles[model_key]
                m = result['metrics']
                val_loss = m.get('val_loss')
                perp = m.get('perplexity')
                if isinstance(val_loss, float) and isinstance(perp, float):
                    metrics_data.append({
                        'Modèle': title,
                        'Validation Loss': val_loss,
                        'Perplexity': perp
                    })
            
            if metrics_data:
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    fig_loss = px.bar(
                        metrics_data,
                        x='Modèle',
                        y='Validation Loss',
                        title='📉 Validation Loss'
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                
                with col_g2:
                    fig_perp = px.bar(
                        metrics_data,
                        x='Modèle',
                        y='Perplexity',
                        title='🎯 Perplexity'
                    )
                    st.plotly_chart(fig_perp, use_container_width=True)
        
        with tab3:
            length_data = []
            for model_key, result in results.items():
                _, title, _ = model_styles[model_key]
                length_data.append({
                    'Modèle': title,
                    'Caractères': len(result['output']),
                    'Lignes': result['output'].count('\n') + 1
                })
            
            col_l1, col_l2 = st.columns(2)
            
            with col_l1:
                fig_chars = px.bar(
                    length_data,
                    x='Modèle',
                    y='Caractères',
                    title='📏 Longueur en Caractères'
                )
                st.plotly_chart(fig_chars, use_container_width=True)
            
            with col_l2:
                fig_lines = px.bar(
                    length_data,
                    x='Modèle',
                    y='Lignes',
                    title='📄 Nombre de Lignes'
                )
                st.plotly_chart(fig_lines, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🧠 <b>Workshop LLM Coding Assistant</b></p>
    <p>Équipe IRA - Nov 2025</p>
    <p>📊 Pre-Training → 🎯 Post-Training (SFT) → 🎯 Alignment (RLHF)</p>
</div>
""", unsafe_allow_html=True)