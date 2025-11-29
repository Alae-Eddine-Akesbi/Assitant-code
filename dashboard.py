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
from transformers import GPT2Tokenizer
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

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
        background: linear-gradient(135deg, #f093fb22 0%, #f5576c22 100%);
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
    
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ARCHITECTURE DU MODÈLE (copie depuis les notebooks)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# ============================================================================
# CACHE DES MODÈLES
# ============================================================================
@st.cache_resource
def load_model(model_path, tokenizer_path, model_name):
    """Charge un modèle et son tokenizer avec mise en cache"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Charger le tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        
        # Créer le modèle
        model = MiniGPT(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            dropout=config['dropout']
        ).to(device)
        
        # Charger les poids
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Extraire les métriques
        metrics = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': checkpoint['history']['val_loss'][-1] if 'history' in checkpoint else 'N/A',
            'perplexity': np.exp(checkpoint['history']['val_loss'][-1]) if 'history' in checkpoint else 'N/A',
            'params': sum(p.numel() for p in model.parameters()),
        }
        
        return model, tokenizer, device, metrics
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de {model_name}: {str(e)}")
        return None, None, None, None

@st.cache_data
def generate_code(_model, _tokenizer, _device, prompt, max_tokens=150, temperature=0.7, top_k=50):
    """Génère du code avec un modèle"""
    try:
        input_ids = _tokenizer.encode(prompt, return_tensors='pt').to(_device)
        
        with torch.no_grad():
            start_time = time.time()
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
            generation_time = time.time() - start_time
        
        generated_text = _tokenizer.decode(output_ids[0].tolist())
        
        return generated_text, generation_time
    except Exception as e:
        return f"Erreur: {str(e)}", 0.0

# ============================================================================
# HEADER PRINCIPAL
# ============================================================================
st.markdown('<h1 class="main-header">🧠 LLM Coding Assistant Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

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
    st.markdown("### 📊 Modèles Disponibles")
    
    show_pretrained = st.checkbox("✅ Pre-Training", value=True)
    show_sft = st.checkbox("✅ Post-Training (SFT)", value=True)
    show_alignment = st.checkbox("⚠️ Alignment (RLHF)", value=False, 
                                  help="Non disponible pour l'instant")
    
    st.markdown("---")
    st.markdown("### 📁 Chemins des Modèles")
    
    pretrained_path = st.text_input(
        "Pre-Training",
        "models/pre_training/mini_gpt_code_FINAL.pt"
    )
    
    sft_path = st.text_input(
        "Post-Training",
        "models/post_training/mini_gpt_sft_FINAL.pt"
    )
    
    alignment_path = st.text_input(
        "Alignment",
        "models/alignment/mini_gpt_rlhf_FINAL.pt",
        disabled=True
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.info("Équipe IRA - Workshop LLM\n\nDate: 29 Nov 2025")

# ============================================================================
# SECTION PRINCIPALE - INPUT
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
    if st.button("🔢 Fibonacci"):
        prompt_input = "def fibonacci(n):"
    if st.button("🧮 Factorial"):
        prompt_input = "<instruction> Write a function to calculate factorial <reasoning>"
    if st.button("🔍 Binary Search"):
        prompt_input = "<instruction> Implement binary search algorithm <reasoning>"
    if st.button("🔄 QuickSort"):
        prompt_input = "def quicksort(arr):"

generate_button = st.button("🚀 Générer avec tous les modèles", type="primary", use_container_width=True)

# ============================================================================
# GÉNÉRATION ET COMPARAISON
# ============================================================================
if generate_button and prompt_input:
    
    st.markdown("---")
    st.markdown("## 🔬 Comparaison des Modèles")
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    total_models = sum([show_pretrained, show_sft, show_alignment])
    current_step = 0
    
    # ========================================================================
    # MODÈLE 1: PRE-TRAINING
    # ========================================================================
    if show_pretrained:
        status_text.text("⏳ Chargement du modèle Pre-Training...")
        progress_bar.progress(current_step / total_models)
        
        model_pre, tokenizer_pre, device_pre, metrics_pre = load_model(
            pretrained_path,
            "models/pre_training/tokenizer",
            "Pre-Training"
        )
        
        if model_pre is not None:
            status_text.text("🔄 Génération avec Pre-Training...")
            generated_pre, time_pre = generate_code(
                model_pre, tokenizer_pre, device_pre,
                prompt_input, max_tokens, temperature, top_k
            )
            
            results['pre'] = {
                'output': generated_pre,
                'time': time_pre,
                'metrics': metrics_pre
            }
        
        current_step += 1
        progress_bar.progress(current_step / total_models)
    
    # ========================================================================
    # MODÈLE 2: POST-TRAINING (SFT)
    # ========================================================================
    if show_sft:
        status_text.text("⏳ Chargement du modèle Post-Training...")
        progress_bar.progress(current_step / total_models)
        
        model_sft, tokenizer_sft, device_sft, metrics_sft = load_model(
            sft_path,
            "models/post_training/tokenizer",
            "Post-Training"
        )
        
        if model_sft is not None:
            status_text.text("🔄 Génération avec Post-Training...")
            generated_sft, time_sft = generate_code(
                model_sft, tokenizer_sft, device_sft,
                prompt_input, max_tokens, temperature, top_k
            )
            
            results['sft'] = {
                'output': generated_sft,
                'time': time_sft,
                'metrics': metrics_sft
            }
        
        current_step += 1
        progress_bar.progress(current_step / total_models)
    
    # ========================================================================
    # MODÈLE 3: ALIGNMENT (RLHF) - Placeholder
    # ========================================================================
    if show_alignment:
        results['alignment'] = {
            'output': "⚠️ Modèle RLHF non encore entraîné.\n\nProchaines étapes:\n- Collecte de préférences humaines\n- Entraînement du reward model\n- Optimisation PPO",
            'time': 0.0,
            'metrics': {'epoch': 'N/A', 'val_loss': 'N/A', 'perplexity': 'N/A', 'params': 'N/A'}
        }
    
    progress_bar.progress(1.0)
    status_text.text("✅ Génération terminée !")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # ========================================================================
    # AFFICHAGE DES RÉSULTATS
    # ========================================================================
    st.markdown("### 📊 Résultats")
    
    # Créer les colonnes selon les modèles actifs
    cols = st.columns(len(results))
    
    # Mapping des styles
    model_styles = {
        'pre': ('pre-training', '🔵 Pre-Training', '#667eea'),
        'sft': ('post-training', '🟣 Post-Training (SFT)', '#f093fb'),
        'alignment': ('alignment', '🔷 Alignment (RLHF)', '#4facfe')
    }
    
    for idx, (model_key, result) in enumerate(results.items()):
        with cols[idx]:
            style_class, title, color = model_styles[model_key]
            
            st.markdown(f'<div class="model-card {style_class}">', unsafe_allow_html=True)
            st.markdown(f"### {title}")
            
            # Métriques
            metrics = result['metrics']
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("⏱️ Temps", f"{result['time']:.2f}s")
                st.metric("📊 Epoch", str(metrics['epoch']))
            with col_m2:
                if isinstance(metrics['val_loss'], float):
                    st.metric("📉 Val Loss", f"{metrics['val_loss']:.4f}")
                else:
                    st.metric("📉 Val Loss", str(metrics['val_loss']))
                
                if isinstance(metrics['perplexity'], float):
                    st.metric("🎯 Perplexity", f"{metrics['perplexity']:.2f}")
                else:
                    st.metric("🎯 Perplexity", str(metrics['perplexity']))
            
            # Code généré
            st.markdown("**Sortie générée:**")
            st.markdown(f'<div class="code-output">{result["output"]}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # GRAPHIQUES COMPARATIFS
    # ========================================================================
    st.markdown("---")
    st.markdown("### 📈 Analyses Comparatives")
    
    tab1, tab2, tab3 = st.tabs(["⏱️ Performance", "📊 Métriques", "📏 Longueur"])
    
    with tab1:
        # Graphique des temps de génération
        fig_time = go.Figure()
        
        for model_key, result in results.items():
            _, title, color = model_styles[model_key]
            fig_time.add_trace(go.Bar(
                name=title,
                x=[title],
                y=[result['time']],
                marker_color=color,
                text=[f"{result['time']:.2f}s"],
                textposition='auto'
            ))
        
        fig_time.update_layout(
            title="⏱️ Temps de Génération par Modèle",
            yaxis_title="Temps (secondes)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    with tab2:
        # Comparaison des métriques
        metrics_data = []
        for model_key, result in results.items():
            _, title, _ = model_styles[model_key]
            m = result['metrics']
            if isinstance(m['val_loss'], float) and isinstance(m['perplexity'], float):
                metrics_data.append({
                    'Modèle': title,
                    'Validation Loss': m['val_loss'],
                    'Perplexity': m['perplexity']
                })
        
        if metrics_data:
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig_loss = px.bar(
                    metrics_data,
                    x='Modèle',
                    y='Validation Loss',
                    title='📉 Validation Loss',
                    color='Modèle'
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col_g2:
                fig_perp = px.bar(
                    metrics_data,
                    x='Modèle',
                    y='Perplexity',
                    title='🎯 Perplexity',
                    color='Modèle'
                )
                st.plotly_chart(fig_perp, use_container_width=True)
    
    with tab3:
        # Longueur des sorties
        length_data = []
        for model_key, result in results.items():
            _, title, _ = model_styles[model_key]
            length_data.append({
                'Modèle': title,
                'Nombre de caractères': len(result['output']),
                'Nombre de lignes': result['output'].count('\n') + 1
            })
        
        col_l1, col_l2 = st.columns(2)
        
        with col_l1:
            fig_chars = px.bar(
                length_data,
                x='Modèle',
                y='Nombre de caractères',
                title='📏 Longueur en Caractères',
                color='Modèle'
            )
            st.plotly_chart(fig_chars, use_container_width=True)
        
        with col_l2:
            fig_lines = px.bar(
                length_data,
                x='Modèle',
                y='Nombre de lignes',
                title='📄 Nombre de Lignes',
                color='Modèle'
            )
            st.plotly_chart(fig_lines, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🧠 <b>Workshop LLM Coding Assistant</b></p>
    <p>Équipe IRA - 2025</p>
    <p>Pre-Training → Post-Training (SFT) → Alignment (RLHF)</p>
</div>
""", unsafe_allow_html=True)
