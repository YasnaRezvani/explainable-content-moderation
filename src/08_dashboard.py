# 08_dashboard.py
# Interactive Content Moderation Dashboard
# Run with: streamlit run 08_dashboard.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import os
import requests
import gdown

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Content Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Drive file ID
GDRIVE_FILES = {
    'model': '1F-OEssWqo0l9PnN-vS_CdKrpxbEUcnoG',           # primary_v3_fold0.pth
    'explanations': '1-WYs_xhfEVhI1EOdiulGzKZt9S78-zF7'    # explanation_dataset_100.json
}

# Local cache directory
CACHE_DIR = Path("cache_artifacts")
CACHE_DIR.mkdir(exist_ok=True)

MODEL_PATH = CACHE_DIR / "primary_v3_fold0.pth"
EXPLANATION_PATH = CACHE_DIR / "explanation_dataset_100.json"

# Model config
MODEL_NAME = "roberta-base"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ============================================================
# GOOGLE DRIVE DOWNLOAD FUNCTIONS
# ============================================================

import gdown

def download_file_from_gdrive(file_id, destination):
    """Download a file from Google Drive using gdown"""
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        gdown.download(url, str(destination), quiet=False)
    except Exception as e:
        raise ValueError(
            f"Failed to download from Google Drive.\n"
            f"File ID: {file_id}\n"
            f"Error: {e}\n\n"
            f"Please verify:\n"
            f"1. File sharing is set to 'Anyone with the link'\n"
            f"2. File ID is correct\n"
            f"3. File exists and is accessible"
        )

def ensure_artifacts_downloaded():
    """Download artifacts from Google Drive if not present (NOT CACHED)"""
    
    files_to_download = []
    
    if not MODEL_PATH.exists():
        files_to_download.append(('model', MODEL_PATH))
    
    if not EXPLANATION_PATH.exists():
        files_to_download.append(('explanations', EXPLANATION_PATH))
    
    if files_to_download:
        # Use Streamlit's built-in progress indicators
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for idx, (file_key, destination) in enumerate(files_to_download):
            file_id = GDRIVE_FILES[file_key]
            
            progress_text.text(f"📥 Downloading {destination.name}... ({idx+1}/{len(files_to_download)})")
            
            try:
                download_file_from_gdrive(file_id, destination)
                
                # Verify file size
                file_size_mb = destination.stat().st_size / (1024 * 1024)
                progress_text.text(f"✅ Downloaded {destination.name} ({file_size_mb:.1f} MB)")
                
            except Exception as e:
                st.error(f"❌ Failed to download {destination.name}")
                st.error(str(e))
                st.stop()
            
            progress_bar.progress((idx + 1) / len(files_to_download))
        
        progress_text.text("✅ All artifacts downloaded!")
        import time
        time.sleep(1)
        progress_bar.empty()
        progress_text.empty()
    
    return True


# ============================================================
# HELPER: Format label names
# ============================================================

def format_label_name(label):
    """Format label name for display (remove underscores, title case)"""
    return label.replace('_', ' ').title()

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model (assumes files are already downloaded)"""
    
    # Check if model file exists
    if not MODEL_PATH.exists():
        st.error(f"❌ Model file not found at {MODEL_PATH}")
        st.error("Please ensure artifacts are downloaded first.")
        st.stop()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label_names)
    )
    
    # Load with weights_only=False for PyTorch 2.6 compatibility
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    )
    
    model.to(DEVICE)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    return model, tokenizer

@st.cache_data
def load_explanations():
    """Load pre-computed explanations (assumes files are already downloaded)"""
    
    # Check if explanation file exists
    if not EXPLANATION_PATH.exists():
        st.error(f"❌ Explanation file not found at {EXPLANATION_PATH}")
        st.error("Please ensure artifacts are downloaded first.")
        st.stop()
    
    with open(EXPLANATION_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# ============================================================
# MODEL WRAPPER FOR INTEGRATED GRADIENTS
# ============================================================

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, embeddings, attention_mask=None):
        outputs = self.model.roberta(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        logits = self.model.classifier(sequence_output)
        return torch.sigmoid(logits)

# ============================================================
# PREDICTION AND EXPLANATION FUNCTIONS
# ============================================================

def predict_text(text, model, tokenizer):
    """Get predictions for a text"""
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    
    return probs

def get_token_attributions(text, target_label_idx, model, tokenizer, n_steps=30):
    """Compute Integrated Gradients attributions"""
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    
    embeddings = model.roberta.embeddings(input_ids)
    baseline_embeddings = torch.zeros_like(embeddings)
    
    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)
    
    attributions, delta = ig.attribute(
        embeddings,
        baseline_embeddings,
        target=target_label_idx,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions.cpu().detach().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    mask = attention_mask[0].cpu().numpy()
    tokens = [t for t, m in zip(tokens, mask) if m == 1]
    attributions = attributions[:len(tokens)]
    
    return tokens, attributions

def visualize_attributions_plotly(tokens, attributions):
    """Create HTML visualization of token attributions"""
    abs_max = max(abs(attributions.min()), abs(attributions.max()))
    
    html = '<div style="line-height: 2.5; font-size: 16px;">'
    
    for token, attr in zip(tokens, attributions):
        display_token = token.replace('Ġ', ' ').replace('Ċ', '\\n')
        if display_token in ['<s>', '</s>', '<pad>']:
            continue
        
        # Normalize to [-1, 1]
        normalized_attr = attr / (abs_max + 1e-8)
        
        # Color: red (negative) -> white (zero) -> green (positive)
        if normalized_attr >= 0:
            # Green for positive
            intensity = int(255 * (1 - normalized_attr))
            color = f'rgb({intensity}, 255, {intensity})'
        else:
            # Red for negative
            intensity = int(255 * (1 + normalized_attr))
            color = f'rgb(255, {intensity}, {intensity})'
        
        html += f'<span style="background-color: {color}; padding: 3px 5px; margin: 2px; border-radius: 3px; border: 1px solid #ccc;">{display_token}</span>'
    
    html += '</div>'
    return html

# ============================================================
# HUMAN-READABLE EXPLANATION GENERATION
# ============================================================

def censor_profanity(word):
    """Censor profanity with asterisks, keeping first and last letter"""
    profanity_list = [
        'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell', 'crap', 'piss',
        'bastard', 'cunt', 'dick', 'pussy', 'cock', 'whore', 'slut', 'fag',
        'nigger', 'retard', 'idiot', 'moron', 'stupid', 'dumb'
    ]
    
    word_lower = word.lower().strip()
    
    for profane in profanity_list:
        if profane in word_lower:
            if len(word) <= 2:
                return '*' * len(word)
            return word[0] + '*' * (len(word) - 2) + word[-1]
    
    return word

def censor_text_display(text):
    """Censor profanity in displayed text for professional presentation"""
    profanity_list = [
        'fuck', 'fucking', 'fucked', 'fucker', 'shit', 'shitty', 'bitch', 'bitches',
        'ass', 'asshole', 'damn', 'damned', 'hell', 'crap', 'piss', 'pissed',
        'bastard', 'cunt', 'dick', 'pussy', 'cock', 'whore', 'slut', 'fag', 'faggot',
        'nigger', 'nigga', 'retard', 'retarded'
    ]
    
    censored = text
    for word in profanity_list:
        # Case-insensitive replacement
        import re
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        if len(word) <= 2:
            replacement = '*' * len(word)
        else:
            replacement = word[0] + '*' * (len(word) - 2) + word[-1]
        censored = pattern.sub(replacement, censored)
    
    return censored

def categorize_toxic_words(words_with_scores):
    """
    Categorize toxic words into semantic concepts.
    
    Args:
        words_with_scores: List of (word, score) tuples
    
    Returns:
        Dictionary of categories with examples
    """
    categories = {
        'personal_insults': [],
        'profanity': [],
        'threats': [],
        'slurs': [],
        'harassment': []
    }
    
    # Category keywords
    insult_keywords = ['stupid', 'idiot', 'moron', 'dumb', 'retard', 'loser', 'pathetic', 'worthless']
    profanity_keywords = ['fuck', 'shit', 'damn', 'hell', 'ass', 'bitch', 'cunt', 'dick', 'cock', 'pussy']
    threat_keywords = ['kill', 'die', 'death', 'hurt', 'bash', 'destroy', 'attack', 'shoot', 'stab']
    slur_keywords = ['nigger', 'fag', 'kike', 'spic', 'chink', 'wetback', 'towel']
    harassment_keywords = ['you', 'your', 'yourself']
    
    for word, score in words_with_scores:
        word_lower = word.lower().strip()
        
        # Skip function words and very short words
        if word_lower in ['a', 'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
            continue
        if len(word_lower) <= 1:
            continue
        
        # Categorize
        if any(keyword in word_lower for keyword in insult_keywords):
            categories['personal_insults'].append(word)
        elif any(keyword in word_lower for keyword in profanity_keywords):
            categories['profanity'].append(word)
        elif any(keyword in word_lower for keyword in threat_keywords):
            categories['threats'].append(word)
        elif any(keyword in word_lower for keyword in slur_keywords):
            categories['slurs'].append(word)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def generate_human_readable_explanation(text, tokens, attributions, label, probability):
    """
    Generate a 4-section human-readable explanation.
    
    Args:
        text: Original text
        tokens: Token list from IG
        attributions: Attribution scores
        label: Predicted label
        probability: Prediction probability
    
    Returns:
        Dictionary with 4 sections
    """
    # STEP 1: Reconstruct original words from subword tokens
    word_scores = []
    current_word = ""
    current_score = 0.0
    current_count = 0
    
    for token, attr in zip(tokens, attributions):
        # Skip special tokens
        if token in ['<s>', '</s>', '<pad>']:
            continue
        
        # Clean token
        clean_token = token.replace('Ġ', ' ').replace('Ċ', '\\n')
        
        # Check if this starts a new word (has leading space or is first token)
        if clean_token.startswith(' ') and current_word:
            # Save previous word
            if current_word.strip() and current_count > 0:
                avg_score = current_score / current_count
                word_scores.append((current_word.strip(), avg_score))
            # Start new word
            current_word = clean_token
            current_score = float(attr)
            current_count = 1
        else:
            # Continue current word
            current_word += clean_token
            current_score += float(attr)
            current_count += 1
    
    # Don't forget last word
    if current_word.strip() and current_count > 0:
        avg_score = current_score / current_count
        word_scores.append((current_word.strip(), avg_score))
    
    # STEP 2: Filter out function words and noise
    function_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'and', 'or', 'but', 'if', 'then', 'so', 'because',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'by',
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'this', 'that', 'these', 'those',
        'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'can', 'may', 'might',
        'such', 'really', 'very', 'too', 'also', 'just'
    }
    
    # Filter and sort
    content_words = [
        (word, score) for word, score in word_scores 
        if word.lower() not in function_words 
        and len(word) > 2
        and score > 0  # Only positive contributions
    ]
    
    # Sort by score
    content_words_sorted = sorted(content_words, key=lambda x: x[1], reverse=True)
    
    # Get top 3 content words
    top_words = content_words_sorted[:3]
    
    # Also get negative attributions for mitigating factors
    negative_words = [
        (word, score) for word, score in word_scores 
        if score < 0
    ]
    negative_words_sorted = sorted(negative_words, key=lambda x: x[1])
    
    # 1️⃣ DECISION SUMMARY
    decision = "FLAGGED" if probability >= 0.5 else "NOT FLAGGED"
    
    if decision == "FLAGGED":
        decision_summary = f"This comment was **{decision}** as **{format_label_name(label)}** with {probability:.0%} confidence."
    else:
        decision_summary = f"This comment was **{decision}** (highest score: {format_label_name(label)} at {probability:.0%})."
    
    # 2️⃣ PRIMARY REASONS (Concept-Level) - NO PARENTHETICAL EXAMPLES
    primary_reasons = []
    
    if top_words and probability >= 0.5:  # Only show reasons if flagged
        # Categorize words
        categories = categorize_toxic_words(top_words)
        
        if 'slurs' in categories:
            primary_reasons.append("Contains identity-based slurs or discriminatory language")
        
        if 'threats' in categories:
            primary_reasons.append("Uses threatening or violent language")
        
        if 'personal_insults' in categories:
            primary_reasons.append("Contains direct personal insults")
        
        if 'profanity' in categories:
            primary_reasons.append("Uses strong profane language")
        
        # If no categories matched, use generic
        if not primary_reasons:
            primary_reasons.append(f"Contains language commonly associated with {format_label_name(label).lower()} content")
    
    # Limit to 3 reasons max
    primary_reasons = primary_reasons[:3]
    
    # 3️⃣ CONTRIBUTING EVIDENCE - TOP 3 MAX, FULL WORDS ONLY
    key_terms = [censor_profanity(word) for word, _ in top_words]
    
    # 4️⃣ MITIGATING FACTORS (Only if relevant and borderline)
    mitigating_factors = None
    
    # Only show if truly borderline (0.45-0.65 range)
    if 0.45 <= probability <= 0.65:
        if negative_words_sorted and abs(negative_words_sorted[0][1]) > 0.1:
            mitigating_factors = "The model detected some neutral or respectful language, which reduced the overall score."
    
    return {
        'decision_summary': decision_summary,
        'primary_reasons': primary_reasons,
        'key_terms': key_terms,
        'mitigating_factors': mitigating_factors,
        'raw_tokens': tokens,
        'raw_attributions': attributions,
        'top_positive': top_words,
        'top_negative': negative_words_sorted[:5]
    }

def has_profanity(text):
    """Check if text contains profanity"""
    profanity_list = [
        'fuck', 'fucking', 'fucked', 'fucker', 'shit', 'shitty', 'bitch',
        'ass', 'asshole', 'damn', 'cunt', 'dick', 'pussy', 'cock', 
        'whore', 'slut', 'fag', 'faggot', 'nigger', 'nigga'
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in profanity_list)

# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    # Header
    st.title("🛡️ Explainable Content Moderation System")
    st.markdown("*An AI-powered system with transparent, interpretable decisions*")

    # ✅ STEP 1: Ensure artifacts are downloaded FIRST (before caching)
    ensure_artifacts_downloaded()
    
    # ✅ STEP 2: Load resources (now files exist and can be cached)
    with st.spinner("Loading model and data..."):
        model, tokenizer = load_model()
        explanations = load_explanations()
    
    # Sidebar
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio(
        "Select a feature:",
        ["🔍 Live Moderation", "📚 Example Gallery", "📊 Batch Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This system uses **RoBERTa** with **Integrated Gradients** "
        "to provide explainable content moderation decisions.\n\n"
        f"**Model Performance:**\n"
        f"- Test Macro F1: 0.677\n"
        f"- Trained on 127k examples\n"
        f"- 6 toxicity categories"
    )
    
    # ============================================================
    # TAB 1: LIVE MODERATION
    # ============================================================
    
    if tab == "🔍 Live Moderation":
        st.header("Live Content Moderation")
        st.markdown("Enter text below to get real-time moderation analysis with explanations.")
        
        # Initialize session state for persistent results
        if 'analysis_done' not in st.session_state:
            st.session_state.analysis_done = False
            st.session_state.user_text = ""
            st.session_state.results = None
        
        # Text input with session state
        user_text = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.user_text,
            height=150,
            placeholder="Type or paste content here...",
            key="text_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col3:
            st.caption("Analysis typically takes 2-3 seconds")
        
        # Handle clear button
        if clear_btn:
            st.session_state.analysis_done = False
            st.session_state.user_text = ""
            st.session_state.results = None
            st.rerun()
        
        # Handle analyze button
        if analyze_btn and user_text.strip():
            with st.spinner("Analyzing..."):
                # Get predictions
                probs = predict_text(user_text, model, tokenizer)
                max_idx = int(probs.argmax())
                max_prob = probs[max_idx]
                max_label = label_names[max_idx]
                
                # Get explanation
                tokens, attributions = get_token_attributions(
                    user_text, max_idx, model, tokenizer, n_steps=30
                )
                
                explanation = generate_human_readable_explanation(
                    user_text, tokens, attributions, max_label, max_prob
                )
                
                # Store results in session state
                st.session_state.analysis_done = True
                st.session_state.user_text = user_text
                st.session_state.results = {
                    'probs': probs,
                    'max_idx': max_idx,
                    'max_prob': max_prob,
                    'max_label': max_label,
                    'explanation': explanation
                }
        
        # Display results if they exist
        if st.session_state.analysis_done and st.session_state.results:
            results = st.session_state.results
            
            # Create anchor for auto-scroll
            st.markdown('<div id="results"></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Overall decision
            if results['max_prob'] >= 0.5:
                st.error(f"⚠️ **FLAGGED** as **{format_label_name(results['max_label'])}** ({results['max_prob']:.1%} confidence)")
            else:
                st.success(f"✅ **NOT FLAGGED** (highest score: {format_label_name(results['max_label'])} at {results['max_prob']:.1%})")
            
            st.markdown("---")
            
            # Explanation
            st.subheader("💡 Explanation")
            
            explanation = results['explanation']
            
            # Decision Summary
            if results['max_prob'] >= 0.5:
                st.error(f"⚠️ {explanation['decision_summary']}")
            else:
                st.success(f"✅ {explanation['decision_summary']}")
            
            # Primary Reasons
            if explanation['primary_reasons']:
                st.markdown("**Why this was flagged:**")
                for reason in explanation['primary_reasons']:
                    st.markdown(f"• {reason}")
            
            # Key Terms
            if explanation['key_terms'] and results['max_prob'] >= 0.5:
                st.markdown("**Key terms:**")
                st.markdown(" · ".join(explanation['key_terms']))
            
            # Mitigating Factors
            if explanation['mitigating_factors']:
                st.markdown("**Context considered:**")
                st.info(explanation['mitigating_factors'])
            
            # Technical details
            with st.expander("🔧 View Technical Details"):
                st.markdown("#### All Category Scores")
                cols = st.columns(3)
                for i, (label, prob) in enumerate(zip(label_names, results['probs'])):
                    with cols[i % 3]:
                        st.metric(
                            format_label_name(label),
                            f"{prob:.1%}",
                            delta="FLAGGED" if prob >= 0.5 else None,
                            delta_color="inverse"
                        )
                
                st.markdown("---")
                
                st.caption("Token-level attribution visualization (green = increases score, red = decreases score)")
                html_viz = visualize_attributions_plotly(
                    explanation['raw_tokens'], 
                    explanation['raw_attributions']
                )
                st.markdown(html_viz, unsafe_allow_html=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top Contributing Tokens:**")
                    for word, score in explanation['top_positive'][:5]:
                        st.caption(f"'{word}': +{score:.4f}")
                
                with col2:
                    if explanation['top_negative']:
                        st.markdown("**Top Suppressing Tokens:**")
                        for word, score in explanation['top_negative']:
                            st.caption(f"'{word}': {score:.4f}")
            
            # Auto-scroll JavaScript
            st.markdown(
                """
                <script>
                    window.location.hash = '#results';
                </script>
                """,
                unsafe_allow_html=True
            )

    # ============================================================
    # TAB 2: EXAMPLE GALLERY
    # ============================================================
    
    elif tab == "📚 Example Gallery":
        st.header("Example Gallery")
        st.markdown(f"Browse through {len(explanations)} pre-computed explanations from the test set.")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_label = st.selectbox(
                "Filter by label:",
                ["All"] + [format_label_name(l) for l in label_names]
            )
        
        with col2:
            filter_decision = st.selectbox(
                "Filter by decision:",
                ["All", "FLAGGED", "NOT FLAGGED"]
            )
        
        with col3:
            filter_confidence = st.selectbox(
                "Filter by confidence:",
                ["All", "High (≥0.8)", "Medium (0.5-0.8)", "Low (<0.5)"]
            )
        
        # Apply filters
        filtered_exps = explanations.copy()
        
        # Convert filter label back to internal format
        if filter_label != "All":
            internal_label = filter_label.lower().replace(' ', '_')
            filtered_exps = [e for e in filtered_exps if e['label'] == internal_label]
        
        if filter_decision != "All":
            filtered_exps = [e for e in filtered_exps if e['decision'] == filter_decision]
        
        if filter_confidence == "High (≥0.8)":
            filtered_exps = [e for e in filtered_exps if e['probability'] >= 0.8]
        elif filter_confidence == "Medium (0.5-0.8)":
            filtered_exps = [e for e in filtered_exps if 0.5 <= e['probability'] < 0.8]
        elif filter_confidence == "Low (<0.5)":
            filtered_exps = [e for e in filtered_exps if e['probability'] < 0.5]
        
        st.markdown(f"**Showing {len(filtered_exps)} examples**")
        
        # Pagination
        examples_per_page = 10
        total_pages = (len(filtered_exps) - 1) // examples_per_page + 1
        
        page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
        
        start_idx = (page - 1) * examples_per_page
        end_idx = min(start_idx + examples_per_page, len(filtered_exps))
        
        # Display examples
        for i, exp in enumerate(filtered_exps[start_idx:end_idx], start=start_idx+1):
            decision_text = "FLAGGED" if exp['decision'] == 'FLAGGED' else "NOT FLAGGED"
            prob = exp['probability']
            
            # Censor original text for display
            censored_text = censor_text_display(exp['text'])
            
            with st.expander(f"#{i} — {decision_text} as {format_label_name(exp['label'])} ({prob:.0%})"):
                st.markdown(f"**Original Text:**")
                st.text(censored_text)
                
                st.markdown("---")
                
                # Decision summary
                if prob >= 0.5:
                    st.error(f"**{decision_text}** as **{format_label_name(exp['label'])}** with {prob:.0%} confidence.")
                else:
                    st.success(f"**{decision_text}** (highest score: {format_label_name(exp['label'])} at {prob:.0%}).")
                
                # Primary reasons
                if prob >= 0.5 and exp['top_contributing_words']:
                    st.markdown("**Why this was flagged:**")
                    
                    words_only = [(w, s) for w, s in exp['top_contributing_words'][:5]]
                    categories = categorize_toxic_words(words_only)
                    
                    displayed_reasons = []
                    
                    if 'slurs' in categories:
                        displayed_reasons.append("• Contains identity-based slurs or discriminatory language")
                    
                    if 'threats' in categories:
                        displayed_reasons.append("• Uses threatening or violent language")
                    
                    if 'personal_insults' in categories:
                        displayed_reasons.append("• Contains direct personal insults")
                    
                    if 'profanity' in categories:
                        displayed_reasons.append("• Uses strong profane language")
                    if not displayed_reasons:
                        displayed_reasons.append(f"• Contains language commonly associated with {format_label_name(exp['label']).lower()} content")
                    
                    for reason in displayed_reasons[:3]:
                        st.markdown(reason)
                    
                    # Key terms
                    st.markdown("**Key terms:**")
                    clean_terms = []
                    for w, _ in exp['top_contributing_words'][:5]:
                        clean = w.strip()
                        if len(clean) > 2 and clean.lower() not in {'you', 'are', 'is', 'the', 'a', 'and', 'or'}:
                            clean_terms.append(censor_profanity(clean))
                        if len(clean_terms) >= 3:
                            break
                    
                    if clean_terms:
                        st.markdown(" · ".join(clean_terms))
                
                # Technical details
                with st.expander("🔧 View Technical Details"):
                    st.markdown("**All Category Scores:**")
                    if 'all_predictions' in exp:
                        cols = st.columns(3)
                        for idx, (label, score) in enumerate(exp['all_predictions'].items()):
                            with cols[idx % 3]:
                                st.metric(format_label_name(label), f"{score:.1%}")
                    
                    st.markdown("---")
                    
                    st.markdown("**Top Contributing Words (with scores):**")
                    for word, score in exp['top_contributing_words'][:5]:
                        st.caption(f"'{word}': +{score:.4f}")
        
    # ============================================================
    # TAB 3: BATCH ANALYSIS
    # ============================================================

    elif tab == "📊 Batch Analysis":
        st.header("Batch Content Analysis")
        st.markdown("Upload a CSV file with a 'text' column to analyze multiple pieces of content.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows")
            st.dataframe(df.head())
            
            if 'text' not in df.columns:
                st.error("❌ CSV must contain a 'text' column")
            else:
                if st.button("🚀 Analyze All", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for i, text in enumerate(df['text']):
                        status_text.text(f"Analyzing {i+1}/{len(df)}...")
                        progress_bar.progress((i + 1) / len(df))
                        
                        probs = predict_text(str(text), model, tokenizer)
                        max_idx = int(probs.argmax())
                        
                        results.append({
                            'text': text,
                            'predicted_label': format_label_name(label_names[max_idx]),
                            'confidence': float(probs[max_idx]),
                            'decision': 'FLAGGED' if probs[max_idx] >= 0.5 else 'NOT FLAGGED',
                            **{f'{format_label_name(label)}_score': float(prob) for label, prob in zip(label_names, probs)}
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    st.success("✅ Analysis complete!")
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Analyzed", len(results_df))
                    with col2:
                        flagged = (results_df['decision'] == 'FLAGGED').sum()
                        st.metric("Flagged", flagged)
                    with col3:
                        st.metric("Clean", len(results_df) - flagged)
                    
                    # Display results
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "moderation_results.csv",
                        "text/csv",
                        key='download-csv'
                    )

if __name__ == "__main__":
    main()              