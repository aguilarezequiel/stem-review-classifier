# ==============================================================================
# utils.py - Utility functions for the Streamlit app
# ==============================================================================
# Text preprocessing, model loading, prediction, and Steam API integration.
# ==============================================================================

import re
import torch
import requests
from transformers import BertForSequenceClassification, BertTokenizer


def clean_text(text):
    """
    Cleans input text applying the same transformations
    used during training.
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[░▒▓█▄▀■□▪▫●○◆◇♠♣♥♦♪♫☆★►◄▲▼←→↑↓]+', '', text)
    text = re.sub(r'\*[^*]+\*', '', text)
    text = re.sub(r'[•►▪▸‣⁃]', '', text)
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1 out of \2', text)
    text = re.sub(r'\d{3,4}\s*x\s*\d{3,4}', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)
    text = re.sub(r"[^a-zA-Z\s.,!?'-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text


def load_model(model_path="model_files"):
    """
    Loads the fine-tuned model and tokenizer.
    Returns model, tokenizer, and device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    """
    Predicts the sentiment of a single review.
    Returns label, confidence, and cleaned text.
    """
    cleaned = clean_text(text)

    if len(cleaned.split()) < 3:
        return "Too short to classify", 0.0, cleaned

    encoded = tokenizer(
        cleaned,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        result = model(input_ids, attention_mask=attention_mask, return_dict=True)

    logits = result.logits
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label = "Positive" if pred == 1 else "Negative"
    return label, confidence, cleaned


def fetch_steam_reviews(app_id, num_reviews=50):
    """
    Fetches English reviews from the Steam API for the given app ID.
    Uses the public endpoint:
      https://store.steampowered.com/appreviews/[APP_ID]?json=1

    Returns a dict with:
      - 'success': bool
      - 'app_id': str
      - 'total_positive': int
      - 'total_negative': int
      - 'total_reviews': int
      - 'review_score_desc': str (e.g. "Very Positive")
      - 'reviews': list of dicts with keys: text, voted_up, playtime, author_id
      - 'error': str (if failed)
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "filter": "all",
        "language": "english",
        "num_per_page": min(num_reviews, 100),
        "purchase_type": "all",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    except ValueError:
        return {"success": False, "error": "Invalid JSON response from Steam API"}

    if data.get("success") != 1:
        return {"success": False, "error": "Steam API returned an error. Check the App ID."}

    summary = data.get("query_summary", {})
    raw_reviews = data.get("reviews", [])

    # Parse reviews, filtering out extremely long ones (>2000 chars)
    # and extremely short ones (<20 chars)
    reviews = []
    for r in raw_reviews:
        text = r.get("review", "")
        if len(text) < 20 or len(text) > 2000:
            continue
        reviews.append({
            "text": text,
            "voted_up": r.get("voted_up", True),
            "playtime_hours": round(r.get("author", {}).get("playtime_forever", 0) / 60, 1),
            "votes_up": r.get("votes_up", 0),
        })

    return {
        "success": True,
        "app_id": app_id,
        "total_positive": summary.get("total_positive", 0),
        "total_negative": summary.get("total_negative", 0),
        "total_reviews": summary.get("total_reviews", 0),
        "review_score_desc": summary.get("review_score_desc", "Unknown"),
        "reviews": reviews,
    }


def analyze_reviews(reviews, model, tokenizer, device):
    """
    Runs the BERT model on a list of review dicts.
    Returns an enriched list with model predictions and an overall summary.
    """
    results = []
    positive_count = 0
    negative_count = 0

    for r in reviews:
        label, confidence, cleaned = predict_sentiment(
            r["text"], model, tokenizer, device
        )
        is_positive = label == "Positive"
        if is_positive:
            positive_count += 1
        else:
            negative_count += 1

        results.append({
            "text": r["text"],
            "cleaned_text": cleaned,
            "steam_label": "Recommended" if r["voted_up"] else "Not Recommended",
            "model_label": label,
            "confidence": confidence,
            "playtime_hours": r["playtime_hours"],
            "votes_up": r["votes_up"],
        })

    total = positive_count + negative_count
    pos_ratio = (positive_count / total * 100) if total > 0 else 0
    neg_ratio = (negative_count / total * 100) if total > 0 else 0

    summary = {
        "total_analyzed": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_ratio": pos_ratio,
        "negative_ratio": neg_ratio,
    }

    return results, summary
