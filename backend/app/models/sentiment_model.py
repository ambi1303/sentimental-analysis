from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

# Load once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_sentiment(text: str) -> dict:
    """
    Returns {"label": str, "score": float} for the top emotion.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    top_idx = torch.argmax(probs).item()
    return {
        "label": model.config.id2label[top_idx],
        "score": float(probs[top_idx])
    }
