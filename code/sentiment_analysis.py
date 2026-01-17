from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sentences = [
    "I absolutely loved this internship experience.",
    "The product quality was very poor and disappointing."
]

results = sentiment_analyzer(sentences)

for text, result in zip(sentences, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']} | Confidence: {result['score']:.4f}")
    print("-" * 50)
