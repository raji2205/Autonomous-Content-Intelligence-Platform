from transformers import pipeline

def summarize_text(text, max_length=150):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def classify_text(text):
    classifier = pipeline("text-classification")
    result = classifier(text)
    return result

if __name__ == "__main__":
    example_text = "Your content here."
    print("Summary:", summarize_text(example_text))
    print("Classification:", classify_text(example_text))
