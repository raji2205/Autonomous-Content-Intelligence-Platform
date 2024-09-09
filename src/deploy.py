from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

summarizer = pipeline("summarization")
classifier = pipeline("text-classification")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    summary = summarizer(data['text'])[0]['summary_text']
    return jsonify({'summary': summary})

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    classification = classifier(data['text'])
    return jsonify({'classification': classification})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
