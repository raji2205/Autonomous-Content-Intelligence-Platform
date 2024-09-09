import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

if __name__ == "__main__":
    df = load_data('data/raw/crawled_data.json')
    df['processed_content'] = df['content'].apply(lambda x: preprocess_text(' '.join(x)))
    df.to_csv('data/processed/processed_data.csv', index=False)
