import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text Cleaning Function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Apply cleaning
df['message'] = df['message'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test Custom Message
while True:
    msg = input("\nEnter message to check spam (or type exit): ")

    if msg.lower() == "exit":
        break

    msg_clean = clean_text(msg)
    msg_vector = vectorizer.transform([msg_clean])
    result = model.predict(msg_vector)

    if result[0] == 1:
        print("⚠️ Spam Message")
    else:
        print("✅ Not Spam")