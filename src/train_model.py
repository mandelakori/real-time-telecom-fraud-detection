import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
import joblib
from preprocessing import clean_text
import numpy as np

# data loading and processing
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'messages.csv')
df = pd.read_csv(data_path, names=['label', 'message'])
df['message'] = df['message'].apply(clean_text)

# 80:20 (training:testing) split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# converting text to vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# model training and evaluating
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# print classification report
report = classification_report(y_test, y_pred)
print(report)

# calculate weighted F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
threshold = 0.90  # only save if F1 >= 90%

# optional: inspect average probabilities for each class on test set
probs = model.predict_proba(X_test_vec)
classes = model.classes_
avg_probs = {cls: np.mean(probs[:, i]) for i, cls in enumerate(classes)}
print("Average prediction probabilities on test set:")
for cls, prob in avg_probs.items():
    print(f"  {cls}: {prob:.3f}")

# model save
if f1 >= threshold:
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    joblib.dump(model, os.path.join(BASE_DIR, 'models', 'fraud_classifier.pkl'))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'))
    print(f"Model saved successfully. Weighted F1-score = {f1:.3f}")
else:
    print(f"Model not saved. Weighted F1-score = {f1:.3f}. Model requires improved training.")
