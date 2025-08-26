import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
from preprocessing import clean_text

# data loading and processing
df = pd.read_csv('data/messages.csv', names=['label', 'message'])
df['message'] = df['message'].apply(clean_text)

# 80:20 (training:testing) split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# converting text to [computer] readable vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# model training and evaluating
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# saving model to path
joblib.dump(model, 'models/fraud_classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
