import joblib
import os
from preprocessing import clean_text

# load model & vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'fraud_classifier.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print("=== Static Fraud Detection ===")
print("Type a message to classify. Type 'exit' to quit.\n")

while True:
    message = input("Enter message: ").strip()
    if message.lower() == 'exit':
        print("Exiting...")
        break
    if not message:
        print("No input detected. Please enter a message.")
        continue

    # clean message
    clean_msg = clean_text(message)
    msg_vec = vectorizer.transform([clean_msg])

    # calculate probabilities and likeliness
    probs = model.predict_proba(msg_vec)[0]
    classes = model.classes_
    prob_dict = {cls: probs[i] for i, cls in enumerate(classes)}
    predicted_class = max(prob_dict, key=prob_dict.get)
    predicted_confidence = prob_dict[predicted_class]

    # print confidence for all classes
    print("Prediction probabilities:")
    for cls, prob in prob_dict.items():
        print(f"  {cls.upper()}: {prob*100:.1f}%")

    print(f"\nFinal prediction: {predicted_class.upper()} ({predicted_confidence*100:.1f}% confidence)\n")
