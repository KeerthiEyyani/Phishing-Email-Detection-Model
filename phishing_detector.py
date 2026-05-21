import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load dataset
data = pd.read_csv("emails.csv")

# Features and labels
X = data["Email"]
y = data["Label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved as model.pkl")

# Test custom email
print("\n----- Test Email -----")

test_email = input("Enter Email Content: ")

test_vector = vectorizer.transform([test_email])

prediction = model.predict(test_vector)

print("\nPrediction:", prediction[0])