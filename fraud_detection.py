import os
import requests
import pandas as pd

# Dataset URL
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

filename = "creditcard.csv"

# Download dataset if not present
if not os.path.exists(filename):
    print("Downloading dataset...")
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)
    print("Download Complete!")

print("Loading dataset...")

data = pd.read_csv(filename)

print("\nFirst 5 Rows:")
print(data.head())

# Features and Target
X = data.drop("Class", axis=1)
y = data["Class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

print("\nTraining Model...")
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("\nModel Accuracy:", accuracy)

prediction = model.predict(X_test[:10])

print("\nPredictions:")
print(prediction)

print("\nActual Values:")
print(y_test.iloc[:10].values)