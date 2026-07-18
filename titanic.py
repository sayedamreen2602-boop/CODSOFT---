import pandas as pd
import requests
import io

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Titanic dataset URL
url = "https://raw.githubusercontent.com/Anshumank399/Titanic-Disaster/master/train.csv"

print("Downloading dataset...")

response = requests.get(url)
data = pd.read_csv(io.StringIO(response.text))

print("\nFirst 5 Rows:")
print(data.head())

# Select required columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Fill missing Age values
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Convert Gender to numbers
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features and Target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Predict Example Passenger
sample = [[3, 0, 22, 7.25]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction: Passenger Survived")
else:
    print("\nPrediction: Passenger Did Not Survive")