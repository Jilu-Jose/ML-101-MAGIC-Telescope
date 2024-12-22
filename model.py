import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle


data = pd.read_csv("magic04.csv", header=None)

data.columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]

data['class'] = data['class'].map({'g': 1, 'h': 0})

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


new_data = [[30.0, 15.0, 1500.0, 0.3, 0.1, 5.0, 50.0, 10.0, 0.1, 200.0]]
prediction = model.predict(new_data)
print("Prediction (1: Gamma, 0: Hadron):", prediction)


pickle.dump(model, open("model.pkl", "wb"))
