import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load any sports dataset (football, basketball, tennis, etc.)
df = pd.read_csv("sports_matches.csv")

# Features & Target
X = df.drop(columns=["Outcome"])  # numeric/categorical features
y = df["Outcome"]  # Win / Loss / Draw

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
