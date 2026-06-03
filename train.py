import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/sales_data.csv")

# Remove unnamed column if exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Show columns
print(df.columns)

# Features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("Model Accuracy:", score)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/sales_model.pkl")

print("Model saved successfully!")