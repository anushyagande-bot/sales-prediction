import joblib
import pandas as pd

# Load model
model = joblib.load("model/sales_model.pkl")

# Example input
input_data = pd.DataFrame({
    'TV': [230.1],
    'Radio': [37.8],
    'Newspaper': [69.2]
})

# Predict
prediction = model.predict(input_data)

print("Predicted Sales:", prediction[0])