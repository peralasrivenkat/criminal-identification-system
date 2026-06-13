# train_classifier.py
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Ensure the models folder exists
os.makedirs("models", exist_ok=True)

# Example feature data (replace with your PCA + ACO features)
X = np.random.rand(6, 200)  # 6 samples, 200 features
y = np.array([0, 0, 0, 1, 1, 1])  # Labels

# Train MLP classifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
model.fit(X, y)

# Save trained model to models folder
joblib.dump(model, "models/classifier_model.pkl")
print("✅ Trained model saved at models/classifier_model.pkl")