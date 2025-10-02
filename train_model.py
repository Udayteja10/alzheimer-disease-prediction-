import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
dataset_path = r"C:\Users\Dell\Desktop\rtrp project\Combined Dataset"

image_size = (64, 64)

X, y = [], []
class_names = sorted(os.listdir(dataset_path))

for label_index, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue
    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(image_size)
            X.append(np.array(img).flatten())
            y.append(label_index)
        except Exception as e:
            print(f"Skipped: {file_path}, error: {e}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Model Evaluation:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

joblib.dump((model, class_names), "ml_model.pkl")
print("✅ Model saved as 'ml_model.pkl'")
