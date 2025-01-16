# Rock and Mine Prediction using Machine Learning
## Objective
The Rock and Mine Prediction project aims to classify objects as either rocks or mines based on sonar returns. Using a supervised machine learning approach, the project employs a labeled dataset to build a predictive model.

### Skills Learned
- Data preprocessing and cleaning techniques.
- Feature engineering and selection.
- Model training and hyperparameter tuning.
- Evaluation of classification models.
- Proficiency in visualization tools like Matplotlib and Seaborn.

### Tools Used
- Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)
- Jupyter Notebook for interactive development
- Version control with Git and GitHub

### Steps to Run the Project
1. Clone the repository.
2. Install the required Python libraries.
3. Run the Jupyter Notebook to explore the dataset and train models.
4. Review results and predictions.

```python
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load Dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
df = pd.read_csv(data_url, header=None)

# Data Overview
print("Dataset Shape:", df.shape)
print("First 5 Rows:\n", df.head())

# Assign Features and Labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode Labels
y = y.map({"R": 0, "M": 1})  # 0 for Rock, 1 for Mine

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importances = model.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), feature_importances, align='center')
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()

# Save the Model (Optional)
import joblib
joblib.dump(model, "rock_mine_predictor.pkl")

# Load and Test Saved Model
loaded_model = joblib.load("rock_mine_predictor.pkl")
sample_data = np.array([X.iloc[0]])
prediction = loaded_model.predict(sample_data)
print("Prediction for Sample Data (0=Rock, 1=Mine):", prediction)
```

### Results

![image](https://github.com/user-attachments/assets/4e350cd9-1326-4bed-b3d8-df337bc6c69b)

1. Dataset Overview.
2. Accuracy, Confusion Matrix, and Classification Report output.
3. Feature Importance Visualization.

---

This project can be further expanded by testing additional machine learning models or incorporating deep learning techniques.
