# 🩺 Diabetes Prediction Using Machine Learning

This project demonstrates how to predict whether a person has diabetes based on diagnostic measurements using machine learning techniques.  
The implementation includes data loading, preprocessing, feature analysis, model training, evaluation, and prediction — all performed in the notebook **`Python Implementation (1) (1).ipynb`**.


 # 📁 Dataset

The dataset used in this project is the **Pima Indians Diabetes Dataset**, which contains medical information about female patients of Pima Indian heritage.

**Dataset Details:**
- **Total records:** 768  
- **Features:** 8 input variables and 1 output variable  
- **Target variable:** `Outcome` (1 = Diabetes, 0 = No Diabetes)

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Family history function (genetic influence) |
| Age | Age in years |
| Outcome | Class variable (0 or 1) |

📦 **Dataset Source:**  
[Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 🧰 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
🧹  Data Preprocessing
Loaded dataset using Pandas

Checked for missing and zero values in key columns

Replaced invalid (0) values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI with median values

Normalized data using StandardScaler

Split dataset into:

Training set: 80%

Testing set: 20%

🧪 Model Building
Selected Logistic Regression as the base model for classification
(can be replaced with SVM, RandomForest, or KNN for comparison)

Trained the model using training data

Predicted diabetes presence on test data

📈 Model Evaluation
Metric	Value (Example)
Accuracy	78%
Precision	0.77
Recall	0.71
F1-score	0.74

📊 Evaluation includes:

Confusion matrix

Classification report

ROC-AUC visualization (optional)

🔍 Prediction Example
python
Copy code
# Example: Predict on a new sample
input_data = np.array([[2, 120, 70, 25, 80, 28.0, 0.35, 35]])
scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)

if prediction[0] == 1:
    print("The person is likely diabetic.")
else:
    print("The person is not diabetic.")
🚀 How to Run
Open the notebook:

bash
Copy code
jupyter notebook "Python Implementation (1) (1).ipynb"
Run each cell step-by-step:

Load and clean data

Split into train/test sets

Train model

Evaluate performance and visualize results

🧩 Visualization
Pairplot and correlation heatmap for feature analysis

Distribution of glucose and BMI levels

Confusion matrix for model performance

Example:

python
Copy code
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Diabetes Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
