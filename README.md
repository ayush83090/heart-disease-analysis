# 🫀 Heart Disease Analysis using Machine Learning

This Jupyter Notebook project explores and analyzes a heart disease dataset using data visualization, preprocessing techniques, and several machine learning models to predict the presence of heart disease in patients.

---

## 📊 Project Objectives

- Understand and explore the dataset related to heart disease.
- Perform data preprocessing and cleaning.
- Visualize key features and their correlations with heart disease.
- Train and evaluate machine learning models for prediction.
- Interpret model performance using metrics such as accuracy, precision, recall, and ROC AUC.

---

## 📁 Dataset

The dataset used is a publicly available **Heart Disease UCI dataset**, containing 303 records with 14 attributes:

- `age`: Age of the patient
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0–3)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results (0–2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels (0–3)
- `thal`: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)
- `target`: Presence of heart disease (1 = yes, 0 = no)

---

## 🔧 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 🚀 Workflow Overview

1. **Data Loading**
   - Load the dataset and understand the structure.

2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables (e.g., `thal`, `cp`)
   - Feature scaling using `StandardScaler`

3. **Exploratory Data Analysis**
   - Visualize feature distributions and correlations
   - Analyze the effect of different features on the target

4. **Model Building**
   - Train multiple models including:
     - Logistic Regression
     - K-Nearest Neighbors
     - Decision Tree
     - Random Forest
     - Support Vector Machine

5. **Model Evaluation**
   - Evaluate using confusion matrix, classification report, and ROC-AUC

---

## 📈 Results

The models were compared based on accuracy, F1-score, and ROC AUC. The best performing model was **[insert best model]** with an accuracy of **[insert accuracy]%**.

---

## 🧠 Conclusion

- Certain features such as chest pain type, thalassemia, and maximum heart rate were highly correlated with heart disease.
- Machine learning models can effectively help in early diagnosis and classification of heart disease risk.

---

## 📌 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-analysis.git
   cd heart-disease-analysis
