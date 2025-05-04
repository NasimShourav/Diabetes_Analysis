# 🩺 Diabetes Prediction Project

This project uses the **Pima Indians Diabetes Dataset** to predict whether a person has diabetes based on various medical features. The dataset is analyzed using **Python**, with libraries such as `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` for building predictive models.

---

## 📌 Objective

The primary goal of this project is to:
1. **Predict** whether a person has diabetes or not (`Outcome = 1` or `0`).
2. Use **Logistic Regression** and **Random Forest Classifier** to evaluate the accuracy of the model.
3. Visualize the model’s performance using **ROC Curve** and **AUC (Area Under the Curve)**.

---

## 📂 Dataset Overview

- **Source**: Kaggle - [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- **Rows**: 768
- **Columns**: 9
  - **Features**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
  - **Target**: `Outcome` (1 = diabetic, 0 = non-diabetic)

---

## 🔧 Tools & Libraries

- **Python 3.x**
- **pandas**, **numpy** for data manipulation
- **matplotlib**, **seaborn** for data visualization
- **scikit-learn** for machine learning

---

## 🔍 Project Workflow

### 1. **Data Cleaning**
- Replaced invalid zero values with `NaN` in medical columns (`Glucose`, `BMI`, etc.).
- Filled missing data with the median of each column to avoid bias.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized feature correlations with a **heatmap**.
- Investigated relationships between features and the target variable (`Outcome`).

### 3. **Feature Engineering**
- Standardized features using **StandardScaler** for better model performance.

### 4. **Modeling**
- Trained two machine learning models:
  - **Logistic Regression**: A basic classification algorithm.
  - **Random Forest Classifier**: An ensemble model.
  
### 5. **Evaluation**
- Used **classification report** to assess model performance.
- Evaluated the models using the **ROC curve** and **AUC** score.

### 6. **Results**
- Random Forest achieved a **74% accuracy** and was considered a good predictor.
- Logistic Regression was evaluated using the **ROC curve**.

---

## 📈 Visuals

- **Correlation Heatmap**: Displays the relationships between the dataset features.
- **Confusion Matrix**: Helps visualize classification performance.
- **ROC Curve**: Used to assess the model’s ability to discriminate between positive and negative classes.

---

## 📊 Results Summary

- **Random Forest**: ~74% accuracy
- **Logistic Regression**: Evaluated with **ROC Curve** and **AUC**
- **Important Features**: `Glucose`, `BMI`, and `Age` were the most influential for diabetes prediction.

---

## 📝 Files in the Repository

- **`Diabetes_Prediction_Project.ipynb`** – Jupyter notebook containing all the code and visualizations.
- **`diabetes.csv`** – Dataset used for training.
- **`README.md`** – Project overview and documentation.

---

## 🛠️ Future Improvements

- **Advanced Models**: Test models like **XGBoost** or **LightGBM** for better performance.
- **Handling Class Imbalance**: Use **SMOTE** or **class weighting** to address imbalanced classes.
- **Model Deployment**: Build a web app using **Flask** or **Streamlit** to make predictions interactively.


