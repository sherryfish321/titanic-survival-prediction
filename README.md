# titanic-survival-prediction  
# 🚢 Titanic Survival Prediction

## 📌 Project Overview
This project applies machine learning techniques to the Kaggle **Titanic: Machine Learning from Disaster** dataset.  
The goal is to predict whether a passenger survived based on socio-demographic and travel-related features.  
It demonstrates a **complete ML pipeline**: Exploratory Data Analysis (EDA), preprocessing, feature engineering, model training, evaluation, and interpretation.

---

## 📂 Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Inspected data types and distributions.
   - Identified missing values (`Age`, `Fare`, `Embarked`, `Cabin`).
   - Observed that **Sex** and **Pclass** were strongly correlated with survival.

2. **Data Preprocessing**
   - Imputed missing values: `Age` (group median), `Fare` (median), `Embarked` (mode).
   - Converted `Cabin` → `Deck`, with missing values replaced by `"U"`.
   - Removed redundant columns (`Name`, `Ticket`, `Cabin`).
   - Ensured no missing values or object-type columns remained.

3. **Feature Engineering**
   - Extracted `Title` from names; grouped rare titles (e.g., Countess, Capt, Rev).
   - Created `FamilySize` = `SibSp + Parch + 1`.
   - Derived `IsAlone` = 1 if passenger traveled alone, else 0.
   - Binned continuous variables: `AgeBin`, `FareBin`.
   - Applied encoding (mapping / one-hot) for categorical variables.

4. **Modeling & Evaluation**
   - Models compared: Logistic Regression, Random Forest, XGBoost.
   - Performance:
     - Logistic Regression → Accuracy: **0.8547**, ROC-AUC: **0.8413**
     - Random Forest → Accuracy: **0.8045**, ROC-AUC: **0.7896**
     - XGBoost → Accuracy: **0.9106**, ROC-AUC: **0.9057**
   - XGBoost was selected as the final model.

5. **Hyperparameter Tuning**
   - Used GridSearchCV with 5-fold CV (ROC-AUC scoring).
   - Best parameters:
     ```python
     {
       "colsample_bytree": 0.8,
       "learning_rate": 0.05,
       "max_depth": 3,
       "n_estimators": 200,
       "subsample": 0.8
     }
     ```

6. **Feature Importance**
   - Top predictors: `Sex`, `Pclass`, `Title`, `IsAlone`, `FareBin`.
   - Confirmed that engineered features significantly contributed to performance.

7. **Submission**
   - Final model retrained on full training data.
   - Predictions generated for test set and saved as `submission.csv`.

---

## 📊 Results
- **Best Model:** XGBoost  
- **Accuracy:** 0.91  
- **ROC-AUC:** 0.905  
- Insights confirmed by feature importance analysis:  
  - Women and first-class passengers had much higher survival rates.  
  - Family-related and socio-economic engineered features improved predictive accuracy.
