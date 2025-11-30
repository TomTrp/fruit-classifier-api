# Fruit Classification with Decision Tree & Random Forest

This project demonstrates how to **train and use machine learning models** (Decision Tree and Random Forest) to classify fruits based on numeric and categorical features.

## 🔧 Features
- ✅ Download dataset directly from Kaggle via kagglehub
- 🛠️ Preprocessing pipeline with ColumnTransformer and OneHotEncoder
- 📏 Train both Decision Tree and Random Forest models
- 🔄 Export trained models as .joblib files for reuse
- 📊 Evaluate models with accuracy and classification report
- 🔗 Easy prediction on new data using saved model

## 🚀 How to Run

### 1. Install Required Tools
- Python 3.8 or higher
- Required Python packages:
    `pip install pandas numpy scikit-learn joblib kagglehub`
- Kaggle API token (to download dataset via kagglehub) or refer to this documentation: [KaggleHub](https://github.com/Kaggle/kagglehub)
  
### 2. Train the Models
- Run `train.py` to download dataset, train models, and save them as .joblib:
    `python test.py`
- Output files:
    `fruit_dt_model.joblib` → Decision Tree model
    `fruit_rf_model.joblib` → Random Forest model
- **Note**: If the dataset cannot be downloaded from Kaggle, you can use the CSV file directly from the `datasets` folder instead.
  
### 3. Test / Predict New Data
- Use `test.py` to load the saved Random Forest model and predict new fruit data:
    `python test.py`
- Example output:
    `Predicted fruit: banana`

### 4. Notes
- Ensure **input data columns** match the features used in training:
    `size (cm), weight (g), avg_price (₹), shape, color, taste`
- To switch training algorithm in `train.py`, update the `train_method` in `createPipeline()` to
    `"decision_tree"` or `"random_forest"`
