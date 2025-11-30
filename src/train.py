import kagglehub
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Download csv from kaggle
# If this csv no longer available use it from datasets/
def getFruitsDataSet():
    file_path = "pranavkapratwar/fruit-classification"
    path = kagglehub.dataset_download(file_path)
    csv_path = os.path.join(path, "fruit_classification_dataset.csv")
    print(csv_path)
    return csv_path

# Pipeline
def createPipeline(preprocessor, train_method = ""):
    if train_method == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=42)
    elif train_method == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"train_method '{train_method}' not supported train_methpd")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline

# Load dataset
csv_path = getFruitsDataSet()
df = pd.read_csv(csv_path)
print(df)

# Extract features and target
x = df.drop("fruit_name", axis=1)
y = df["fruit_name"]

# Define categorical and numeric columns
categorical_features = ["shape", "color", "taste"]
numeric_features = ["size (cm)", "weight (g)", "avg_price (₹)"]

# Create ColumnTransformer for encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features)
    ],
    remainder="passthrough"
)

# Split Train/Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt_pipeline = createPipeline(preprocessor, "decision_tree")
rf_pipeline = createPipeline(preprocessor, "random_forest")

# Train
dt_pipeline.fit(x_train, y_train)
rf_pipeline.fit(x_train, y_train)

# Test
y_pred_dt = dt_pipeline.predict(x_test)
y_pred_rf = rf_pipeline.predict(x_test)

# Evaluate result
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

dump(dt_pipeline, "fruit_dt_model.joblib")
dump(rf_pipeline, "fruit_rf_model.joblib")
