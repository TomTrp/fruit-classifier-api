import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Download csv from kaggle
def getFruitsDataSet():
    file_path = "pranavkapratwar/fruit-classification"
    path = kagglehub.dataset_download(file_path)
    csv_path = os.path.join(path, "fruit_classification_dataset.csv")
    print(csv_path)
    return csv_path

def createDecisionTreePipeline(preprocessor):
    dt_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    return dt_pipeline



# Load dataset
csv_path = getFruitsDataSet()
df = pd.read_csv(csv_path)
print(df)


# Extract features and target
X = df.drop("fruit_name", axis=1)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_pipeline = createDecisionTreePipeline(preprocessor)

dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
