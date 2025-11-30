# Fruit Classification API with Decision Tree & Random Forest

This project demonstrates how to **train and deploy machine learning models** (Decision Tree and Random Forest) to classify fruits based on numeric and categorical features.
It also includes a FastAPI service to serve predictions via REST API.

## 🔧 Features
- ✅ Download dataset directly from Kaggle via kagglehub
- 🛠️ Preprocessing pipeline with ColumnTransformer and OneHotEncoder
- 📏 Train both Decision Tree and Random Forest models
- 🔄 Export trained models as .joblib files for reuse
- 📊 Evaluate models with accuracy and classification report
- 🔗 Easy prediction on new data using saved model
- 🌐 Deploy trained models as a FastAPI REST API
- ⚙️ Ready for Docker containerization

## 🚀 How to Run

### 1. Install Required Tools
- Python 3.10 or higher
- Required Python packages:
    `pip install pandas numpy scikit-learn joblib kagglehub fastapi uvicorn seaborn pydantic`
- Kaggle API token (to download dataset via kagglehub) **or refer to this documentation**: [KaggleHub](https://github.com/Kaggle/kagglehub)
  
### 2. Train the Models
- Run `train.py` to download dataset, train models, and save them as .joblib:
    `python test.py`
- Output files:
    `fruit_dt_model.joblib` → Decision Tree model
    `fruit_rf_model.joblib` → Random Forest model
- **Note**: If the dataset cannot be downloaded from Kaggle, you can use the CSV file directly from the `datasets` folder instead.
  
### 3. Run FastAPI Server
- Start the FastAPI service:
    `uvicorn app.main:app --reload`
- Open your browser at: 
    [Swagger UI](http://localhost:8000/docs)
- Available endpoints:
| Method | Endpoint        | Description                 |
| ------ | :-------------: | ---------------------------:|
| `GET`  | `/health`       | Health check                |
| `POST` | `/predict/dt`   | Predict using Decision Tree |
| `POST` | `/predict/rf`   | Predict using Random Forest |

- Example JSON Request:
```json
{
    "size_cm": 12.3,
    "weight_g": 150,
    "avg_price": 40,
    "shape": "long",
    "color": "yellow",
    "taste": "sweet"
}
```

- Example Response:
```json
{
  "model": "random_forest",
  "prediction": "banana"
}
```

### 4. (Optional) Run with Docker
- Build Docker image:
    `docker build -t fruit-classifier-api .`
- Run the container:
    `docker run -d -p 8000:8000 fruit-classifier-api`
- Visit: `http://localhost:8000/docs`

### 5. Notes
- Ensure **input data columns** match the features used in training:
    `size (cm), weight (g), avg_price (₹), shape, color, taste`
- To switch training algorithm in `train.py`, update the `train_method` in `createPipeline()` to
    `"decision_tree"` or `"random_forest"`
