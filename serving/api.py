import joblib
import pandas as pd
from fastapi import FastAPI

def drop_columns(X):
    columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    return X[columns]

# Load pipeline
pipeline = joblib.load("../artifacts/pipeline.pkl")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Titanic survival prediction API"}

@app.post("/predict")
async def perdict(passenger: dict):

    try:
        if not isinstance(passenger, dict):
            raise ValueError("Invalid input data")

        # List of required columns (update this based on your model's requirements)
        required_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        if not all(col in passenger for col in required_columns):
            raise ValueError(f"Input data must contain the following columns: {required_columns}")

        data = pd.DataFrame([passenger])
        data = drop_columns(data)

        prediction = pipeline.predict(data)
    except Exception as e:
        return {"error": str(e)}

    return {"Survived": prediction[0].item()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)