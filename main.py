from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = FastAPI()

MODEL_PATH = "model.pkl"

@app.post("/learn")
async def learn(file: UploadFile = File(...)):
    try:
        # Read CSV file
        df = pd.read_csv(file.file)

        # Check if the dataset has enough columns
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="CSV file must have at least two columns")

        # Prepare data
        X = df.iloc[:, :-1]  # Features (all except last column)
        y = df.iloc[:, -1]   # Target (last column)

        # Convert categorical target variable if needed
        if y.dtype == 'O':  
            le = LabelEncoder()
            y = le.fit_transform(y)
            joblib.dump(le, "label_encoder.pkl")

        # Split data and train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, MODEL_PATH)

        return {"message": "Model trained successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask")
async def ask(q: str):
    try:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=400, detail="Model not trained yet. Please upload a CSV file first.")

        # Load model
        model = joblib.load(MODEL_PATH)

        # Convert query into appropriate format (Assuming numeric input for now)
        data = [float(i) for i in q.split(",")]
        prediction = model.predict([data])

        # Decode categorical output if applicable
        if os.path.exists("label_encoder.pkl"):
            le = joblib.load("label_encoder.pkl")
            prediction = le.inverse_transform(prediction)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
