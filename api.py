import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Import RAG module
from rag import answer_query_with_rag

# Load environment variables
load_dotenv()

app = FastAPI(title="House Price Prediction API with RAG")

# Load ML components
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_compressed.joblib"
DATA_PATH = BASE_DIR / "data" / "india_housing_prices.csv"

model = joblib.load(MODEL_PATH)
feature_names = list(model.feature_names_in_)
df = pd.read_csv(DATA_PATH)

class QueryRequest(BaseModel):
    query: str

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    query: Optional[str] = "Explain this prediction"

@app.get("/")
async def root():
    return {"message": "House Price Prediction API is running!"}

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    """Answers a query using the RAG system."""
    try:
        answer = answer_query_with_rag(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-and-explain")
async def predict_and_explain(request: PredictionRequest):
    """Predicts house price and provides an explanation using RAG."""
    try:
        # 1. Feature Engineering (Simplified adaptation of app.py logic)
        input_data = request.features
        row = {feat: 0 for feat in feature_names}
        
        # Map basic features
        for k, v in input_data.items():
            if k in row:
                row[k] = v
            
            # Map Categorical features (One-hot encoding)
            # Example: "State": "Delhi" -> "State_Delhi": 1
            key = f"{k}_{v}"
            if key in row:
                row[key] = 1

        # Fallback for Price_per_SqFt if not provided
        if "Price_per_SqFt" not in row or row["Price_per_SqFt"] == 0:
            row["Price_per_SqFt"] = df["Price_per_SqFt"].median()

        # Handle Amenities (special case in app.py)
        if "Amenities" in input_data:
            amenities_list = input_data["Amenities"]
            if isinstance(amenities_list, list):
                # Look for matching Amenities columns
                for f in feature_names:
                    if f.startswith("Amenities_"):
                        stored_set = set(a.strip() for a in f.replace("Amenities_", "").split(","))
                        if stored_set == set(amenities_list):
                            row[f] = 1
                            break

        input_df = pd.DataFrame([row], columns=feature_names)
        
        # 2. Predict
        prediction = model.predict(input_df)[0]
        
        # 3. Explain using RAG
        # Create a detailed prompt for the explanation
        explain_query = f"""
        The predicted price for a house with the following features is ₹{prediction:,.2f} Lakhs.
        Features: {input_data}
        User Query: {request.query}
        Please explain why this price is reasonable or factor in any relevant market trends.
        """
        explanation = answer_query_with_rag(explain_query)
        
        return {
            "predicted_price": float(prediction),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
