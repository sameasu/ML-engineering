import joblib
import os
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#tokenizer = AutoTokenizer.from_pretrained("markusleonardo/DistilRoBERTa-For-Semantic-Similarity")
model = AutoModelForSequenceClassification.from_pretrained("markusleonardo/DistilRoBERTa-For-Semantic-Similarity")

class ModelPredictor:
    def __init__(self, model_path: str):
        if os.path.exists('model/svm_model.pkl'):
            self.model = joblib.load(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained("markusleonardo/DistilRoBERTa-For-Semantic-Similarity")

    def predict(self, text: str):
        prediction = self.model.predict([text])[0]
        probability = self.model.predict_proba([text])[0]
        return {"prediction": prediction, "probability": probability.tolist()}
