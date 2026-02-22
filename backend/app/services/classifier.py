import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import os

class TicketClassifierService:
    def __init__(self):
        # Current file is in backend/app/services/classifier.py
        # We want to reach backend/training/
        app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/app
        backend_path = os.path.dirname(app_path) # backend
        training_path = os.path.join(backend_path, 'training')
        model_path = os.path.join(training_path, 'models', 'fine_tuned_bert')
        
        # Load the fine-tuned BERT model and tokenizer
        # local_files_only=True ensures paths with spaces work correctly
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        
        # Load the label encoder from the model directory (same location as model weights)
        with open(os.path.join(model_path, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        self.model.eval()

    def predict(self, text):
        # First, check for obvious sales intent using keywords
        # This handles cases where the BERT model might misclassify based on tech keywords
        text_lower = text.lower()
        
        # Sales intent keywords - presence of these with purchase context = Sales
        sales_keywords = [
            'buy', 'purchase', 'order', 'pricing', 'price', 'quote', 
            'interested in buying', 'would like to buy', 'want to buy',
            'looking to purchase', 'need to order', 'how much does',
            'cost of', 'available for sale', 'in stock', 'can i get',
            'interested in purchasing', 'like to order', 'place an order'
        ]
        
        # Check for sales intent
        has_sales_intent = any(keyword in text_lower for keyword in sales_keywords)
        
        # If has sales intent AND doesn't have clear problem indicators, classify as Sales
        problem_indicators = [
            'not working', 'broken', 'error', 'issue', 'problem', 'failed', 
            'crash', 'bug', 'fix', 'help me fix', 'stopped working', 
            'doesn\'t work', 'won\'t start', 'can\'t access'
        ]
        has_problem = any(indicator in text_lower for indicator in problem_indicators)
        
        if has_sales_intent and not has_problem:
            return "Sales and Pre-Sales"
        
        # Fall back to BERT model for other cases
        # Preprocess text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get the predicted class index
        predicted_class_id = torch.argmax(logits, dim=1).item()
        
        # Decode the class label
        predicted_label = self.label_encoder.inverse_transform([predicted_class_id])[0]
        
        return predicted_label
