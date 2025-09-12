from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import joblib

class SpamModelTrainer:
    def __init__(self, max_features=3000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = MultinomialNB()
        
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the spam detection model
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, cm, report, X_test, y_test, y_pred
    
    def save_model(self, filepath):
        """
        Save the trained model and vectorizer
        """
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model and vectorizer
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        return self