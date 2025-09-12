from .data_preprocessing import TextPreprocessor
from .model_training import SpamModelTrainer
import pandas as pd

class SpamDetector:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.trainer = SpamModelTrainer()
        self.is_trained = False
        
    def load_and_preprocess_data(self, filepath, text_column, label_column, dataset_type='sms'):
        """
        Load and preprocess the dataset
        """
        if dataset_type == 'sms':
            # SMS data typically uses tab separation with no header
            df = pd.read_csv(filepath, sep='\t', names=['label', 'text'])
            # Convert labels to binary (spam=1, ham=0)
            df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        else:
            # For other datasets, assume standard CSV format
            df = pd.read_csv(filepath)
        
        # Preprocess the text
        df['processed_text'] = self.preprocessor.preprocess_dataset(df, text_column)
        
        return df
    
    def train_model(self, df, text_column='processed_text', label_column='label'):
        """
        Train the spam detection model
        """
        # Create TF-IDF features
        X = self.trainer.vectorizer.fit_transform(df[text_column]).toarray()
        y = df[label_column]
        
        # Train the model
        accuracy, cm, report, X_test, y_test, y_pred = self.trainer.train(X, y)
        self.is_trained = True
        
        return accuracy, cm, report, X_test, y_test, y_pred
    
    def predict_text(self, text):
        """
        Predict if a text is spam or not
        """
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")
        
        # Preprocess the text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Transform using the vectorizer
        text_vector = self.trainer.vectorizer.transform([processed_text]).toarray()
        
        # Make prediction
        prediction = self.trainer.model.predict(text_vector)
        prediction_proba = self.trainer.model.predict_proba(text_vector)
        
        return prediction[0], prediction_proba[0]
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        self.trainer.save_model(filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        self.trainer.load_model(filepath)
        self.is_trained = True
        return self