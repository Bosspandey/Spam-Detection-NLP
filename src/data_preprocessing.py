import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt_tab')
except:
    nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess the text by:
        1. Removing special characters and digits
        2. Converting to lowercase
        3. Tokenizing
        4. Removing stopwords
        5. Applying stemming
        """
        # Clean the text
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        processed_words = [self.ps.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(processed_words)
    
    def preprocess_dataset(self, df, text_column):
        """
        Preprocess an entire dataset column
        """
        return df[text_column].apply(self.preprocess_text)