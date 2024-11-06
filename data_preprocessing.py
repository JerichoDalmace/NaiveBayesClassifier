import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os

class TextPreprocessor:
    def __init__(self):
        # Create nltk_data directory in the current working directory
        nltk.data.path.append(os.getcwd())
        
        # Download required NLTK data with error handling
        try:
            self._download_nltk_data()
        except Exception as e:
            print(f"Error downloading NLTK data: {str(e)}")
            print("Attempting to use existing NLTK data...")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_data(self):
        """Download required NLTK data with error handling"""
        required_packages = [
            'punkt',
            'stopwords',
            'wordnet',
            'omw-1.4'  # Added for better word tokenization
        ]
        
        # Create nltk_data directory if it doesn't exist
        data_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(data_dir, exist_ok=True)
        
        for package in required_packages:
            try:
                # Force download to the local directory
                nltk.download(package, download_dir=data_dir, quiet=True, raise_on_error=True)
            except Exception as e:
                print(f"Warning: Could not download {package}. Error: {str(e)}")
                # Try downloading without specifying directory
                try:
                    nltk.download(package, quiet=True)
                except Exception as inner_e:
                    print(f"Failed to download {package} to default location: {str(inner_e)}")
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenize with error handling
            try:
                tokens = word_tokenize(text)
            except LookupError:
                # Fallback to basic splitting if word_tokenize fails
                tokens = text.split()
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Warning: Error processing text: {str(e)}")
            return text

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Clean reviews
        data['cleaned_review'] = data['review'].apply(preprocessor.clean_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data['cleaned_review'],
            data['sentiment'],
            test_size=test_size,
            random_state=random_state,
            stratify=data['sentiment']
        )
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        raise