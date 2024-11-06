from PyQt6.QtCore import QThread, pyqtSignal
import pandas as pd

class AnalysisThread(QThread):
    progress_updated = pyqtSignal(int)
    analysis_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, model, preprocessor):
        super().__init__()
        self.file_path = file_path
        self.model = model
        self.preprocessor = preprocessor
    
    def run(self):
        try:
            # Load CSV file
            df = pd.read_csv(self.file_path)
            
            if 'review' not in df.columns:
                raise ValueError("CSV file must contain a 'review' column")
            
            total_rows = len(df)
            processed_rows = 0
            
            # Process reviews
            cleaned_reviews = []
            for review in df['review']:
                cleaned_reviews.append(self.preprocessor.clean_text(review))
                processed_rows += 1
                progress = int((processed_rows / total_rows) * 50)
                self.progress_updated.emit(progress)
            
            df['cleaned_review'] = cleaned_reviews
            
            # Get predictions and probabilities
            predictions = self.model.predict(df['cleaned_review'])
            probabilities = self.model.predict_proba(df['cleaned_review'])
            
            # Add results to dataframe
            df['sentiment'] = predictions
            df['confidence'] = [max(prob) for prob in probabilities]
            
            self.progress_updated.emit(100)
            self.analysis_complete.emit(df)
            
        except Exception as e:
            self.error_occurred.emit(str(e))