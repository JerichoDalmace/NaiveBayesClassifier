from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import joblib

class SentimentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        
        self.param_grid = {
            'tfidf__max_features': [1000, 2000, 3000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__alpha': [0.1, 0.5, 1.0]
        }
    
    def train(self, X_train, y_train):
        # Perform grid search
        grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Use best model
        self.pipeline = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_score_
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def save_model(self, filepath):
        joblib.dump(self.pipeline, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        classifier = cls()
        classifier.pipeline = joblib.load(filepath)
        return classifier