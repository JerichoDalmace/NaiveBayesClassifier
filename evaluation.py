from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lime.lime_text import LimeTextExplainer

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
    
    def generate_report(self):
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        self.plot_confusion_matrix()
        self.analyze_feature_importance()
        self.generate_lime_explanation()
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def analyze_feature_importance(self):
        tfidf = self.model.pipeline.named_steps['tfidf']
        clf = self.model.pipeline.named_steps['clf']
        
        feature_names = tfidf.get_feature_names_out()
        feature_importance = clf.feature_log_prob_
        
        # Plot top features for each class
        plt.figure(figsize=(12, 6))
        for i, class_label in enumerate(clf.classes_):
            top_features = np.argsort(feature_importance[i])[-10:]
            plt.subplot(1, 2, i+1)
            plt.barh(range(10), feature_importance[i][top_features])
            plt.yticks(range(10), [feature_names[j] for j in top_features])
            plt.title(f'Top Features for Class {class_label}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def generate_lime_explanation(self):
        explainer = LimeTextExplainer(class_names=self.model.pipeline.named_steps['clf'].classes_)
        
        # Generate explanation for a random test instance
        idx = np.random.randint(len(self.X_test))
        exp = explainer.explain_instance(
            self.X_test.iloc[idx],
            self.model.predict_proba,
            num_features=6
        )
        exp.save_to_file('lime_explanation.html')