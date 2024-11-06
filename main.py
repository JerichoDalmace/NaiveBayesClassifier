from data_preprocessing import load_and_preprocess_data
from model import SentimentClassifier
from evaluation import ModelEvaluator

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data('Train.csv')
    
    # Train model
    print("\nTraining model...")
    classifier = SentimentClassifier()
    best_score = classifier.train(X_train, y_train)
    print(f"Best cross-validation score: {best_score:.4f}")
    
    # Save model
    classifier.save_model('sentiment_classifier.pkl')
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(classifier, X_test, y_test)
    evaluator.generate_report()
    
    # Example predictions
    example_reviews = [
        "The professor was amazing and very helpful!",
        "I did not like the class; it was too hard.",
        "The course content was well-organized and engaging."
    ]
    
    print("\nExample Predictions:")
    predictions = classifier.predict(example_reviews)
    probabilities = classifier.predict_proba(example_reviews)
    
    for review, pred, prob in zip(example_reviews, predictions, probabilities):
        print(f"\nReview: {review}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {max(prob):.2f}")

if __name__ == "__main__":
    main()