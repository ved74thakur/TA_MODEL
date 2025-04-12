import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Let's define a function to load and prepare the dataset
def load_data(file_path):
    """
    Load the dataset containing summaries and Captions
    """
    # Assuming the data is in CSV format with columns: Summary, Caption, Category
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} rows")
    return df

# Feature engineering functions
def extract_features(df):
    """
    Extract features from Summary and Caption pairs
    """
    print("Extracting features...")
    
    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for summaries and Captions
    Summary_embeddings = model.encode(df['Summary'].tolist())
    Caption_embeddings = model.encode(df['Caption'].tolist())
    
    # Calculate cosine similarity between Summary and Caption
    cosine_similarities = []
    for i in range(len(df)):
        similarity = 1 - cosine(Summary_embeddings[i], Caption_embeddings[i])
        cosine_similarities.append(similarity)
    
    # Create additional features
    features = pd.DataFrame({
        'cosine_similarity': cosine_similarities,
        'Summary_length': df['Summary'].apply(len),
        'Caption_length': df['Caption'].apply(len),
        'length_ratio': df['Summary'].apply(len) / df['Caption'].apply(len),
        'word_count_Summary': df['Summary'].apply(lambda x: len(x.split())),
        'word_count_Caption': df['Caption'].apply(lambda x: len(x.split())),
        'word_count_ratio': df['Summary'].apply(lambda x: len(x.split())) / df['Caption'].apply(lambda x: len(x.split())),
    })
    
    # You could add more features here
    return features, df['Category']

# Model training and evaluation
def train_evaluate_model(X, y, n_splits=5):
    """
    Train and evaluate models using stratified k-fold cross-validation
    """
    # Initialize the stratified k-fold 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define the models to try
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'SVM': SVC(class_weight='balanced', probability=True),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100)
    }
    
    # Metrics to evaluate
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'f1_macro': make_scorer(f1_score, average='macro')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining and evaluating {name}:")
        
        from imblearn.pipeline import Pipeline

        # Apply SMOTE for handling class imbalance
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y)) - 1))), # Then apply SMOTE
            ('classifier', model)  
        ])
        
        # Cross-validation
        cv_results = cross_validate(pipeline, X, y, 
                                    cv=skf, 
                                    scoring=scoring, 
                                    return_estimator=True,
                                    return_train_score=True)
        
        # Store results
        results[name] = {
            'cv_results': cv_results,
            'accuracy': cv_results['test_accuracy'].mean(),
            'precision': cv_results['test_precision_macro'].mean(),
            'recall': cv_results['test_recall_macro'].mean(),
            'f1': cv_results['test_f1_macro'].mean()
        }
        
        print(f"  Accuracy: {results[name]['accuracy']:.4f}")
        print(f"  Precision (macro): {results[name]['precision']:.4f}")
        print(f"  Recall (macro): {results[name]['recall']:.4f}")
        print(f"  F1 Score (macro): {results[name]['f1']:.4f}")
    
    return results

# Function to select the best model
def select_best_model(results):
    """
    Select the best model based on F1 macro score
    """
    scores = {name: results[name]['f1'] for name in results}
    best_model = max(scores, key=scores.get)
    print(f"\nBest model: {best_model} with F1 macro score: {scores[best_model]:.4f}")
    return best_model, results[best_model]

# Function for detailed evaluation of the best model
def detailed_evaluation(X, y, best_model_name, best_model_results):
    """
    Perform detailed evaluation of the best model
    """
    print("\nDetailed evaluation of the best model...")
    
    # Get the best estimator from cross-validation
    best_estimator = best_model_results['cv_results']['estimator'][0]
    
    # Initialize a new StratifiedKFold for evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    
    all_predictions = []
    all_true_labels = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Refit the pipeline on this fold
        best_estimator.fit(X_train, y_train)
        
        # Get predictions
        y_pred = best_estimator.predict(X_test)
        
        # Store for aggregation
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_test)
    
    # Overall classification report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions))
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.show()
    
    return {
        'classification_report': classification_report(all_true_labels, all_predictions, output_dict=True),
        'confusion_matrix': cm
    }

# Feature importance analysis for interpretability
def analyze_feature_importance(X, best_model_results):
    """
    Analyze feature importance for the best model (if applicable)
    """
    best_estimator = best_model_results['cv_results']['estimator'][0]
    
    # Check if the model supports feature importance
    if hasattr(best_estimator[-1], 'feature_importances_'):
        importances = best_estimator[-1].feature_importances_
        feature_names = X.columns
        
        # Create and plot feature importance
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    elif hasattr(best_estimator[-1], 'coef_'):
        # For models like logistic regression
        coefficients = best_estimator[-1].coef_
        feature_names = X.columns
        
        # If multiclass, take average absolute coefficient
        if len(coefficients.shape) > 1:
            importances = np.mean(np.abs(coefficients), axis=0)
        else:
            importances = np.abs(coefficients)
            
        # Create and plot feature importance
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    else:
        print("Feature importance not available for this model type")
        return None

# Main function to run the entire pipeline
def main(file_path):
    """
    Run the complete classification pipeline
    """
    # Step 1: Load data
    df = load_data(file_path)
    
    # Step 2: Data analysis and preprocessing
    print("\n=== Dataset Analysis ===")
    print(f"Total number of articles: {len(df)}")
    print("Category Distribution:")
    print(df['Category'].value_counts())
    print(f"Total number of different categories: {df['Category'].nunique()}")
    print("Text Length Statistics:")
    print(f"Average Summary Length: {df['Summary'].apply(len).mean():.2f} characters")
    print(f"Average Caption Length: {df['Caption'].apply(len).mean():.2f} characters")
    print(f"Longest Summary: {df['Summary'].apply(len).max()} characters")
    print(f"Longest Caption: {df['Caption'].apply(len).max()} characters")
    print("Analysis completed successfully!")
    print(f"Dataset contains {len(df)} articles")
    
    # Step 3: Feature extraction
    X, y = extract_features(df)
    
    # Step 4: Model training and evaluation with class imbalance handling
    results = train_evaluate_model(X, y)
    
    # Step 5: Select the best model
    best_model_name, best_model_results = select_best_model(results)
    
    # Step 6: Detailed evaluation
    evaluation_metrics = detailed_evaluation(X, y, best_model_name, best_model_results)
    
    # Step 7: Feature importance analysis
    feature_importance = analyze_feature_importance(X, best_model_results)
    
    print("\nClassification pipeline completed successfully!")
    
    return {
        'X': X,
        'y': y,
        'results': results,
        'best_model_name': best_model_name,
        'best_model_results': best_model_results,
        'evaluation_metrics': evaluation_metrics,
        'feature_importance': feature_importance
    }

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "manual_annotations.csv"
    
    # Uncomment the line below to run the pipeline
    pipeline_results = main(file_path)
    
    # print("\nTo run the classification pipeline, please provide the correct file path to your dataset.")
    # print("Example usage: pipeline_results = main('path_to_your_dataset.csv')")