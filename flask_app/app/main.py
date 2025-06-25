import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def load_and_preprocess_data(file_path):
    """
    Load the cleaned dataset and perform any necessary preprocessing steps.
    Assumes the file is in the same directory.
    """
    # Load the cleaned dataset
    df = pd.read_csv(file_path)
    print("Loaded dataset with shape:", df.shape)
    print("Columns:", df.columns.tolist())
    
    # Fix target column if needed: if the minimum value is 1, shift to start at 0.
    if df['Customer Satisfaction Rating'].min() == 1:
        df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'] - 1

    # Drop columns that are not needed (if they exist)
    cols_to_drop = ['Ticket ID', 'Customer Name', 'Customer Email', 'Date of Purchase']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    return df

def create_preprocessing_pipeline(X):
    """
    Create a preprocessing pipeline that applies scaling and one-hot encoding.
    It separates numeric and categorical features.
    """
    # Separate numeric and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Define numeric transformer: impute missing values and scale the data
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define categorical transformer: impute and one-hot encode
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    return preprocessor

def main():
    # Define file path for the cleaned dataset (ensure the file is in the same directory)
    file_path = "cleaned_dataset.csv"

    # Load data
    df = load_and_preprocess_data(file_path)

    # Separate features and target
    X = df.drop('Customer Satisfaction Rating', axis=1)
    y = df['Customer Satisfaction Rating']

    # Optional: Encode any remaining categorical columns that may not be handled by the pipeline
    # This is just a fallback; the preprocessing pipeline will handle most object types.
    label_enc = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = label_enc.fit_transform(X[col].astype(str))

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X)
    X_processed = preprocessor.fit_transform(X)

    # Balance the classes using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Define the XGBoost classifier
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')

    # Define hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Run hyperparameter search on training data
    random_search.fit(X_train, y_train)

    # Retrieve the best model
    best_model = random_search.best_estimator_
    print("Best Hyperparameters:", random_search.best_params_)

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

