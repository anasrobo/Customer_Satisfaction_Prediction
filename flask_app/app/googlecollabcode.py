pip install imbalanced-learn

from google.colab import files
uploaded = files.upload()

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
df = pd.read_csv('customer_support_tickets.csv')

# --- Initial Exploration ---
print("Initial shape:", df.shape)
print("Columns:\n", df.columns)
df.info()

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Descriptive statistics
print("\nDescriptive stats:\n", df.describe())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# --- Data Cleaning ---

# Convert dates to datetime format
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')

# Create a new duration feature (in minutes)
df['Response Duration (mins)'] = (df['Time to Resolution'] - df['First Response Time']).dt.total_seconds() / 60

# Drop columns with too many NaNs or not useful
df.drop(['Resolution', 'Time to Resolution'], axis=1, inplace=True)

# Fill missing values in datetime
df['First Response Time'] = df['First Response Time'].fillna(method='ffill')

# Clean the target column
df = df.dropna(subset=['Customer Satisfaction Rating'])
df['Customer Satisfaction Rating'] = pd.to_numeric(df['Customer Satisfaction Rating'], errors='coerce')
df = df.dropna(subset=['Customer Satisfaction Rating'])

# Fill missing values in categorical columns
cat_cols = ['Customer Name', 'Customer Email', 'Customer Gender',
            'Product Purchased', 'Ticket Type', 'Ticket Subject',
            'Ticket Description', 'Ticket Status', 'Ticket Priority',
            'Ticket Channel']

for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# Drop noisy or identifying columns
df.drop(['Ticket ID', 'Customer Name', 'Customer Email', 'Ticket Description'], axis=1, inplace=True)

# One-hot encode categorical columns
encode_cols = ['Customer Gender', 'Product Purchased', 'Ticket Type',
               'Ticket Subject', 'Ticket Status', 'Ticket Priority',
               'Ticket Channel']

df = pd.get_dummies(df, columns=encode_cols, drop_first=True)

# --- Final Cleaned Dataset ---
print("\nCleaned Shape:", df.shape)
print("\nCleaned Columns:\n", df.columns)
df.head()
sns.heatmap(df.corr(), annot=True)
df['Customer Satisfaction Rating'].value_counts()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('/content/customer_support_tickets.csv')

# Show data summary
def explore_data(df):
    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Basic Statistics ---")
    print(df.describe())

explore_data(df)

# 🔧 Step 1: Convert date/time columns
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')

# Convert to duration in hours
df['First Response Delay (hrs)'] = (df['First Response Time'] - df['Date of Purchase']).dt.total_seconds() / 3600
df['Resolution Time (hrs)'] = (df['Time to Resolution'] - df['Date of Purchase']).dt.total_seconds() / 3600

# Drop raw time columns (optional)
df.drop(['First Response Time', 'Time to Resolution'], axis=1, inplace=True)

# 🔧 Step 2: Handle missing values
# Numeric imputer
num_cols = ['Customer Satisfaction Rating', 'First Response Delay (hrs)', 'Resolution Time (hrs)']
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical imputer
cat_imputer = SimpleImputer(strategy='most_frequent')
for col in df.select_dtypes(include='object').columns:
    df[col] = cat_imputer.fit_transform(df[[col]]).ravel()

# 🔧 Step 3: Encode categorical variables
encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# 🔧 Step 4: Correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# 🔧 Step 5: Save cleaned dataset
df.to_csv('/content/cleaned_customer_support.csv', index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_customer_support.csv'")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load dataset (replace with your actual file path in Colab)
df = pd.read_csv('/content/customer_support_tickets.csv')

# Display basic info
def explore_data(df):
    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Basic Statistics ---")
    print(df.describe())

explore_data(df)

# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encoding categorical variables
encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Detect correlation
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# Save cleaned dataset
df.to_csv('/content/cleaned_dataset.csv', index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (replace 'your_dataset.csv' with actual file)
df = pd.read_csv('cleaned_customer_support.csv')

# Drop unnecessary columns
drop_cols = ['Ticket ID', 'Customer Name', 'Customer Email', 'Date of Purchase', 'Ticket Description']
df.drop(columns=drop_cols, inplace=True)

# Handle categorical features (Label Encoding)
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values
df.fillna(df.median(), inplace=True)

# Define features and target
X = df.drop(columns=['Customer Satisfaction Rating'])
y = df['Customer Satisfaction Rating']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

cleaned_file_path = "/content/cleaned_dataset.csv"  # Change path if needed
df.to_csv(cleaned_file_path, index=False)

from google.colab import files
files.download(cleaned_file_path)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/content/cleaned_dataset.csv')

# Optional columns to drop if they exist
columns_to_drop = ['Ticket ID', 'Customer Name', 'Customer Email', 'Date of Purchase']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Convert datetime strings to timestamps
for col in ['First Response Time', 'Time to Resolution']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').astype(np.int64) // 10**9

# Encode any remaining object (categorical) columns
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col].astype(str))

# Separate features and target
X = df.drop('Customer Satisfaction Rating', axis=1)
y = df['Customer Satisfaction Rating']

# Select best features
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Split and SMOTE
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)

# Show results
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
plt.figure(figsize=(10, 6))
feat_imp = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("🎯 Feature Importance")
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Drop irrelevant columns if they exist
df = df.drop(columns=[col for col in ['Ticket ID', 'Customer Name', 'Customer Email', 'Date of Purchase'] if col in df.columns], errors='ignore')

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and labels
X = df.drop('Customer Satisfaction Rating', axis=1)
y = df['Customer Satisfaction Rating'] - 1  # Subtract 1 to make classes start at 0

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_resampled, y_resampled)

# Predict
y_pred = xgb_model.predict(X_test)

# Metrics
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Print columns to verify
print("Available columns:", df.columns)

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Shift target column (Customer Satisfaction Rating) to start from 0
df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'] - 1

# Features and target
X = df.drop('Customer Satisfaction Rating', axis=1)
y = df['Customer Satisfaction Rating']

# SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}\n")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv('/content/cleaned_dataset.csv')  # Replace with your actual cleaned file path
print("Available columns:", df.columns)

# Fix target column if needed
if df['Customer Satisfaction Rating'].min() == 1:
    df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'] - 1

# Features and target
X = df.drop('Customer Satisfaction Rating', axis=1)
y = df['Customer Satisfaction Rating']

# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Full pipeline with preprocessing and SMOTE
X_processed = preprocessor.fit_transform(X)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')

# Hyperparameter tuning grid
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

# Run hyperparameter search
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
