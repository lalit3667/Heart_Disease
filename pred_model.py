import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load your dataset (replace 'heart.csv' with your actual dataset path)
data = pd.read_csv("heart.csv")  # Ensure you have the heart disease dataset

# Define feature columns and target variable
feature_cols = [
    'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
    'max_hr', 'exercise_angina', 'oldpeak', 'st_slope'
]
target_col = 'HeartDisease'  # Replace with your actual target column name

# Preprocess the dataset
X = data[feature_cols]
y = data[target_col]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric and categorical features
numeric_features = ['age', 'resting_bp', 'cholesterol', 'max_hr', 'oldpeak']
categorical_features = ['sex', 'chest_pain_type', 'exercise_angina', 'st_slope']

# Create a preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the model
pipeline.fit(X_train, y_train)

# Save the model and the scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(pipeline, model_file)

# Save the scaler (if needed, but it's integrated into the pipeline)
# with open("scaler.pkl", "wb") as scaler_file:
#     pickle.dump(scaler, scaler_file)

# Print accuracy on the test set (optional)
print("Model training complete. Test accuracy:", pipeline.score(X_test, y_test))
