from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(X):
    # Get list of categorical columns (object/string type)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    # Get list of numerical columns (int/float type)
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Numeric pipeline: fill missing with 0, then scale values
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: fill missing with "Unknown", then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Apply numeric pipeline to numeric cols and categorical pipeline to categorical cols
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # Return the preprocessor and list of numerical columns
    return preprocessor, numerical_cols
