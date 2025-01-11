import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import kagglehub

def load_data():
    # Load your data here (replace this with actual loading code)
    path = kagglehub.dataset_download("pablomgomez21/drugs-a-b-c-x-y-for-decision-trees")
    data = pd.read_csv(path+'/drug200.csv')

    # Separate features and target
    X = data.drop('Drug', axis=1)
    y = data['Drug']

    return X, y

def preprocess_data(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(exclude=['object']).columns
    string_features = X.select_dtypes(include=['object']).columns

    # Define transformers for categorical and numeric features
    string_transformer = Pipeline([
        ('one_hot_encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    numeric_transformer = Pipeline([
        ('power_transformer', PowerTransformer())
    ])

    # Combine the transformers in a ColumnTransformer
    preprocessor = ColumnTransformer([
        ('string_transformer', string_transformer, string_features),
        ('numeric_transformer', numeric_transformer, numeric_features)
    ])

    # Apply label encoding to the target variable
    label = LabelEncoder()
    y_train_encoded = label.fit_transform(y_train)
    y_test_encoded = label.transform(y_test)

    return X_train, X_test, y_train_encoded, y_test_encoded, preprocessor, label
