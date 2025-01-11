import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from data_loader import load_data, preprocess_data
from sklearn.pipeline import Pipeline

# Load data and preprocess
X, y = load_data()
X_train, X_test, y_train, y_test, preprocessor, label = preprocess_data(X, y)

# Hyperparameter values for the two runs
run_1_params = {
    "max_depth": 2,
    "min_samples_split": 4,
    "min_samples_leaf": 6
}

run_2_params = {
    "max_depth": 5,
    "min_samples_split": 7,
    "min_samples_leaf": 9
}

# Create a custom experiment
experiment_name = "Drug_Prediction_Experiment"
experiment_id = mlflow.create_experiment(experiment_name)

# Start first MLflow run in the custom experiment
with mlflow.start_run(experiment_id=experiment_id, run_name="run_1"):
    # Log hyperparameters for run 1
    mlflow.log_param("max_depth", run_1_params["max_depth"])
    mlflow.log_param("min_samples_split", run_1_params["min_samples_split"])
    mlflow.log_param("min_samples_leaf", run_1_params["min_samples_leaf"])

    # Create and train model for run 1
    model = DecisionTreeClassifier(
        random_state=42, 
        max_depth=run_1_params["max_depth"], 
        min_samples_split=run_1_params["min_samples_split"], 
        min_samples_leaf=run_1_params["min_samples_leaf"]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model for run 1
    pipeline.fit(X_train, y_train)
    
    # Predict and calculate metrics for run 1
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics for run 1
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model for run 1
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Generate and save confusion matrix plot for run 1
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label.classes_, yticklabels=label.classes_)
    plt.title('Confusion Matrix (Run 1)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save the confusion matrix as an artifact
    cm_image_path = "confusion_matrix_run_1.png"
    plt.savefig(cm_image_path)
    mlflow.log_artifact(cm_image_path)
    
    print(f"Run 1 - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Start second MLflow run in the custom experiment
with mlflow.start_run(experiment_id=experiment_id, run_name="run_2"):
    # Log hyperparameters for run 2
    mlflow.log_param("max_depth", run_2_params["max_depth"])
    mlflow.log_param("min_samples_split", run_2_params["min_samples_split"])
    mlflow.log_param("min_samples_leaf", run_2_params["min_samples_leaf"])

    # Create and train model for run 2
    model = DecisionTreeClassifier(
        random_state=42, 
        max_depth=run_2_params["max_depth"], 
        min_samples_split=run_2_params["min_samples_split"], 
        min_samples_leaf=run_2_params["min_samples_leaf"]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model for run 2
    pipeline.fit(X_train, y_train)
    
    # Predict and calculate metrics for run 2
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics for run 2
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model for run 2
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Generate and save confusion matrix plot for run 2
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label.classes_, yticklabels=label.classes_)
    plt.title('Confusion Matrix (Run 2)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save the confusion matrix as an artifact
    cm_image_path = "confusion_matrix_run_2.png"
    plt.savefig(cm_image_path)
    mlflow.log_artifact(cm_image_path)
    
    print(f"Run 2 - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
