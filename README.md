# Drug Prediction Model with MLflow

This repository demonstrates how to build and evaluate a machine learning model to predict drugs based on certain features using a Decision Tree classifier. The project uses **MLflow** to manage and log experiments, model metrics, and model artifacts. Additionally, the code is structured in two parts: data preprocessing (separated into a `data_loader.py` file) and the model training and evaluation (in the `main_script.py` file).

## Requirements

To run this project, you'll need the following Python libraries:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `mlflow`

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Directory Structure

- **data_loader.py**: This file contains functions for loading and preprocessing the data.
- **main_script.py**: This is where the machine learning model is trained and evaluated. It logs the experiments with MLflow and produces metrics and confusion matrix plots.
- **requirements.txt**: A text file listing all the required libraries for the project.

---

## Functions and Code Explanation

### **data_loader.py**

This file contains two functions for loading and preprocessing the data:

1. **load_data()**:
   - This function loads the dataset (a CSV file) and separates it into feature columns `X` and the target column `y`.
   - It returns the feature matrix `X` and target vector `y`.

2. **preprocess_data()**:
   - This function splits the data into training and test sets (80% training and 20% testing).
   - It separates numeric and string-based features from the dataset.
   - It applies a **PowerTransformer** to numeric features to scale them and uses **OneHotEncoder** to transform categorical features.
   - It also applies **LabelEncoder** to the target variable (`Drug`) for binary classification.

### **main_script.py**

This script is where the machine learning model is built, trained, and evaluated. It also logs the experiments with MLflow. Below is a detailed breakdown:

1. **Data Loading and Preprocessing**:
   - The data is loaded from `data_loader.py` using the `load_data()` function and preprocessed with the `preprocess_data()` function.

2. **Hyperparameters Setup**:
   - The script defines two sets of hyperparameters for two different model runs. These hyperparameters are for the `DecisionTreeClassifier`, which controls the complexity of the decision tree.
   
   ```python
   run_1_params = {
       "max_depth": 4,
       "min_samples_split": 10,
       "min_samples_leaf": 5
   }
   
   run_2_params = {
       "max_depth": 5,
       "min_samples_split": 15,
       "min_samples_leaf": 10
   }
   ```

3. **MLflow Experiment**:
   - A new custom experiment is created using `mlflow.create_experiment()`.
   - The model training is done within the context of two separate runs, where each run logs different hyperparameters and evaluates the model.

4. **Model Training**:
   - A `DecisionTreeClassifier` is used to train the model on the training data.
   - The model is evaluated using various metrics such as accuracy, precision, recall, and F1-score. The metrics are logged to MLflow.
   - A confusion matrix is also generated and saved as an artifact in MLflow.

5. **Results Logging**:
   - The script logs the metrics (accuracy, precision, recall, F1 score) and model parameters (e.g., max depth, min samples split) using `mlflow.log_metric()` and `mlflow.log_param()`.
   - It saves the trained model using `mlflow.sklearn.log_model()`.

6. **Confusion Matrix**:
   - The confusion matrix is visualized using **Seaborn** and saved as an image artifact in MLflow. It is also displayed for each run.

---

## Images

Below are some relevant images that explain the concepts of decision trees and confusion matrices:

 **Run1 vs Run2**:
   MLflow helps you keep track of your experiments, including parameters, metrics, and artifacts like models and plots.
   ![Confusion Matrix](https://github.com/AdanALalawni/MLFLOW_TASK6/blob/main/assest/run1%20vs%20run2.png)

 **Run2 confusion matrix**:

   ![MLflow UI](https://github.com/AdanALalawni/MLFLOW_TASK6/blob/main/assest/run2.png)



