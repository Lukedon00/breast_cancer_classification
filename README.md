# Breast Cancer Classification with Multinomial Naive Bayes

This project demonstrates the classification of the Breast Cancer dataset using the Multinomial Naive Bayes algorithm. The breast cancer dataset is loaded from scikit-learn's built-in datasets and used to train and evaluate a classification model.

## Project Overview

- **Dataset**: Breast Cancer dataset from `sklearn.datasets`
- **Algorithm**: Multinomial Naive Bayes
- **Objective**: Classify whether a tumor is benign or malignant based on features like mean radius, mean texture, etc.
- **Tools Used**: Python, Pandas, scikit-learn

## Steps:
1. Load the Breast Cancer dataset.
2. Preprocess the data and create features and target variables.
3. Split the data into training and testing sets.
4. Train the model using Multinomial Naive Bayes.
5. Evaluate the model using accuracy score, confusion matrix, and classification report.

## Results:
- **Accuracy**: 93.57%
- **Confusion Matrix**:
    - [[x, y], [z, w]]
- **Classification Report**:  
    ```
    precision    recall  f1-score   support
    0       x.xx      x.xx      x.xx      xxx
    1       y.yy      y.yy      y.yy      yyy
    ```

## How to Run

### Prerequisites
- Python 3.x
- Required Libraries (see `requirements.txt`)

### Steps to Run:
1. Clone the repository.
    ```bash
    git clone https://github.com/your-username/breast_cancer_classification.git
    ```
2. Navigate to the project directory.
    ```bash
    cd breast_cancer_classification
    ```
3. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or the Python script:
    - Jupyter Notebook: Open the `breast_cancer_classification.ipynb` file and run the cells.
    - Python Script: Execute the Python file in your terminal.
    ```bash
    python breast_cancer_classification.py
    ```

## License
This project is licensed under the MIT License.