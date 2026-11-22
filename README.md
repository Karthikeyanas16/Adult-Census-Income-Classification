# Adult Census Income Classification

## M.Tech (AIML/DSE) – Machine Learning Assignment 2

**Institution:** BITS Pilani  
**Programme:** M.Tech (Artificial Intelligence and Machine Learning / Data Science and Engineering)  
**Course:** Machine Learning  
**Dataset:** UCI Adult Census Income Dataset  
**Problem Type:** Binary Classification

---

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple supervised machine learning classification algorithms to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related attributes.

The project uses the UCI Adult Census Income dataset and implements five classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The performance of each model is evaluated using six metrics: Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

The primary objective is to compare the classification performance of these algorithms and identify the most suitable model for the selected dataset.

## 2. Dataset Description

### Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/2/adult

The Adult Census Income dataset contains demographic and employment-related information collected from census data.

### Dataset Characteristics

| Attribute | Description |
|---|---|
| Dataset name | Adult Census Income |
| Source | UCI Machine Learning Repository |
| Original records | 48,842 |
| Records after duplicate removal | 48,794 |
| Input features | 14 |
| Target variable | income |
| Classification type | Binary |
| Training records | 39,035 |
| Testing records | 9,759 |

### Target Variable

The target variable is `income`.

- Class 0: <=50K
- Class 1: >50K

### Class Distribution

After data cleaning, the class distribution was:

| Income Category | Percentage |
|---|---:|
| <=50K | 76.06% |
| >50K | 23.94% |

The dataset has an imbalance between the two income categories. Stratified sampling was used to preserve the class distribution during train-test splitting.

### Dataset Features

The dataset contains demographic, educational and employment-related attributes, including:

- Age
- Workclass
- Final weight (fnlwgt)
- Education
- Education number
- Marital status
- Occupation
- Relationship
- Race
- Sex
- Capital gain
- Capital loss
- Hours per week
- Native country

## 3. Data Preprocessing

The following preprocessing activities were performed:

1. Loaded the Adult Census Income dataset using the UCI ML Repository.
2. Identified missing values in categorical attributes.
3. Handled missing values through the preprocessing pipeline.
4. Checked and removed duplicate records.
5. Normalized the target variable by removing inconsistent trailing punctuation.
6. Converted the income target into binary numerical labels.
7. Applied One-Hot Encoding to categorical features.
8. Applied Standard Scaling to numerical features.
9. Used a ColumnTransformer to combine preprocessing operations.
10. Split the dataset into training and testing sets using an 80:20 ratio.
11. Used random_state=42 and stratified sampling.
12. Fitted the preprocessing pipeline only on training data and transformed the testing data separately.

The preprocessing pipeline generated 108 transformed features.

## 4. GitHub Repository Link

**Project Repository:**

https://github.com/Karthikeyanas16/Adult-Census-Income-Classification

The repository contains the Jupyter Notebook, testing dataset and project dependency information.

### Repository Structure

```text
Adult-Census-Income-Classification/
│
├── README.md
├── requirements.txt
├── test_data.csv
├── model_performance.csv
│
└── model/
    └── ML_Assignment_2_Adult_Income.ipynb
```

The notebook contains the implementation and execution of all five classification models.

## 5. Machine Learning Models Implemented

### 5.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of an observation belonging to a particular class.

It was used to predict whether an individual's annual income exceeds $50,000.

### 5.2 Decision Tree Classifier

Decision Tree is a supervised learning algorithm that makes predictions using a sequence of feature-based decision rules.

It divides the dataset into smaller subsets based on feature conditions and predicts the income category.

### 5.3 K-Nearest Neighbors (KNN)

KNN is a supervised classification algorithm that predicts the class of an observation based on the majority class among its nearest neighboring observations.

The model was implemented using five nearest neighbors.

### 5.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It assumes that numerical features follow a Gaussian distribution within each class and that features are conditionally independent.

### 5.5 Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance.

The implementation used 100 decision trees and combined their predictions through majority voting.

## 6. Model Evaluation and Performance Comparison

All five models were evaluated using the same testing dataset.

The following six evaluation metrics were calculated:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### 6.1 Consolidated Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8556 | 0.9101 | 0.7403 | 0.6113 | 0.6696 | 0.5827 |
| Decision Tree | 0.8165 | 0.7561 | 0.6114 | 0.6404 | 0.6255 | 0.5043 |
| KNN | 0.8393 | 0.8677 | 0.6829 | 0.6139 | 0.6465 | 0.5442 |
| Gaussian Naive Bayes | 0.6024 | 0.8195 | 0.3692 | 0.9324 | 0.5289 | 0.3761 |
| Random Forest | 0.8540 | 0.9062 | 0.7238 | 0.6306 | 0.6740 | 0.5828 |

### 6.2 Graphical Representation

Graphical comparisons were generated to evaluate the performance of all five classification algorithms.

The visualizations include:

1. Individual metric comparison across all five models.
2. Accuracy comparison.
3. AUC comparison.
4. Precision comparison.
5. Recall comparison.
6. F1 Score comparison.
7. MCC comparison.
8. Confusion matrices for all five classification algorithms.

These visualizations provide an intuitive understanding

### Confusion Matrix Visualization

![Confusion Matrices](screenshots/confusion_matrices.png)

### Individual Evaluation Metric Comparison

![Metric Comparison](screenshots/metric_comparison.png)

## 7. Model Performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved the highest accuracy (85.56%), AUC (0.9101) and precision (0.7403), demonstrating strong overall classification performance. |
| Decision Tree | Achieved 81.65% accuracy. Its performance was comparatively lower across most evaluation metrics. |
| K-Nearest Neighbors | Achieved 83.93% accuracy and 0.8677 AUC, demonstrating reasonable classification performance. |
| Gaussian Naive Bayes | Achieved the highest recall (0.9324), but its accuracy (60.24%) and precision (0.3692) were relatively low. |
| Random Forest | Achieved the highest F1 Score (0.6740) and MCC (0.5828), providing a good balance between precision and recall. |

## 8. Overall Winner

Random Forest is selected as the overall winner based on its highest F1 Score of 0.6740 and MCC of 0.5828.

Logistic Regression also demonstrated excellent performance, achieving the highest accuracy of 85.56% and AUC of 0.9101.

The final model selection depends on the classification objective and the importance of balancing precision and recall.

## 9. Execution Instructions

### Prerequisites

- Python 3.12
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install -r requirements.txt
```

### Clone the repository


```bash
git clone https://github.com/Karthikeyanas16/Adult-Census-Income-Classification.git
```

### Execute the assignment

1. Open the project in Jupyter Notebook.
2. Navigate to the model directory.
3. Open `ML_Assignment_2_Adult_Income.ipynb`.
4. Execute the notebook cells sequentially.
5. Review the model evaluation results and graphical visualizations.

The notebook contains the complete implementation of dataset loading, preprocessing, model training, evaluation and comparison.


