
# Airline Passenger Satisfaction Prediction

![AI](https://img.shields.io/badge/Artificial%20Intelligence-Enabled-0072C6)
![Data Science](https://img.shields.io/badge/Data%20Science-Enabled-9B59B6)

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Contributors](https://img.shields.io/badge/contributors-4-orange)

![Pandas](https://img.shields.io/badge/Pandas-1.2.4-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/Numpy-1.21.0-blue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.3-orange?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11.2-ff69b4?logo=seaborn)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24.2-yellow?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5.0-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.4.3-violet?logo=keras)
![XGBoost](https://img.shields.io/badge/XGBoost-1.4.2-green?logo=xgboost)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Exploration & Analysis](#data-exploration--analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [License](#license)

---

## Project Overview

The **Airline Passenger Satisfaction Prediction** project aims to analyze and predict passenger satisfaction levels based on various features related to flight experiences. By leveraging machine learning and deep learning techniques, the project identifies key factors influencing passenger satisfaction and builds predictive models to classify satisfaction levels effectively.

## Dataset

The project utilizes a dataset comprising airline passenger information, including demographics, flight details, and satisfaction ratings. The dataset is split into training and testing sets:

- **Train Dataset:** `Dataset/train.csv`
- **Test Dataset:** `Dataset/test.csv`

## Project Structure

```
├── Dataset
│   ├── train.csv
│   └── test.csv
├── Models
│   ├── cv_estimator.sav
│   ├── cv_selector.sav
│   ├── log_r.sav
│   ├── knn.sav
│   ├── rf_rs.sav
│   ├── dtc.sav
│   ├── svc.sav
│   └── mlp.h5
├── README.md
└── Airline_Passenger_Satisfaction.ipynb
```

- **Dataset:** Contains the training and testing data.
- **Models:** Stores serialized machine learning models.
- **README.md:** Project documentation.
- **Airline_Passenger_Satisfaction.ipynb:** Jupyter Notebook with the complete analysis and modeling workflow.

## Installation

To replicate the project environment, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/airline-passenger-satisfaction.git
   cd airline-passenger-satisfaction
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   ```

   *If a `requirements.txt` file is not provided, install the necessary libraries manually:*

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras xgboost pickle5
   ```

## Usage

1. **Navigate to the Project Directory**
   Ensure you are in the root directory of the project.

2. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook Airline_Passenger_Satisfaction.ipynb
   ```

3. **Follow the Notebook Steps**
   The notebook is organized into the following sections:
   - Loading Libraries
   - Loading Dataset
   - Data Exploration
   - Data Analysis
   - Data Cleaning
   - Data Preprocessing
   - Feature Selection
   - Model Training and Evaluation
   - Results Visualization

## Data Exploration & Analysis

- **Data Overview:** Displaying the first few rows, shape, columns, and statistical summaries.
- **Visualization:** Creating count plots, histograms, and heatmaps to understand data distribution and correlations.

## Data Preprocessing

- **Handling Missing Values:** Identifying and imputing missing data.
- **Encoding Categorical Variables:** Applying label encoding and one-hot encoding to transform categorical features.
- **Feature Scaling:** Scaling numerical features using Min-Max Scaler.
- **Data Splitting:** Dividing the dataset into training and testing sets (70% training, 30% testing).

## Feature Selection

- **Recursive Feature Elimination with Cross-Validation (RFECV):** Selecting the most relevant features using a Random Forest classifier.
- **Feature Importance:** Identifying and visualizing the top features influencing passenger satisfaction.

## Model Training and Evaluation

Implemented and evaluated multiple models using different strategies:

1. **Logistic Regression with GridSearchCV**
2. **K-Nearest Neighbors with GridSearchCV**
3. **Random Forest with RandomizedSearchCV**
4. **Decision Tree Classifier with GridSearchCV**
5. **Support Vector Machine with GridSearchCV**
6. **Bagging with Random Forest**
7. **AdaBoost Classifier**
8. **Multilayer Perceptron (MLP)**
9. **XGBoost Classifier**

**Evaluation Metrics:**
- Accuracy
- AUC-ROC
- Precision, Recall, F1-Score
- Confusion Matrix

## Results

- **Model Performance:** A comparative bar chart showcasing the accuracy of each model.
- **Feature Importance:** Detailed analysis of feature importance from the XGBoost model, highlighting the top factors affecting passenger satisfaction.

**Key Findings:**
- XGBoost achieved the highest accuracy among all models.
- Top features influencing satisfaction include `Online boarding`, `Inflight wifi service`, `Type of Travel`, `Flight_class_Business`, and `Flight_class_Eco`.



## License

This project is licensed under the [MIT License](LICENSE).
