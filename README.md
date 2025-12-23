# Loan Approval Prediction Using Logistic Regression

## Overview
This project performs exploratory data analysis (EDA) and binary classification on a loan approval dataset using Logistic Regression. The goal is to analyze how demographic, financial, and property-related features influence loan approval decisions and to build a predictive model based on these features.

## Dataset
The dataset is loaded from a CSV file containing applicant information and loan approval status. Key features include Gender, Marital Status, Education, Dependents, Self-Employment status, Credit History, Loan Amount, and Property Area. The target variable is Loan_Status, indicating whether a loan was approved or not.

## Data Cleaning and Preprocessing
Missing values in LoanAmount are filled using the mean, while missing values in Credit_History are filled using the median. Rows with remaining missing values are dropped. Categorical variables are converted into numerical format using manual mapping to make them suitable for machine learning models.

## Exploratory Data Analysis
Multiple count plots are generated to visualize the relationship between Loan_Status and categorical features such as Gender, Marital Status, Education, Self-Employment, Property Area, and Dependents. These visualizations help in understanding feature importance and approval trends.

## Feature Encoding
Categorical features are encoded numerically:
- Gender: Male → 1, Female → 0  
- Married: Yes → 1, No → 0  
- Education: Graduate → 1, Non-Graduate → 0  
- Dependents: 0, 1, 2, 3+ → 0, 1, 2, 3  
- Self Employed: Yes → 1, No → 0  
- Property Area: Urban → 1, Rural → 0  

Loan_Status is mapped to binary values for classification.

## Model Training
The dataset is split into training and testing sets using a 70-30 split. A Logistic Regression model from Scikit-learn is trained on the training data to learn patterns associated with loan approval.

## Model Evaluation
Predictions are generated on the test set. Model performance is evaluated using accuracy score. Predicted values and actual test labels are printed to allow comparison between model output and ground truth.

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to Run
Clone the repository, install required dependencies, update the dataset path in the script, and run the Python file to perform EDA and train the Logistic Regression model.

## Applications
Loan approval systems, financial risk assessment, credit scoring, decision support systems, and machine learning education.

## Future Improvements
Potential enhancements include feature scaling, handling class imbalance, hyperparameter tuning, confusion matrix and ROC-AUC analysis, and comparison with other classification algorithms.

## Author
Soban Saeed
GitHub: https://github.com/shizu75

## License
MIT
