This project builds a Loan Approval Prediction pipeline using Python, scikit-learn, and imbalanced-learn.

Key steps:

Data Loading & Cleaning – Reads the loan dataset, fixes column names, cleans the loan_status field, and removes invalid entries to avoid NaN target values.

Feature Engineering – Separates numeric and categorical columns, removes IDs, and prepares them for modeling.

Preprocessing Pipelines – Uses SimpleImputer for missing values, StandardScaler for numeric scaling, and OneHotEncoder for categorical encoding.

Handling Class Imbalance – Integrates SMOTE to oversample the minority class.

Model Training – Trains both Logistic Regression and Decision Tree models inside pipelines.

Evaluation – Calculates accuracy, precision, recall, F1-score, confusion matrix, and plots ROC curves.

Hyperparameter Tuning – Uses GridSearchCV to optimize Decision Tree parameters for better performance.

The result is a robust, end-to-end machine learning workflow that can handle messy data, imbalanced classes, and provide interpretable performance metrics.
