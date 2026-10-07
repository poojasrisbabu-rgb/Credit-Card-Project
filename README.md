Credit Card Default Prediction

## Project Overview

This project uses machine learning to predict whether a credit card
customer is likely to default on the next month's payment. The notebook
works with a credit-card dataset and treats `default payment next month`
as the target variable, renamed to `target`.

## Objectives

-   Explore and understand the credit-card dataset.
-   Inspect missing values, duplicate records, summary statistics, and
    correlations.
-   Prepare categorical and numerical features for machine learning.
-   Address class imbalance using SMOTE.
-   Select useful features and scale the selected data.
-   Train and compare classification models using evaluation metrics.

## Technologies Used

-   Python
-   Jupyter Notebook
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   imbalanced-learn (SMOTE)

## Dataset

The notebook loads a file named `creditcard.csv`:

``` python
data = pd.read_csv("creditcard.csv")
```

Place the dataset in the same directory as the notebook, or update the
path in the notebook. The dataset is expected to contain customer and
payment-related columns, including the default-payment target column.
The dataset itself is not included in this README.

## Workflow

### 1. Data Loading and Exploration

The notebook imports the required libraries and loads the dataset into a
Pandas DataFrame. It examines the first and last rows, dataset
dimensions, column names, data types, descriptive statistics, missing
values, duplicate rows, and feature correlations. Correlation heatmaps
are used for visual exploration.

### 2. Data Cleaning and Preparation

Several payment, bill, and payment-amount columns are renamed to use
month labels. The target column is renamed to `target`. Category values
in `SEX`, `MARRIAGE`, `EDUCATION`, and `target` are converted to
readable labels before encoding.

### 3. Outlier Handling

An IQR-based function is defined to cap numerical values below the lower
bound or above the upper bound. Boxplots are used to inspect numerical
distributions before and after this operation.

### 4. Encoding Categorical Features

The notebook uses `LabelEncoder` for the target and sex columns, and
`OneHotEncoder` for education and marriage categories.

### 5. Class Balancing

The notebook imports and applies SMOTE (Synthetic Minority Over-sampling
Technique) to balance the target classes. SMOTE should be applied only
to the training data in a leakage-safe workflow; review the current
notebook implementation before using its results for formal evaluation.

### 6. Feature Selection and Transformation

`SelectKBest` with the `f_classif` scoring function is configured to
select 25 features. The notebook also explores a Yeo-Johnson
`PowerTransformer` to transform numerical data.

### 7. Feature Scaling and Data Splitting

The selected features are standardized using `StandardScaler`. The
notebook then splits the scaled data into training and testing sets
using an 80:20 ratio and `random_state=40`.

### 8. Model Training and Evaluation

The notebook trains and compares the following classification
algorithms: - Logistic Regression - Decision Tree Classifier - Random
Forest Classifier - AdaBoost Classifier - Gradient Boosting Classifier

The evaluation metrics include: - **Accuracy:** proportion of
predictions that are correct. - **Precision:** proportion of predicted
positive cases that are truly positive. - **Recall:** proportion of
actual positive cases correctly identified. - **F1-score:** harmonic
mean of precision and recall.

The notebook stores rounded metric values in a results DataFrame for
model comparison.

## How to Run

1.  Install Python and Jupyter Notebook (or use Jupyter in Anaconda).

2.  Keep the notebook and `creditcard.csv` in the same folder.

3.  Install the required libraries:

    ``` bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
    ```

4.  Start Jupyter Notebook:

    ``` bash
    jupyter notebook
    ```

5.  Open the credit-card project notebook and run the cells in order.

## Expected Output

When run successfully, the notebook displays dataset summaries,
missing-value and duplicate checks, correlation heatmaps, boxplots,
selected-feature information, train/test shapes, classification metrics,
classification reports, and a model-comparison table. Exact metric
values depend on the dataset and the notebook execution.

## Important Notes

-   The notebook currently contains some implementation details that
    should be checked before relying on its scores. In particular, the
    output labels for Decision Tree metrics say "Linear Regression,"
    although the model used is a `DecisionTreeClassifier`.
-   Fit preprocessing steps such as feature selection and scaling on the
    training data only, then apply the fitted transformations to the
    test data. Apply SMOTE only to the training split to avoid data
    leakage.
-   Confirm that the target column is excluded from the feature matrix
    and that categorical encoding, transformed data, and row indices are
    aligned correctly before training.
-   For a fair comparison, use consistent evaluation settings across all
    models and consider a confusion matrix and ROC-AUC/PR-AUC where
    appropriate.

## Project Structure

``` text
credit-card-default-prediction/
├── credit card project notebook.ipynb
├── creditcard.csv
└── README.md
```

## Conclusion

This project demonstrates an end-to-end workflow for credit-card default
classification, from exploratory data analysis and preprocessing to
feature selection, scaling, model training, and comparison using
classification metrics. Model performance should be interpreted only
after checking the preprocessing workflow and preventing data leakage.
