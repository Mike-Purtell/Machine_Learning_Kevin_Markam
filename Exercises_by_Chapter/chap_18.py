# /// script
# dependencies = [
#     "marimo",
#     "polars==1.43.2",
#     "scikit-learn==1.9.0",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", layout_file="layouts/chap_10.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    import os
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    # from sklearn.impute import SimpleImputer
    # from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    # from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    # from sklearn.compose import make_column_transformer
    # from sklearn.pipeline import make_pipeline

    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        accuracy_score,
        classification_report,
        confusion_matrix,
        np,
        os,
        pl,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 18:  Class imbalance
    - Finished XXX September YY, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - A small amount of imbalance (like in the Titanic dataset) tends not to matter.
    - As the amount of class imbalance increases, more specialized techniques need to be applied

    #### 18.1 Workflow recap
    A common issue when working on a classification problem is known as
    class imbalance, where classes are not equally represented in the
    dataset.

    Class imbalance is inherent to many domains. For example, fraudulent transactions in a credit card dataset would have a natural class imbalance since only a tiny fraction of transactions are likely to be fraudulent.

    Class imbalance can occur in binary classification problems where there are only two classes and in multiclass problems with more than two classes. In binary classification, the class that
    has more samples is called the “majority class”, and the class that has fewer samples is called the “minority class”.

    So why does class imbalance even matter? In brief, Machine Learning models often have a harder time predicting the minority class because there are fewer examples of this class to learn from. In
    other words, your model won’t be able to learn as much about the patterns in the minority class, and thus it may have a hard time differentiating between the classes.
    Keep in mind that some class imbalance is present in most datasets. In the Titanic dataset, for example, the majority class represents 62% of the samples and the minority class represents 38%. There’s even more imbalance in the census dataset, with a split of 76% to 24%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now read in the titanic dataset:
    """)
    return


@app.cell
def _(os, pl):
    cols = [
        'Parch', 'Fare', 'Embarked', 'Sex', 
        'Name', 'Age', 'Cabin', 'SibSp'
        ]

    # read the training dataset and use it to define X and y
    if not os.path.exists('assets/MLtrain.csv'):
        print("Downloading training dataset...")
        df = pl.read_csv('http://bit.ly/MLtrain')
        df.write_csv('assets/MLtrain.csv')
    else:
        print("Loading training dataset from local file...")
        df = pl.read_csv('assets/MLtrain.csv')

    df = pl.read_csv('assets/MLtrain.csv')
    X = df[cols]
    y = df.select('Survived')

    # read the new dataset and define X_new
    if not os.path.exists('assets/MLnewdata.csv'):
        print("Downloading new dataset...")
        df_new = pl.read_csv('http://bit.ly/MLnewdata')
        df_new.write_csv('assets/MLnewdata.csv')
    else:
        print("Loading new dataset from local file...")
        df_new = pl.read_csv('assets/MLnewdata.csv')

    X_new = df_new.select(cols)
    return (y,)


@app.cell
def _(pl, y):
    y.select(pl.col('Survived').value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    now load the census data
    """)
    return


@app.cell
def _(os, pl):
    if not os.path.exists('assets/census_dataset.csv'):
        print("Downloading census dataset...")
        census = pl.read_csv('http://bit.ly/censusdataset')
        census.write_csv('assets/census_dataset.csv')
    else:
        print("Loading census dataset from local file...")
        census = pl.read_csv('assets/census_dataset.csv')

    census_cols = ['workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
    census_X = census.select(pl.col(census_cols))
    census_y = census.select(pl.col('class'))
    return (census_y,)


@app.cell
def _(census_y, pl):
    census_y.select(pl.col('class').value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A small amount of class imbalance (like in the Titanic dataset) tends not to matter, and you should
    just use all of the usual techniques that you’ve learned in this book. But as the amount of class
    imbalance increases, more specialized techniques need to be applied to the problem, and those
    techniques are the focus of this chapter.

    #### 18.2 Preparing the mammography dataset
    This chapter will use a dataset of mammography scans designed to detect the presence of cancer.

    The dataset has 6 features as well as a target
    column called “class”. Each sample is an object of interest that was extracted from a scan, and each object was translated into these features using a computer vision algorithm.

    The class column has two possible values, -1 and 1. -1 means the object did not indicate the presence of cancer, and 1 means the object did indicate the presence of cancer. These labels were assigned by
    a human expert and thus represent the “ground truth” labels for the dataset.
    """)
    return


@app.cell
def _(os, pl):
    if not os.path.exists('assets/scanrecords.csv'):
        print("Downloading scan records dataset...")
        scan = pl.read_csv('http://bit.ly/scanrecords')
        scan.write_csv('assets/scanrecords.csv')
    else:
        print("Loading scan records dataset from local file...")
        scan = pl.read_csv('assets/scanrecords.csv')

    scan.head()
    return (scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -1 (meaning “non-cancerous”) is the majority class, and 1 (meaning“cancerous”) is the minority class. This is severe class imbalance because the minority class makes up only 2% of the dataset.
    """)
    return


@app.cell
def _(pl, scan):
    scan.select(pl.col('class').value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although sklearn doesn’t require it, we’re going to map the strings in the class column to the integers 0 and 1, in which 0 means “non-cancerous” and 1 means “cancerous”.
    """)
    return


@app.cell
def _(pl, scan):
    scan_1 = (
        scan
        .with_columns(
            pl.col('class')
            .str.strip_chars("'")  # Remove surrounding single quotes from the class column
            .str.to_integer(dtype=pl.Int8) # , strict=False)
        )
    )
    scan_1
    return (scan_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that the dataset is ready, define two objects, scan_X and scan_y, to represent
    the feature matrix and the target.
    """)
    return


@app.cell
def _(scan_1):
    scan_X = scan_1.drop('class')
    scan_y = scan_1.select('class')
    return scan_X, scan_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 18.3 Evaluating a model with a train/test split
    We have been using cross-validation as the model evaluation procedure, but in this chapter, we will use train/test split.
    Here is a quick overview of how we use train/test split for model evaluation:
    - Split the dataset into training and testing sets.
    - Fit the model on the training set.
    - Use the fitted model to make predictions on the testing set and evaluate those predictions.

    K-fold cross-validation is really just a train/test split run “K” times in a
    systematic way and then averaging the results. Cross-validation is superior because it outputs a lower
    variance estimate of model performance. Will look at use-cases where test/train split is a better choice.

    Let’s start by splitting the dataset into training and testing sets. We’ll use a split of 75%
    for training and 25% for testing. Will set a random_state for reproducibility
    """)
    return


@app.cell
def _(scan_X, scan_y, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        scan_X, 
        scan_y,
        test_size=0.25,
        random_state=1,
        stratify=scan_y
        )
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    With stratified sampling the class proportions will be approximately equal in the training
    and testing sets. This is important when there is severe class imbalance, otherwise you
    might have insufficient examples of the minority class in either the training or testing set.

    Now we’ll fit our logistic regression model on the training set, and use the fitted model to make
    predictions on the testing set. The predict method outputs class predictions of 0 or 1, which we’ll
    store as y_pred.
    As an aside, we’re not using a Pipeline here because the features are entirely numeric and don’t
    need any preprocessing.
    """)
    return


@app.cell
def _(LogisticRegression, X_test, X_train, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train.to_series())
    y_pred = logreg.predict(X_test)
    y_pred
    return logreg, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, let’s evaluate the model’s predictions. We’ve used accuracy as our evaluation metric throughout
    the book, so we’ll try that here. We have to import the accuracy_score function from the metrics
    module, and then we pass it the true values followed by the predicted values.
    It outputs an accuracy of 98%, which sounds great. But as you might recall, about 98% of the target
    values are 0, so an uninformed model could achieve 98% accuracy by always predicting class 0.
    """)
    return


@app.cell
def _(accuracy_score, y_pred, y_test):
    accuracy_score(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the accuracy value isn’t telling us much about how the model is actually performing, it’s
    not a particularly useful evaluation metric in this case, or in any case in which there’s severe class
    imbalance.

    #### 18.4: Exploring the results with a confusion matrix
    Before we choose an evaluation metric other than accuracy, let’s first use a confusion matrix to get a
    better understanding of our results.
    We’ll import it from the metrics module, and then pass it the true values followed by the predicted
    values. (Note that the order is important, because your confusion matrix will be incorrect if you pass
    the predicted values first instead of second.)
    """)
    return


@app.cell
def _(confusion_matrix, y_pred, y_test):
    confusion_matrix(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It can be hard to remember what each of the four boxes represents, so a nice alternative is to use the
    plot_confusion_matrix function.
    We pass it a fitted model, X_test, and y_test. It makes predictions for X_test,
    and outputs a confusion matrix by comparing the results to y_test. Thus it duplicates some of the
    work we’ve already done, though it ends up with the same result.
    The confusion matrix is now labeled, with the true labels as the rows and the predicted labels as the
    columns. You can see that:
    - In 2721 cases, the model predicted 0 and that was correct. These are called True Negatives.
    - In 28 cases, it predicted 1 and that was correct. These are True Positives.
    - In 37 cases, it predicted 0 and that was incorrect. These are False Negatives.
    - In 10 cases, it predicted 1 and that was incorrect. These are False Positives.

    'liblinear' was the old default for small datasets in 0.23.2, while modern sklearn defaults to 'lbfgs'. So the book example has 37 false negatives while I have 38, and the book has 28 true postive and I get 27.
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, X_test, logreg, y_test):
    disp = ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, cmap='Blues')
    disp.figure_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This should illustrate how the confusion matrix helps us to understand the performance of our
    classifier much better than the accuracy score.
    In particular, it highlights a troubling issue, which we can see by examining the bottom row. There
    were 65 samples in which cancer was present in the testing set, and it was detected in only 28 of
    those cases. Given the context, in which the model is attempting to detect the presence of cancer,
    you could imagine that it would be highly problematic to miss 37 out of 65 cases. This is an issue
    that we will address by the end of this chapter.

    18.5 Calculating rates from a confusion matrix

    Before discussing a solution to the problem from the previous lesson, let’s first calculate a few rates
    from the confusion matrix to help us quantify what we want to improve and what tradeoffs we’ll be
    making. Here’s another look at the confusion matrix.
    """)
    return


@app.cell
def _(confusion_matrix, y_pred, y_test):
    confusion_matrix(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    True Positive Rate answers the question: When cancer is present, how often is that correctly predicted?
    We divide the True Positives by the entire bottom row, and we get 41.5%. True Positive Rate is also
    known as recall.
    """)
    return


@app.cell
def _():
    print(f'{100* 27 / (38 + 27):.3f} %')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    True Negative Rate answers the question: When cancer is not present, how often is that correctly
    predicted? We divide the True Negatives by the entire top row, and we get 99.6%.
    """)
    return


@app.cell
def _():
    print(f'{100 * 2721 / (2721 + 10):.3f} %')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    False Positive Rate answers the question: When cancer is not present, how often is that incorrectly
    predicted? We divide the False Positives by the entire top row, and we get 0.4%. As you might have
    figured out, the False Positive Rate is 1 minus the True Negative Rate.
    """)
    return


@app.cell
def _():
    print(f'{100 * 10 / (2721 + 10):.3f} %')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another way to see these results without doing the calculations yourself is with the classification
    report. After importing it and passing it the true and predicted values, we can see the following:
    - The True Positive Rate is the recall for class 1, which is 43%.
    - The True Negative Rate is the recall for class 0, which has been rounded to 100%.
    - The False Positive Rate is not directly shown, but it’s 1 minus the True Negative Rate, thus it
    rounds to 0%.
    - The overall accuracy is also shown, which is 98%, though that’s not our focus here.
    """)
    return


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the True Positive Rate of 42% seems quite problematic, you might think that the solution is
    just to work on maximizing that metric alone. However, the danger of only focusing on True Positive
    Rate is that you might end up with a confusion matrix similar to this.
    """)
    return


@app.cell
def _(confusion_matrix, np, y_pred, y_test):
    confusion_matrix(y_test, np.ones_like(y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The True Positive Rate in this case is 100%, but that’s because we’re always predicting the positive
    class. As such, the False Positive Rate would also be 100%. I think you would agree that this is not a useful solution.

    #### 18.6 Using AUC as the evaluation metric

    # Break on page 267. Completed 18.5, ready to start 18.6
    """)
    return


@app.cell
def _():
    #
    return


if __name__ == "__main__":
    app.run()
