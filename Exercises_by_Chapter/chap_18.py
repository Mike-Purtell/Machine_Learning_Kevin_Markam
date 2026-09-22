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
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import RocCurveDisplay
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        RocCurveDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
        np,
        os,
        pl,
        roc_auc_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 18:  Class imbalance
    - Finished Tuesday September 22, 2026

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
    - Accuracy is misleading with class imbalance
    - In this mammography dataset, a model can get ~98% accuracy by predicting the majority class all the time. That’s why confusion matrices, recall, precision, and AUC are more informative than accuracy here. AUC measures class separation, not just thresholded predictions

    - A high AUC means the model ranks positive samples above negative samples well.
    But a default threshold of 0.5 may still produce poor recall if the positive class is rare. Cost-sensitive learning and threshold tuning help balance error tradeoffs

    - class_weight='balanced' makes the model pay more attention to the minority class.
    Then changing the decision threshold lets you trade off between false positives and false negatives based on what matters most in the real problem.

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
    With the problem now understood, we can focus on the solution. The first step is choosing a more appropriate evaluation metric for tuning the model. We’ll use AUC (Area Under the ROC Curve), which measures how well the model separates the classes by assigning higher predicted probabilities to class‑1 samples than to class‑0 samples.

    To illustrate this, we’ll have the fitted model output predicted probabilities—rather than class labels—using predict_proba. We’ll store these values in y_score and print them. The first number in the array shows that the model assigns a 0.15% probability to the first test sample belonging to the positive class; the second number shows a 0.19% probability for the second sample.
    """)
    return


@app.cell
def _(X_test, logreg):
    y_score = logreg.predict_proba(X_test)[:, 1]
    y_score
    return (y_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    AUC is a measure of how well the model succeeds at assigning higher probabilities to class 1 samples
    than class 0 samples. In other words, AUC doesn’t care about the actual predicted probability values,
    rather it cares only about the rank ordering of the values. As such, it can be used with any classifier
    that outputs predicted probabilities, regardless of whether those probabilities are well-calibrated.

    Area Under the ROC Curve (AUC):
    - Measures how effectively the model separates the two classes
    - Rewards models that assign higher probabilities to class‑1 samples than to class‑0 samples
    - Works with any classifier that outputs predicted probabilities

    Let’s go ahead and calculate the AUC for our model. We’ll import roc_auc_score and pass it the true labels along with the predicted probabilities (not the class predictions). The result is an AUC of 0.93, which indicates strong class‑separation performance.
    """)
    return


@app.cell
def _(roc_auc_score, y_score, y_test):
    roc_auc_score(y_test, y_score)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A perfect model would achieve an AUC of 1.0, while a completely uninformed model would score 0.5. More formally, an AUC of 0.93 means that if you randomly select one class‑1 sample and one class‑0 sample, there’s a 93% chance the model will assign a higher predicted probability to the class‑1 sample

    A natural follow‑up question is: If the AUC is high (0.93), why is the True Positive Rate so low (0.43)?

    The answer comes down to the decision threshold. The threshold is the predicted‑probability cutoff (between 0 and 1) above which the model assigns the positive class. By default, this threshold is 0.5. So a predicted probability of 0.7 results in a class‑1 prediction, while a probability of 0.2 results in a class‑0 prediction.

    From the right column of the confusion matrix, we can see that the model predicted class 1 for only 38 samples in the test set.
    """)
    return


@app.cell
def _(confusion_matrix, y_pred, y_test):
    confusion_matrix(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Equivalently, the model predicted a probability greater than 0.5 for only 38 samples. (This works by creating a boolean array and then counting the number of True values.)
    """)
    return


@app.cell
def _(y_score):
    sum(y_score > 0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With that in mind, we can infer two things from the combination of a high AUC and a low True Positive Rate:

    - The high AUC tells us the model is doing a good job separating the classes.
    - The low True Positive Rate tells us the default decision threshold isn’t serving us well.

    To understand this second issue more clearly, we’ll plot the ROC curve using plot_roc_curve. Its API is similar to plot_confusion_matrix. You pass the fitted model, X_test, and y_test.

    So what are we looking at? The ROC curve plots the True Positive Rate (y‑axis) against the False Positive Rate (x‑axis) across all possible decision thresholds.

    For example, one point on the curve corresponds to the default threshold of 0.5, where the True Positive Rate is 43% and the False Positive Rate is nearly zero.

    Changing the decision threshold simply moves you to a different point on the curve. You could, for instance, move to a point with a True Positive Rate around 90% and a False Positive Rate around 10% just by adjusting the threshold. The threshold values themselves aren’t shown on the plot—only the resulting (TPR, FPR) pairs.
    """)
    return


@app.cell
def _(RocCurveDisplay, X_test, logreg, y_test):
    disp_1 = RocCurveDisplay.from_estimator(logreg, X_test, y_test)
    disp_1.figure_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interpreting the ROC curve:

    - Plots the True Positive Rate (TPR) against the False Positive Rate (FPR) across all possible decision thresholds
    - Changing the decision threshold moves you to a different point on the curve
      - The threshold values themselves are not shown on the plot
    - The AUC represents the proportion of the plot that lies beneath the ROC curve

    Given what we’ve learned so far, it’s helpful to map out our next steps. In lesson 18.7, we’ll work on improving the model’s AUC. Then, in lesson 18.8, we’ll explore alternative decision thresholds so we can balance the True Positive Rate and False Positive Rate in a way that better fits our goals.

    #### 18.7 Cost-senstive learning
    Now that we know our next goal is to improve the model’s AUC, how do we actually do that? The good news is that we can use any of the techniques covered in this book, including hyperparameter tuning, feature selection, trying non‑linear models, and others. All of these approaches have the potential to boost AUC.

    In this lesson we’ll focus on one technique we haven’t discussed yet that’s especially useful when dealing with class imbalance: cost‑sensitive learning.

    The key idea behind cost‑sensitive learning is that not all prediction errors carry the same “cost.” That cost might represent real monetary impact or the real‑world consequences of different types of mistakes.

    Under severe class imbalance, False Negatives—cases where positive samples are incorrectly labeled as negative—typically have a higher cost than False Positives, where negative samples are labeled as positive. This makes sense: positive samples are rare, and we care more about finding them than about occasionally misclassifying a negative sample. Put simply, we would rather incur a False Positive than a False Negative.

    So how does cost‑sensitive learning actually work? In scikit‑learn, it’s implemented through the class_weight parameter available in several models, including logistic regression and random forests.

    When you set class_weight='balanced', scikit‑learn increases the “weight” of minority‑class samples relative to majority‑class samples. In practical terms, the model is penalized more heavily for mistakes on the minority class (i.e., False Negatives) than for mistakes on the majority class (i.e., False Positives). Because the model tries to minimize total cost, it often becomes more inclined to predict the minority class, helping counteract the effects of class imbalance.

    Let’s try this out by creating a logistic regression model that uses class_weight='balanced'. This setting applies weights inversely proportional to the class frequencies in the training data, though you can also specify custom weights for each class if you prefer.
    """)
    return


@app.cell
def _(LogisticRegression):
    logreg_cost = LogisticRegression(
        solver='liblinear',
        class_weight='balanced', 
        random_state=1
    )
    return (logreg_cost,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We’ll fit our logistic regression model on the training set and use it to generate class predictions and predicted probabilities for the test set. When we compute the AUC, it increases from 0.93 to 0.94 simply by enabling class weighting.

    Keep in mind that class weighting isn’t guaranteed to improve AUC, so it should be treated like any other tunable parameter — something we’ll explore in the next chapter.
    """)
    return


@app.cell
def _(X_test, X_train, logreg_cost, roc_auc_score, y_score, y_test, y_train):
    logreg_cost.fit(X_train, y_train)
    y_pred_1 = logreg_cost.predict(X_test)
    y_score_1 = logreg_cost.predict_proba(X_test)[:, 1]
    roc_auc_score(y_test, y_score)
    return y_pred_1, y_score_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s look at the classification report to see how our rates have changed:

    - The True Positive Rate increased from 43% to 88%.
    - The True Negative Rate decreased from about 100% to 89%, meaning the False Positive Rate rose from roughly 0% to 11%.

    Even though this model may better reflect our priorities, its overall accuracy has dropped from 98% to 89%. This highlights an important point: a more useful classifier can sometimes have lower accuracy than the null accuracy, especially when class imbalance is involved.
    """)
    return


@app.cell
def _(classification_report, y_pred_1, y_test):
    print(classification_report(y_test, y_pred_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changes due to cost-sensitive learning:
    - TPR: 0.43 → 0.88
    - FPR: 0.00 → 0.11

    #### 18.8 Tuning the decision threshold
    At this point, we could continue tuning various aspects of the model to further increase the AUC, but instead we’ll move on to our final step: adjusting the decision threshold.

    Let’s look at the ROC curve for our class‑weighted logistic regression model. Using the default threshold of 0.5, the model achieves a True Positive Rate of 88% and a False Positive Rate of 11%, represented by a single point on the curve. If we want to move to a different point, one that better reflects our priorities, we simply change the threshold.
    """)
    return


@app.cell
def _(RocCurveDisplay, X_test, logreg_cost, y_test):
    disp_2 = RocCurveDisplay.from_estimator(logreg_cost, X_test, y_test)
    disp_2.figure_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before tuning the threshold, it’s helpful to look at the current confusion matrix, which reflects the default threshold of 0.5. You’ll notice that there are now far more True Positives and False Positives than before.
    """)
    return


@app.cell
def _(confusion_matrix, y_pred_1, y_test):
    confusion_matrix(y_test, y_pred_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More specifically, the model now predicts the positive class 352 times, compared with only 38 positive predictions previously.
    """)
    return


@app.cell
def _(y_score_1):
    sum(y_score_1 > 0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s say we’re still uncomfortable with having 8 False Negatives and want to reduce them even further. If we lower the threshold to 0.25, the model ends up predicting the positive class 870 times.
    """)
    return


@app.cell
def _(y_score_1):
    sum(y_score_1 > 0.25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The boolean array created by this condition can be converted into class predictions simply by multiplying it by 1.
    """)
    return


@app.cell
def _(y_score_1):
    (y_score_1 > 0.25) * 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In fact, the boolean array can be passed directly to the confusion‑matrix function, which will handle the conversion automatically.

    By lowering the threshold to 0.25, we can see that the number of False Negatives drops from 8 to 4, but the number of False Positives rises from 295 to 809. More generally, decreasing the threshold shifts samples from the left column of the confusion matrix to the right column.
    """)
    return


@app.cell
def _(confusion_matrix, y_score_1, y_test):
    confusion_matrix(y_test, y_score_1 > 0.25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looking at the classification report, the True Positive Rate has increased to 94%, while the False Positive Rate has risen to 30%. That shift moves us to a new point on the ROC curve.
    """)
    return


@app.cell
def _(classification_report, y_score_1, y_test):
    print(classification_report(y_test, y_score_1 > 0.25, zero_division=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changes due to decreasing the threshold:
    - TPR: 0.88 → 0.94
    - FPR: 0.11 → 0.30

    Now, let’s say we feel the original threshold produced too many False Positives. In that case, we could raise the threshold to 0.75, which shifts more samples from the right column of the confusion matrix to the left.
    """)
    return


@app.cell
def _(confusion_matrix, y_score_1, y_test):
    confusion_matrix(y_test, y_score_1 > 0.75)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see from the classification report that the True Positive Rate has dropped to 77%, while the False Positive Rate has fallen to 4%. Once again, this moves us to a different point on the ROC curve.
    """)
    return


@app.cell
def _(classification_report, y_score_1, y_test):
    print(classification_report(y_test, y_score_1 > 0.75, zero_division=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changes due to increasing the threshold:
    - TPR: 0.88 → 0.77
    - FPR: 0.11 → 0.04

    Keep in mind that changing the threshold doesn’t alter the model itself. Instead, it simply lets us trade off between two types of errors: False Positives and False Negatives.

    There’s no single “correct” threshold we’re trying to discover. The right threshold is the one that best aligns with your priorities. While there is a method for selecting the point on the ROC curve closest to the upper‑left corner and treating that as the optimal threshold, that value is only meaningful if it matches what you care about.
    """)
    return


if __name__ == "__main__":
    app.run()
