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
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    # from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import RocCurveDisplay
    # from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.metrics import PrecisionRecallDisplay

    return (
        ConfusionMatrixDisplay,
        GridSearchCV,
        LogisticRegression,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        classification_report,
        confusion_matrix,
        cross_val_score,
        np,
        os,
        pl,
        roc_auc_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 19:  Class walkthrough
    - Finished Wednesday September 23, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - TBD
    - TBD
    - TBD
    #### 19.1 Best practices for class imbalance
    We just covered a lot of new ideas in the previous chapter: class imbalance, the confusion matrix and its derived rates, the classification report, ROC curves and AUC (Area Under the Curve), the decision threshold, and cost‑sensitive learning. That’s a big toolkit, and now we have seen how all these pieces fit together when evaluating and tuning a classifier.

    To properly demonstrate these concepts, I intentionally bent my real workflow in two ways that aren’t ideal. In this chapter, I’m going to walk through my actual best‑practice workflow for class‑imbalance problems so you can see how these steps are handled in a production‑quality setting.

    So what were the two shortcuts I took before?

    1. I used a simple train/test split instead of cross‑validation.
    Cross‑validation is generally the better choice because it’s easier to use consistently and it produces more reliable estimates of model performance.

    2. I tuned the decision threshold on the same dataset used for model optimization.
    In practice, it’s better to tune the threshold on data different from the data used to train and optimize the model. Experiments show that this leads to more reliable estimates of the True Positive Rate and False Positive Rate.

    In this chapter, I’m going to walk through my full workflow, start to finish,  incorporating both of these two improvements.

    #### 19.2: Splitting the dataset

    Step 1 in my real workflow is to split the dataset into training and testing sets. At first glance, this might seem contradictory, since I just said we’ll be using cross‑validation in this chapter rather than a train/test split.

    That’s true — but in this chapter we’re using train_test_split for a different purpose than we did in chapter 18:

    In chapter 18, the train/test split was used for model evaluation.

    In this chapter, the split is used to set aside independent data for tuning the decision threshold.

    If you want a deeper explanation of why this matters, you can revisit lesson 10.12.

    Anyway, the train_test_split code here is the same as what we used in lesson 18.3, including the use of stratified sampling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now read in the mammograpy dataset:
    """)
    return


@app.cell
def _(os, pl, train_test_split):
    if not os.path.exists('assets/scanrecords.csv'):
        print("Downloading scan records dataset...")
        scan = pl.read_csv('http://bit.ly/scanrecords')
        scan.write_csv('assets/scanrecords.csv')
    else:
        print("Loading scan records dataset from local file...")
        scan = pl.read_csv('assets/scanrecords.csv')

    scan.head()

    scan_X = scan.drop('class')
    scan_y = scan.select('class')

    X_train, X_test, y_train, y_test = train_test_split(scan_X, scan_y,
    test_size=0.25,
    random_state=1,
    stratify=scan_y)
    return X_test, X_train, scan_X, scan_y, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    #### 19.3: Optimizing the model on the training set
    Step 2 is to optimize the model (or Pipeline) using only the training set. As you can see, I’m using cross‑validation as the evaluation procedure and AUC as the metric to optimize. The key detail is that I’m passing only the training data to cross_val_score. I’m deliberately keeping the testing set untouched so it can serve as an independent dataset for the next step, where we’ll tune the decision threshold.
    Note that 'roc_auc' is the probability that the model assigns a higher score to a randomly chosen positive case than to a randomly chosen negative case.
    - ROC = Receiver Operating Characteristic curve
    - AUC = Area Under the Curve
    """)
    return


@app.cell
def _(LogisticRegression, X_train, cross_val_score, y_train):
    logreg = LogisticRegression(solver='liblinear')
    cross_val_score(
        logreg,
        X_train,
        y_train.to_series(),
        cv=5,
        scoring='roc_auc'
    ).mean()
    return (logreg,)


@app.cell
def _(mo):
    mo.md(r"""
    To optimize the model, we’ll use a grid search. Normally, this would involve tuning every step of the Pipeline, but in this case we’re focusing the search solely on the model itself.

    Because cost‑sensitive learning is valuable when dealing with class imbalance, we’re including the class_weight parameter in the search. We’ll try four options:

    - None, the default, meaning no cost‑sensitive learning.

    - 'balanced', the option we used earlier. It assigns weights inversely proportional to class frequencies. Since our data is roughly 98% class 0 and 2% class 1, this corresponds to a weight of 2 for class 0 and 98 for class 1.

    - Custom weights: {0:1, 1:99} — this applies an even stronger emphasis on class 1 than 'balanced'.

    - Custom weights: {0:3, 1:97} — this applies a slightly lower emphasis on class 1 than 'balanced'.
    """)
    return


@app.cell
def _():
    im_params = {}
    im_params['l1_ratio'] = [0, 1]
    im_params['C'] = [0.1, 1, 10]
    im_params['class_weight'] = [None, 'balanced', {"'-1'":1, "'1'":99}, {"'-1'":3, "'1'":97}]
    return (im_params,)


@app.cell
def _(mo):
    mo.md(r"""
    Now that we’ve defined the parameter grid, we can pass it to GridSearchCV and use AUC as the optimization metric. As before, the search is run only on the training set so that the testing set remains untouched for threshold tuning.

    The grid search yields an AUC of 0.92, a modest improvement over the 0.91 from the unoptimized model.
    """)
    return


@app.cell
def _(GridSearchCV, X_train, im_params, logreg, y_train):
    training_grid = GridSearchCV(
        logreg, 
        im_params, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1
    )
    training_grid.fit(X_train, y_train.to_series())
    training_grid.best_score_
    return (training_grid,)


@app.cell
def _(mo):
    mo.md(r"""
    Here’s the best parameter set the search found. Interestingly, it uses one of the custom class‑weight configurations.
    """)
    return


@app.cell
def _(training_grid):
    training_grid.best_params_
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now that we’ve identified the best parameters, we can save the model configured with those settings as an object called best_model.
    """)
    return


@app.cell
def _(training_grid):
    best_model = training_grid.best_estimator_
    best_model
    return (best_model,)


@app.cell
def _(mo):
    mo.md(r"""
    #### 19.4: Evaluating the model on the testing set
    Step 3 is to use our best model to generate predictions for the testing set and evaluate those predictions. Because we kept the testing set completely untouched during Step 2, the model has never seen this data, which means it can serve as a genuinely independent check on performance.
    """)
    return


@app.cell
def _(X_test, best_model):
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]
    return y_pred, y_score


@app.cell
def _(mo):
    mo.md(r"""
    We’ll evaluate the predicted probabilities using AUC and the ROC curve. The AUC comes out to 0.94, which is our best estimate of how well the trained model will perform on truly new, unseen data.
    """)
    return


@app.cell
def _(RocCurveDisplay, X_test, best_model, roc_auc_score, y_score, y_test):
    y_test_series = y_test.to_series()
    print(roc_auc_score(y_test_series, y_score))
    disp = RocCurveDisplay.from_estimator(\
        best_model, 
        X_test, 
        y_test_series
    )
    disp.figure_
    return


@app.cell
def _(mo):
    mo.md(r"""
    We’ll evaluate the class predictions using a confusion matrix and the classification report. The results show a True Positive Rate (class 1 recall) of 95% and a False Positive Rate (1 minus class 0 recall) of 24%
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, X_test, best_model, y_test):
    confusion_matrix_display = ConfusionMatrixDisplay.from_estimator(
        best_model, 
        X_test, 
        y_test
    )
    confusion_matrix_display.figure_
    return


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 19.5: Tuning the decision threshold
    Step 4 is to tune the decision threshold based on our priorities, specifically, our tolerance of False Negatives versus False Positives.

    This is the same process you saw in lesson 18.8, but with one important improvement: this time we’re tuning the threshold using data the model never saw during optimization. That separation matters, because it produces more reliable estimates of the True Positive Rate and False Positive Rate.

    Let’s say we want to reduce the False Positive Rate, and we’re willing to accept a slightly lower True Positive Rate to get there. To make that tradeoff, we’ll nudge the decision threshold upward to 0.55.
    """)
    return


@app.cell
def _(confusion_matrix, np, y_score, y_test):
    threshold_predictions = np.where(
        y_score > 0.55,
        "'1'",
        "'-1'"
    )
    confusion_matrix(
        y_test.to_series(),
        threshold_predictions,
        labels=["'-1'", "'1'"]
    )
    return


@app.cell
def _(classification_report, np, y_score, y_test):
    _threshold_predictions = np.where(
        y_score > 0.55,
        "'1'",
        "'-1'"
    )
    print(classification_report(
        y_test.to_series(),
        _threshold_predictions,
        labels=["'-1'", "'1'"]
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The True Positive Rate drops from 95% to 92%, and the False Positive Rate falls from 24% to 20%. Let’s assume we’re satisfied with that tradeoff and move on to the final step.

    #### 19.6: Retraining the model and making predictions
    Step 5 is to apply our chosen decision threshold when making predictions on new data.

    Before doing that, it’s essential to retrain our best model on all available data (the full scan_X and scan_y). Otherwise, we’d be discarding valuable information. In other words, we take the hyperparameters selected during Step 2 and fit the model using the entire dataset.
    """)
    return


@app.cell
def _(best_model, scan_X, scan_y):
    print("Fitting the best model...")
    best_model.fit(scan_X, scan_y.to_series())
    print(best_model.fit(scan_X, scan_y.to_series()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We’ll use the model to generate predictions for new data, whose true class labels are unknown. Since I don’t have any actual new samples, I’ll create a small synthetic dataset for demonstration.

    To keep the results reproducible, I’ll set NumPy’s random seed. Then I’ll use randint to generate a 4×6 array of integers between 0 and 2. This gives us four simulated samples of new data, each with six features.
    """)
    return


@app.cell
def _(np):
    np.random.seed(1)
    scan_X_new = np.random.randint(0, 3, (4, 6))
    scan_X_new
    return (scan_X_new,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We’ll generate predictions by passing the new data to the predict_proba method and storing the resulting probabilities.
    """)
    return


@app.cell
def _(best_model, scan_X_new):
    scan_y_new_score = best_model.predict_proba(scan_X_new)[:,1]
    return (scan_y_new_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we’ll predict class 1 whenever the predicted probability exceeds our decision threshold of 0.55; otherwise, we’ll assign class 0. These are the resulting class predictions for the four new samples.
    """)
    return


@app.cell
def _(scan_y_new_score):
    (scan_y_new_score > 0.55) * 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 19.7 : Should I use and ROC curve or a precision-recall curve?
    One alternative to ROC curves you may have encountered is the precision–recall curve. In this lesson, I’ll explain how the precision–recall curve works and then compare it to the ROC curve.

    To begin, let’s revisit the confusion matrix for our best model. We’ll fit the model on X_train and y_train, then generate both class predictions and predicted probabilities for X_test.
    """)
    return


@app.cell
def _(X_test, X_train, best_model, y_train):
    best_model.fit(X_train, y_train.to_series())
    y_pred_1 = best_model.predict(X_test)
    y_score_1 = best_model.predict_proba(X_test)[:, 1]
    return (y_pred_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then build the confusion matrix by comparing the true labels with the predicted labels. From that matrix, we’ll calculate two key rates for this part of the analysis.
    """)
    return


@app.cell
def _(confusion_matrix, y_pred_1, y_test):
    confusion_matrix(y_test, y_pred_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first rate is recall, which is another name for the True Positive Rate. It answers the question: When cancer is present, how often does the model correctly identify it? We compute it by dividing the True Positives by the entire bottom row of the confusion matrix, giving us a recall of 95%. **62/(3 + 62)**

    The second rate is precision. It answers the question: When the model predicts cancer, how often is that prediction correct? We calculate it by dividing the True Positives by the entire right column of the confusion matrix, which gives us 9%. Unlike the other rates we’ve computed, precision uses a column total rather than a row total. **62/(650 + 62)**

    Both precision and recall appear in the classification report, and the values we just computed for class 1 match the entries shown here.
    """)
    return


@app.cell
def _(classification_report, y_pred_1, y_test):
    print(classification_report(y_test, y_pred_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we’ve covered precision and recall, let’s plot the precision–recall curve using plot_precision_recall_curve. Its API is similar to plot_roc_curve, you pass the fitted model, X_test, and y_test.
    """)
    return


@app.cell
def _(PrecisionRecallDisplay, X_test, best_model, y_test):
    disp_1 = PrecisionRecallDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
    )
    disp_1.figure_
 
    return


@app.cell
def _():
    return


@app.cell
def _():
    print(f'{100*281/290:.1f} %')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
