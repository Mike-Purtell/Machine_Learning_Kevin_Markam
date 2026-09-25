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

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import RocCurveDisplay

    from sklearn.metrics import classification_report
    from sklearn.metrics import PrecisionRecallDisplay
    from sklearn.metrics import average_precision_score

    return (
        ConfusionMatrixDisplay,
        GridSearchCV,
        LogisticRegression,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        average_precision_score,
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
    - Separate your data for separate purposes. Use a train/test split not for model evaluation (that's what cross-validation is for) but to reserve an independent set for tuning the decision threshold — tuning the threshold on the same data used to optimize the model gives overly optimistic/unreliable TPR and FPR estimates.

    - AUC vs. average precision is a "what do you care about" choice, not a "which is more correct" choice. AUC reflects performance on both classes (sensitivity + specificity), while average precision focuses solely on the positive class and ignores True Negatives — so it's more robust to reporting an "artificially high" score under severe class imbalance, but only tells you about the positive class. For problems like cancer detection where both classes matter, AUC is the better primary metric.

    - Optimize threshold-independent metrics first, then tune the threshold. AUC and average precision summarize performance across all thresholds, so use them to select/tune the model. Threshold-dependent metrics (F1, F-beta, balanced accuracy, Cohen's kappa, MCC) should only be used afterward, to pick the best decision threshold — using them during model tuning risks optimizing around a default 0.5 threshold that isn't right for your problem.

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
def _(mo):
    mo.md(r"""
    The precision–recall curve plots precision (y‑axis) against recall (x‑axis) across all possible decision thresholds. Like the ROC curve, it helps you choose a threshold that aligns with your priorities.

    Just as an ROC curve can be summarized by the area under the curve, a precision–recall curve can be summarized the same way. Scikit‑learn uses average precision as its summary metric, one of several possible ways to compute the area under a precision–recall curve.

    **Interpreting the precision-recall curve:**
    - Plot of precision vs recall for all possible decision thresholds
    - Move to another point on the curve by changing the threshold
    - Average precision is the percentage of the plot underneath the curve

    To compute average precision, we import average_precision_score and pass it the true labels along with the predicted probabilities. The function returns a score of 54.7 %.
    """)
    return


@app.cell
def _(average_precision_score, y_score, y_test):
    print(f'{100*average_precision_score(y_test, y_score, pos_label="\'1\'"):.1f} %')
    return


@app.cell
def _(mo):
    mo.md(r"""
    A perfect model would score 1.0, while a completely uninformed model would score roughly the fraction of positive samples—in this case, about 0.02. For comparison, remember that an uninformed model’s ROC AUC is 0.5, which means average precision and AUC have very different baseline levels for the same problem.

    **Precision-recall scores:**
    - Perfect model: 1.0
    - Uninformed model: 0.02 in this case (fraction of positive samples)

    In this chapter, I recommended using AUC as the primary evaluation metric when dealing with class imbalance. That said, many practitioners prefer average precision instead. So which one should you use?

    To answer that, I’ll start with the most common critique of AUC and then give you my response. It may help to keep the confusion matrix in mind as we walk through this discussion, so I’ll display it here.
    """)
    return


@app.cell
def _(confusion_matrix, y_pred, y_test):
    confusion_matrix(y_test, y_pred)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s quickly run through the rates again:
    - True Positive Rate (recall): 62 out of 65, which is 95%.
    - False Positive Rate: 649 out of 2,731, which is 24%.
    - Precision is 62 out of 711, which is 9%.

    With that in mind, here’s the most common critique of AUC in highly imbalanced settings:
    When the classes are severely imbalanced, the number of True Negatives becomes extremely large. That makes the False Positive Rate look artificially low, which in turn makes the AUC look artificially high. As a result, AUC may no longer give a realistic picture of the model’s performance.

    **Critique of AUC in cases of class imbalance**
    - **Effects of severe class imbalance**
      - The number of True Negatives becomes extremely large.
      - The False Positive Rate becomes artificially low.
      - As a result, the AUC becomes artificially high and no longer reflects real performance.
    - **Example**
      - If True Negatives increase from 2,082 to 200,000…
      - The False Positive Rate would drop from 0.24 to 0.003.
      - AUC would increase.
      - Precision would remain 0.09.
    - **Proposed solution**
      - Use the precision–recall curve and average precision.
      - These metrics are more realistic in imbalanced settings because they ignore the number of True Negatives.

    As an example, imagine increasing the number of True Negatives from 2,082 to 200,000 while leaving all other values unchanged. The False Positive Rate would drop from 24% to 0.3%, and the AUC would rise—though we can’t say by how much, since AUC can’t be computed directly from a confusion matrix. The model would appear excellent based on its AUC, even though its precision would still be only 9%.

    According to this critique, the solution is to rely on the precision–recall curve and average precision, which offer a more realistic assessment in imbalanced settings because they ignore the number of True Negatives.

    **Responses to the Critique of AUC**
    1. The choice between AUC and average precision depends on what you want to measure
    - AUC evaluates performance across both classes. In this context, it measures how well the model detects cancer when it’s present and how well it avoids predicting cancer when it’s not present.
    - Average precision, by contrast, focuses solely on the positive class. Because precision and recall ignore True Negatives, the average precision score is unchanged whether you have two thousand or two million True Negatives.

    Choosing between the two:
    - Use AUC when you care about performance across both classes.
    - Use average precision when your priority is strictly how well the model identifies the positive class.

    For cancer detection, performance on both classes matters, so AUC is the more appropriate choice.

    2. Precision can be “artificially low” in the same scenario
    In cases of severe class imbalance, it’s true that the False Positive Rate can look artificially low, making the model appear better than it is. But the critique cuts both ways: precision can look artificially low, making the model appear worse than it is.

    Consider the hypothetical confusion matrix:
    - If someone doesn’t have cancer, they have only a 0.3% chance of being incorrectly told they do. 649/200,649 = 0.003
    - If someone does have cancer, they have only a 5% chance of being incorrectly told they don’t. 3/65 = 0.05

    Despite these strong characteristics, the model’s precision is still 9%. Even if all 3 False Negatives were moved into the True Positive box—giving a True Positive Rate of 100%—precision would still be 9%. Precision alone would make this model sound terrible, even though it’s clearly quite good.

    3. The actual AUC score is irrelevant
    During model tuning, the purpose of an evaluation metric is simply to provide something meaningful to maximize when comparing models. AUC works well because it measures how effectively the model separates the classes.

    Once you’ve maximized AUC, you can adjust the decision threshold to achieve the True Positive Rate and False Positive Rate that match your priorities. The AUC score itself is not your business objective, so it doesn’t matter if the score is “artificially high.” What matters is that it helps you choose a better model.

    **Bottom Line**
    - Both AUC and average precision are reasonable metrics to maximize, even in cases of class imbalance.
    - Neither metric perfectly represents a model’s performance—each highlights different aspects.
    - Choose AUC when you care about performance across both classes.
    - Choose average precision when your focus is strictly on how well the model identifies the positive class.

    #### 19.8: Can I use a different metric such as F1 score?
    There are several other metrics commonly used for class‑imbalance problems, including F1 score, F‑beta score, balanced accuracy, Cohen’s kappa, and Matthews correlation coefficient.
    All of the metrics listed above—F1 score, F‑beta score, balanced accuracy, Cohen’s kappa, and Matthews correlation coefficient—require you to choose a decision threshold. In contrast, AUC and average precision summarize a classifier’s performance across all possible thresholds.

    Because of that, using AUC or average precision during model tuning allows you to first maximize the model’s overall ability to separate the classes, and then adjust the decision threshold afterward to match your specific priorities.

    If you try to maximize F1 score (or any other threshold‑dependent metric) during model tuning, you end up optimizing the model’s hyperparameters around the default decision threshold of 0.5. But that threshold may be far from optimal for your specific problem. In other words, you could easily miss out on a better model simply because you tuned it using a non‑optimal threshold.

    If you want to use any of these other metrics, I recommend using them **only to choose between different decision thresholds** after you’ve already optimized your model for **AUC** or **average precision**. This way, you first maximize the model’s overall ability to separate the classes, and then use the threshold‑dependent metrics to fine‑tune the decision threshold to match your priorities.

    #### 19.9: Should I use resampling to fix class imbalance?
    In cases of class imbalance, there’s a set of techniques collectively known as resampling that is often used. Resampling refers to any technique that transforms the training data in order to achieve more balance between the classes. In other words, resampling attempts to fix the class imbalance at the dataset level rather than working around it, which is what we’ve done in this chapter.
    Here are the two most common resampling approaches:
    - Undersampling (or downsampling) is the process of deleting samples from the majority class.
    - Oversampling (or upsampling) is the process of creating new samples from the minority class,
    either by duplicating existing samples or by simulating new samples. One popular oversampling
    method that simulates new samples is SMOTE (Synthetic Minority Over‑sampling Technique).

    Both of these approaches can be done in either a directed, strategic fashion or in a random fashion.
    Or they can be done together, in which you both undersample and oversample.

    Regardless of the specific approach, keep in mind that the act of resampling risks deleting important samples and/or adding meaningless new samples.

    All of that being said, is resampling actually helpful? Experimental results show that resampling
    methods can be helpful, but are not always helpful. And while there are dozens of different resampling algorithms, no one algorithm works best across all datasets and models, meaning that it’s hard to give practical advice for which one to use.
    If you decide to pursue resampling, keep in mind that it’s not yet supported by sklearn, though it may eventually be available. In the meantime, you can use the imbalanced-learn library, which is
    supposed to be fully compatible with scikit-learn. Personally, I tend not to use this approach in order to avoid adding additional complexity or project dependencies.

    Here are two guidelines for the proper use of resampling:
    - Treat resampling like any other preprocessing technique. Namely, it should be
    included in a Pipeline in order to avoid data leakage.
    - The resampling technique should only ever be applied to training data, and not testing
    data. The model should always be tested on the natural, imbalanced data so that it can output
    a realistic estimate of model performance.
    """)
    return


if __name__ == "__main__":
    app.run()
