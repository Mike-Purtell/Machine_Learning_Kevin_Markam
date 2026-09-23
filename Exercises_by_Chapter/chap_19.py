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
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import confusion_matrix
    # from sklearn.metrics import ConfusionMatrixDisplay
    # from sklearn.metrics import roc_auc_score
    # from sklearn.metrics import RocCurveDisplay
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import classification_report
    return os, pl, train_test_split


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

    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
