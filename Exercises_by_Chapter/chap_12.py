# /// script
# dependencies = [
#     "marimo",
#     "polars==1.43.2",
#     "scikit-learn==1.9.0",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", layout_file="layouts/chap_10.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs
    import numpy as np

    import os

    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer    
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import RandomizedSearchCV
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.pipeline import Pipeline
    from sklearn.ensemble import VotingClassifier

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="`sklearn.utils.parallel.delayed` should be used"
    )
    return (
        CountVectorizer,
        LogisticRegression,
        OneHotEncoder,
        RandomForestClassifier,
        SimpleImputer,
        VotingClassifier,
        cross_val_score,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 12:  Ensembling multiple models
    - Started Tuesday September 15, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - TBD

    #### 12.1 Introduction to ensembling
    We’ve been using two different models, logistic regression and random forests. The process of  ensembling produces a combined model, known as an ensemble, that is more accurate than any of the individual models.
    The process for ensembling is:
    - For a regression, calculate the average of predictions from individual
    regressors to use as your prediction.
    - For a classification, average the predicted probabilities output by the
    classifiers, or let the classifiers vote on which class to predict

    Ensembling works because:
    - “One-off” errors made by each model will be discarded when ensembling
    - Ensemble has a lower variance than any individual model

    #### 12.2 Ensembling logistic regression and random forest
    """)
    return


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    OneHotEncoder,
    SimpleImputer,
    cross_val_score,
    make_column_transformer,
    make_pipeline,
    os,
    pl,
):
    # list of 6 columns we will select from the dataset
    cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

    # read the training dataset and use it to define X and y
    training_data_path = './assets/MLtrain.csv'
    if not os.path.exists(training_data_path):
        print("Downloading training dataset...")
        df = pl.read_csv('http://bit.ly/MLtrain')
        df.write_csv(training_data_path)
    else:
        print("Loading training dataset from local file...")
        df = pl.read_csv(training_data_path)

    df = pl.read_csv(training_data_path)
    X = df[cols]
    y = df.select('Survived')

    # read the new dataset and define X_new
    new_data_path = './assets/MLnewdata.csv'
    if not os.path.exists(new_data_path):
        print("Downloading new dataset...")
        df_new = pl.read_csv('http://bit.ly/MLnewdata')
        df_new.write_csv(new_data_path)
    else:
        print("Loading new dataset from local file...")
        df_new = pl.read_csv(new_data_path)

    X_new = df_new.select(cols)

    # create 4 instances of transformers
    # two SimpleImputers, one OneHotEncoder, and one CountVectorizer
    imp = SimpleImputer()
    imp_constant = SimpleImputer(strategy='constant', fill_value = 'missing')
    ohe = OneHotEncoder()
    vect = CountVectorizer()

    # create a 2-step pipeline of constant value imputation and one-hot encoding
    imp_ohe = make_pipeline(imp_constant, ohe)

    # build the column transformer which 
    #     imputes and one-hot encodes Embarked and Sex
    #     vectorizes Name
    #     imputes Age and Fare
    #     passes through Parch
    ct = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age', 'Fare']),
        ('passthrough', ['Parch'])
    )

    # create an instance of LogisticRegression
    logreg = LogisticRegression(
        solver='liblinear',
        # l1_ratio=0.0,
        random_state=1,
    )

    # create a 2-step pipeline, fit it to X and y, and make predictions on X_new
    pipe = make_pipeline(ct, logreg)
    pipe.fit(X, y.to_series())
    print(cross_val_score(pipe, X, y.to_series(), scoring ='accuracy').mean())
    return X, X_new, ct, logreg, pipe, y


@app.cell
def _(RandomForestClassifier, X, cross_val_score, ct, make_pipeline, y):
    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    rf_pipe = make_pipeline(ct, rf)
    print(cross_val_score(rf_pipe, X, y.to_series(), scoring ='accuracy').mean())
    return rf, rf_pipe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create the ensemble using the VotingClassifier class, which we’ll import from the ensemble module. Then create an instance called vc and pass it a list of tuples, in which the first element of the tuple is a name and the second element is a classifier object. The options for the voting parameter are 'soft', in which predicted probabilities are averaged, and 'hard', in which only class predictions are taken into account. Try soft voting first, with n_jobs to -1 to enable parallel processing.
    """)
    return


@app.cell
def _(VotingClassifier, logreg, rf):
    vc = VotingClassifier(
        [
            ('clf1', logreg),
            ('clf2', rf)
        ],
        voting='soft',
        n_jobs=-1
    )
    return (vc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create a new pipeline called vc_pipe in which VotingClassifier is the second step instead of a model
    """)
    return


@app.cell
def _(ct, make_pipeline, vc):
    vc_pipe = make_pipeline(ct, vc)
    vc_pipe
    return (vc_pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 12.3 Combining predicted probabilities
    Examine how the VotingClassifier makes predictions when using soft voting, by looking at the predicted probabilities output by the logistic regression and random forest Pipelines for the first 3 samples in X_new. The left column is the predicted probability of class 0 (for each sample), and the right column is the predicted probability of class 1.
    """)
    return


@app.cell
def _(X, X_new, pipe, y):
    pipe.fit(X, y.to_series())
    pipe.predict_proba(X_new)[:3]
    return


@app.cell
def _(X, X_new, rf_pipe, y):
    rf_pipe.fit(X, y.to_series())
    rf_pipe.predict_proba(X_new)[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now use the VotingClassifier to output predicted probabilities, using fit and predict_proba, just like other classifiers. If you examine its predicted probabilities, you’ll see that it’s simply averaging the two sets of probabilities from logistic regression and random forests. For example, the 0.19 in the left column of the second sample is the average of 0.14 (from logistic regression) and 0.24 (from random forests).
    """)
    return


@app.cell
def _(X, X_new, vc_pipe, y):
    vc_pipe.fit(X, y.to_series())
    vc_pipe.predict_proba(X_new)[:3]
    return


@app.cell
def _(X_new, vc_pipe):
    vc_pipe.predict(X_new[:3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the 3 cases we just examined, logistic regression and random forests agreed on the class predictions. Let’s now examine a case in which the two models disagree. For sample 80, logistic regression predicts class 0 but without much confidence, whereas random forests predicts class 1 with more confidence.
    """)
    return


@app.cell
def _(X_new, pipe):
    pipe.predict_proba(X_new)[80]
    return


@app.cell
def _(X_new, rf_pipe):
    rf_pipe.predict_proba(X_new)[80]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When VotingClassifier averages the predicted probabilities for this sample, the class 1 value is higher than the class 0 value, thus it will predict class 1.
    """)
    return


@app.cell
def _(X_new, vc_pipe):
    vc_pipe.predict_proba(X_new)[80]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s move on to cross-validation to see how the VotingClassifier Pipeline with soft voting performs. Its score is 0.818, which is better than either model alone.
    """)
    return


@app.cell
def _(X, cross_val_score, vc_pipe, y):
    cross_val_score(vc_pipe, X, y.to_series(), cv=5, scoring='accuracy').mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #BREAK AT START OF 12.4, page 187
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
