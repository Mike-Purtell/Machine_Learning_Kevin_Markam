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


    return (
        CountVectorizer,
        LogisticRegression,
        OneHotEncoder,
        RandomForestClassifier,
        SimpleImputer,
        cross_val_score,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 11:  Comparining linear and non-linear models
    - Started Tuesday September 15, 2026  - first page is 170
    - Completed XXX September YYY, 2026

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
    #### 11.1 Trying a random forest model
    following blocks setup ML objects used in the chapter.
    """)
    return


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    OneHotEncoder,
    SimpleImputer,
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
        l1_ratio=0.0,
        random_state=1,
    )

    # create a 2-step pipeline, fit it to X and y, and make predictions on X_new
    pipe = make_pipeline(ct, logreg)
    pipe.fit(X, y.to_series())
    pipe.predict(X_new)
    return X, ct, pipe, y


@app.cell
def _(X, cross_val_score, pipe, y):
    print(cross_val_score(pipe, X, y.to_series(), scoring ='accuracy').mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create an instance of RandomForestClassifier called rf. Set random_state to 1 for reproducibility. Random forest can be computationally expensive, set n_jobs parameter to -1 (just like grid search and randomized search), to enable parallel processing.
    """)
    return


@app.cell
def _(RandomForestClassifier, ct, make_pipeline):
    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    rf_pipe = make_pipeline(ct, rf)
    rf_pipe
    return (rf_pipe,)


@app.cell
def _(X, cross_val_score, rf_pipe, y):
    cross_val_score(
        rf_pipe, 
        X, 
        y.to_series(), 
        cv=5, 
        scoring ='accuracy'
        ).mean()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
