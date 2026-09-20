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

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline



    return (
        CountVectorizer,
        FunctionTransformer,
        LogisticRegression,
        OneHotEncoder,
        SimpleImputer,
        make_column_transformer,
        make_pipeline,
        np,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 16:  Workflow review #3
    - Finished Sunday September 20, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - Reviewed workflow developed in recent chapters
    - Learned what the role of dataframe tools is for Machine Learning

    #### 16.1 Workflow recap
    This chapter has a final review of the core workflow built throughout this
    book, including features developed in the previous chapter.
    We begin by importing python libraries (always at the beginning of this file), the four transformer classes we’re using, one modeling class, and two composition functions.
    Create a list of the eight dataset columns we will use.
    """)
    return


@app.cell
def _():
    cols = [
        'Parch', 'Fare', 'Embarked', 'Sex', 
        'Name', 'Age', 'Cabin', 'SibSp'
        ]
    return (cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now read in all of our training data and use it to define X and y
    """)
    return


@app.cell
def _(cols, os, pl):
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
    return X, X_new, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create five instances of our transformers, with two instances of SimpleImputer, two instances of OneHotEncoder, and one instance of CountVectorizer.
    """)
    return


@app.cell
def _(CountVectorizer, OneHotEncoder, SimpleImputer):
    imp = SimpleImputer()
    imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
    ohe = OneHotEncoder()
    ohe_ignore = OneHotEncoder(handle_unknown='ignore')
    vect = CountVectorizer()
    return imp, imp_constant, ohe, ohe_ignore, vect


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define two custom functions that will be used for feature engineering.
    """)
    return


@app.cell
def _(np):
    def first_letter(df):
        arr = np.asarray(df, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        values = []
        for value in arr[:, 0]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                values.append(None)
            else:
                values.append(str(value)[0])

        return np.asarray(values, dtype=object).reshape(-1, 1)

    def sum_cols(df):
        arr = np.asarray(df, dtype=float)
        return arr.sum(axis=1, keepdims=True)

    return first_letter, sum_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We convert two NumPy functions and our two custom functions into transformers.
    """)
    return


@app.cell
def _(FunctionTransformer, first_letter, np, sum_cols):
    ceiling = FunctionTransformer(np.ceil)
    clip = FunctionTransformer(np.clip, kw_args={'a_min':5, 'a_max':60})
    letter = FunctionTransformer(first_letter)
    total = FunctionTransformer(sum_cols)
    return ceiling, clip, letter, total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create four Pipelines that combine the various transformers.
    """)
    return


@app.cell
def _(
    ceiling,
    clip,
    imp,
    imp_constant,
    letter,
    make_pipeline,
    ohe,
    ohe_ignore,
):
    imp_ohe = make_pipeline(imp_constant, ohe)
    imp_ceiling = make_pipeline(imp, ceiling)
    imp_clip = make_pipeline(imp, clip)
    letter_imp_ohe = make_pipeline(letter, imp_constant, ohe_ignore)
    return imp_ceiling, imp_clip, imp_ohe, letter_imp_ohe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now build the ColumnTransformer, which imputes and one-hot encodes Embarked and Sex, vectorizes Name, imputes and takes the ceiling of Fare, imputes and clips Age, imputes and one-hot encodes the first letter of Cabin, and adds SibSp and Parch.
    """)
    return


@app.cell
def _(
    imp_ceiling,
    imp_clip,
    imp_ohe,
    letter_imp_ohe,
    make_column_transformer,
    total,
    vect,
):
    ct = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp_ceiling, ['Fare']),
        (imp_clip, ['Age']),
        (letter_imp_ohe, ['Cabin']),
        (total, ['SibSp', 'Parch'])
    )
    return (ct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create an instance of LogisticRegression.
    """)
    return


@app.cell
def _(LogisticRegression):
    logreg = LogisticRegression(solver='liblinear', random_state=1)
    return (logreg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create a two-step modeling Pipeline and fit the Pipeline to X and y.
    """)
    return


@app.cell
def _(X, ct, logreg, make_pipeline, y):
    pipe = make_pipeline(ct, logreg)
    pipe.fit(X, y)
    pipe
    return (pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now use the fitted Pipeline to make predictions on X_new
    """)
    return


@app.cell
def _(X_new, pipe):
    pipe.predict(X_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are many other steps that could be incorporated into this workflow to potentially improve performance, including hyperparameter tuning, trying a different model, ensembling, feature selection, feature standardization, and additional feature engineering.

    #### 16.2: Whats the role of dataframe tools?
    If we can do all of our data transformations in sklearn, then what’s
    the role of dataframe tools?
    - data exploration and visualization. A deep understanding of the dataset greatly helps with development of the Machine Learning workflow, especially feature selection and feature transformation.
    - testing out data transformations for Machine Learning to make sure any new ideas really work
    - for anything other than Machine Learning, all of the data transformations should still be executed using dataframe tools.

    Dataframes have a huge role in the data science workflow. However, if your
    goal is Machine Learning, then it’s best to shift as much of your workflow as possible to sklearn or other machine learining libraries (for example XGBoost).
    """)
    return


if __name__ == "__main__":
    app.run()
