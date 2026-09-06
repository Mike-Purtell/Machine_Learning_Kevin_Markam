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
app = marimo.App(width="medium", layout_file="layouts/chap_08.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline

    import polars as pl
    import polars.selectors as cs

    return (pl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 9:  Workflow review #2
    Completed on Sunday September 6, 2026

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
    - TBD
    #### 9.1 Workflow recap

    In this chapter, we’re going to review the workflow that we’ve built so far and also discuss the concept of data leakage.

    To start, we’re going to walk through all of the code that’s necessary to recreate our workflow up to this point. The import block above has imported the three transformer classes we’re using, one modeling class, and two composition functions, and the polars datafame library.

    Next, define the columns we will select from the data, and read the training data to define X and y.
    """)
    return


@app.cell
def _(pl):
    cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']
    df_new = pl.read_csv('http://bit.ly/MLnewdata')
    X_new = df_new[cols]
    X_new # .shape  # 6 columns, 418 rows
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BREAK HERE, page 130
    """)
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
