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
app = marimo.App(width="medium", layout_file="layouts/chap_2.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Master Machine Learning - Chapter 2
    ## Review of the Machine Learning workflow
    Completed on Sunday August 23, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - Ran all code blocks from the book, after modification from pandas to polars
    - Output of all code blocks exactly matches what is shown in the book.
    - There was no impact from sklearn version differences.
    - Section 2.4 has a great comparison of linear regression that I know well to logistic regression, where my experience is very light. This comparison sets a nice tone for me to carry while studying this book
    - Chapter 2 is a binary classification problem, but nice to see the multiclass explanation in 2.5
    - Interesting idea to use a series for binary classification
    - Need to remember closing point, to use KFold instead of StratifiedKFold for regression problems.
    """)
    return


@app.cell
def _():
    import polars as pl
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    return LogisticRegression, cross_val_score, pl, sklearn


@app.cell
def _(sklearn):
    sklearn.__version__
    return


@app.cell
def _(pl):
    df = pl.read_csv('Datasets/titanic_train.csv', n_rows=10)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    X = df.select(['Parch', 'Fare'])
    X
    return (X,)


@app.cell
def _(df):
    y = df.select('Survived').to_series()
    y
    return (y,)


@app.cell
def _(y):
    y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic Regression
    Logistic regression (referred to as logreg in the example below) is used for classification problems, where the target variable is categorical. Contrast with regression, where the target is continuous.
    For classification problems where samples are in an arbitrary (ie random) order,
    just pass an integer to the cv parameter of cross_val_score as shown here, and it will use stratified sampling without shuffling. Otherwise need use StratifiedKFold as a splitter and
    set shuffle to True, then pass the splitter object cv.
    """)
    return


@app.cell
def _(LogisticRegression, X, cross_val_score, y):
    logreg = LogisticRegression(solver='liblinear', random_state=1)
    cross_val_score(logreg, X, y, cv=3, scoring='accuracy').mean()
    return (logreg,)


@app.cell
def _(X, logreg, y):
    logreg.fit(X, y)
    return


@app.cell
def _(pl):
    df_new =pl.read_csv('Datasets/titanic_new.csv', n_rows=10)
    df_new
    return (df_new,)


@app.cell
def _(df_new):
    X_new = df_new.select(['Parch', 'Fare'])
    X_new
    return (X_new,)


@app.cell
def _(X_new, logreg):
    logreg.predict(X_new)
    return


@app.cell
def _(X_new, logreg, pl):
    predictions = pl.DataFrame({'Prediction': logreg.predict(X_new)})
    predictions
    return (predictions,)


@app.cell
def _(X_new, pl, predictions):
    pl.concat([X_new, predictions], how='horizontal')
    return


@app.cell
def _(X_new, logreg):
    logreg.predict_proba(X_new)
    return


@app.cell
def _(X_new, logreg):
    logreg.predict_proba(X_new)[:, 1]
    return


if __name__ == "__main__":
    app.run()
