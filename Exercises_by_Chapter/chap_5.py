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
    # Chapter 5: Workflow review #1
    Completed on Thursday September 3, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - Great overview of material covered thus far
    - Best to review this chapter before deep dive reviews of first 4 chapters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 Recap of our workflow
    """)
    return


@app.cell
def _():
    import polars as pl
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline

    return (
        LogisticRegression,
        OneHotEncoder,
        make_column_transformer,
        make_pipeline,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Read 10 rows of training data, use it to define X and y
    """)
    return


@app.cell
def _(pl):
    cols = ['Parch', 'Fare', 'Embarked', 'Sex']
    df = pl.read_csv('http://bit.ly/MLtrain', n_rows=10)
    X = df.select(cols)
    y = df.select('Survived')
    return X, cols, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Read 10 rows of new data, use it to define X_new
    """)
    return


@app.cell
def _(cols, pl):
    df_new = pl.read_csv('http://bit.ly/MLnewdata', n_rows=10)
    X_new = df_new.select(cols)
    return (X_new,)


@app.cell
def _(OneHotEncoder):
    ohe = OneHotEncoder()
    return (ohe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Build column transformer
    """)
    return


@app.cell
def _(make_column_transformer, ohe):
    ct = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        ('passthrough', ['Parch', 'Fare'])
    )
    print(ct)
    ct
    return (ct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Setup logistic regression
    """)
    return


@app.cell
def _(LogisticRegression):
    logreg = LogisticRegression(solver='liblinear', random_state=1)
    print(logreg)
    logreg
    return (logreg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### create 2-step pipeline
    - fit X and y
    - make prediction on X_new
    """)
    return


@app.cell
def _(X, X_new, ct, logreg, make_pipeline, y):
    pipe = make_pipeline(ct, logreg)
    pipe.fit(X, y)
    pipe.predict(X_new)
    return (pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 Comparing ColumnTransformer and Pipeline

    Step |ColumnTransformer| Pipeline|
    |:--:|--|--|
    |1| selected 2 cols, Embarked & Sex | ColumnTransformer, recieved 4 cols as input|
    |2| selected 2 other cols, Parch & Fare | LogisticRegression model, 7 cols|
    |3| horiz stack, 5 cols from ohe, 2 from passthrough | |
    """)
    return


@app.cell
def _(mo):
    # use marimo mermaid to visualize by recreating the diagram from the book

    diagram = mo.mermaid(
        """
        flowchart TD
            X[\"X\\n4 columns\"] --> CT[\"ColumnTransformer\"]

            CT -->|\"Embarked, Sex\"| OHE[\"OneHotEncoder\\n5 columns\"]
            CT -->|\"Parch, Fare\"| PT[\"passthrough\\n2 columns\"]

            OHE --> M[\"7 columns\"]
            PT --> M

            M --> LR[\"LogisticRegression\"]
        """
    )
    diagram

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **ColumnTransformer**
    - Selects subsets of columns and transforms each subset independently, concatenates
    results side-by-side
    - Only does data transformations, never includes a model
    - Each subset of columns is transformed independently

    **Pipeline**
    - Series of ordered steps
    - Output of each step is input to next step
    - Last step is a model or a transformer. All other steps are transformers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 Creating a  Pipeline diagram
    new feature of scikit-learn 0.23
    """)
    return


@app.cell
def _():
    from sklearn import set_config
    set_config(display='diagram')
    return (set_config,)


@app.cell
def _(pipe):
    pipe
    return


@app.cell
def _(pipe):
    print(pipe)
    return


@app.cell
def _(pipe, set_config):
    set_config(display='text')
    pipe
    return


if __name__ == "__main__":
    app.run()
