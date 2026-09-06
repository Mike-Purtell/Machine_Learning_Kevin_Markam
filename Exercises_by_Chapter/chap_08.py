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
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import make_union

    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.impute import KNNImputer
    from sklearn.impute import MissingIndicator

    # Iterative Imputer is expermental, requires enabling this feature explicitly
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    import polars as pl
    import polars.selectors as cs

    return (
        CountVectorizer,
        OneHotEncoder,
        SimpleImputer,
        make_column_transformer,
        make_pipeline,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 8:  Fixing Common workflow problems
    Completed on Sunday September 6, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - OneHotEncoder treats missing values as a new category starting with scikit-learn 0.24, which I am using. Book uses earlier version that throws errors when column passed to OneHotEncoder has any missing values.
    - TBD
    - TBD
    - TBD
    - TBD
    #### 8.1 Two new problems
    So far this book has only used the first 10 rows of the Titanic dataset, to simpify examination of the input and output of each workflow step.

    The rest of the book will use the full Titanic dataset. This will expose common problems with real datasets, which we will figure out how to handle appriopriately. Will begin by reading the training data into df and reading the new data into df_new. df_new willl have one fewer column than df because it doesn’t contain the target column of Survived.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(pl):
    # Initialize our dataframes for training and new data removed n_rows paramter to use the full datast
    df = pl.read_csv('http://bit.ly/MLtrain')
    df_new = pl.read_csv('http://bit.ly/MLnewdata')
    print(f'{df.shape = }')             # 11 columns, 891 rows
    print(f'{df_new.shape = }')     # 10 columns, 418 rows
    return df, df_new


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### count the number of missing values by column for in each dataframe
    """)
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell
def _(df_new):
    df_new.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Features with missing values:
    - Problematic:
      - Embarked: Missing values in df
      - Fare: Missing value in df_new
    - Not problematic:
      - Cabin: Not currently using
      - Age: Already being imputed
    #### 8.2 Problem 1: Missing values in a categorical feature

    Six feature columns we use are:
    """)
    return


@app.cell
def _(df):
    cols = ['Parch',  'Fare', 'Embarked', 'Sex', 'Name', 'Age']
    X = df.select(cols)
    y = df.select('Survived')
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I don't see errors that book mentioned due to missing values values in Embarked. This was explained as an attribute of OneHotEncoder, where prior to scikit-learn 0.24 errors were raised if input contained missing values. For 0.24 onward that I am using, OneHotEncoder treats missing values as a new category, and runs without error.
    """)
    return


@app.cell
def _(
    CountVectorizer,
    OneHotEncoder,
    SimpleImputer,
    X,
    make_column_transformer,
):
    ohe = OneHotEncoder()
    vect = CountVectorizer()
    imp = SimpleImputer()
    ct = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age']),
        ('passthrough', ['Parch', 'Fare'])
    )
    ct.fit_transform(X)
    imp
    return imp, ohe, vect


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even though my version of scikit-learn can handle missing values passed to OneHotEncoder, I will follow along with the steps to fix this with SimpleImputer. My thinking is that imputing a few missing values is better than making them an unknown category.

    First step is to create a new instance of SimpleImputer, and use a strategy of 'constant' to replace missing values. Note the missing_values parameter set to None, for polars dataframe. The book did not  use the missing_values paramter, so default is NaN, which is what pandas uses for misssing values.
    """)
    return


@app.cell
def _(SimpleImputer):
    imp_constant = SimpleImputer(
        strategy = 'constant',
        fill_value = 'missing',
        missing_values = None   # to show missing values as 'missing', had to add this parameter to imp_constant
    )
    imp_constant
    return (imp_constant,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Imputation strategies for categorical features:
    - Most frequent value
    - User-defined value
    Create a two-step pipeline that only contains transformers. First step is imputation using our new imputer, and second step in one-hot encoding. Call this pipelinge imp_ohe to show of the two steps it contains.
    """)
    return


@app.cell
def _(X, imp_constant, make_pipeline, ohe):
    imp_ohe = make_pipeline(imp_constant, ohe)
    imp_ohe.fit_transform(X.select('Embarked'))
    return (imp_ohe,)


@app.cell
def _(imp_ohe):
    imp_ohe[1].categories_
    return


@app.cell
def _(imp, imp_ohe, make_column_transformer, vect):
    ct_imp_ohe = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age']),
        ('passthrough', ['Parch', 'Fare'])
    )
    return (ct_imp_ohe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are two things about the imp_ohe Pipeline:
    - Only transformer objects may be included in a ColumnTransformer.
      - imp_ohe is eligible because all of its steps are transformers.
    - It's OK to apply imp_ohe to the Sex column, as it has no missing values, and makes the imputation step is a NOP. It will simply get passed to the one-hot encoding step.

    Notice below that the output matrix is now much wider than before because the Name column of X contains a large number of unique words
    """)
    return


@app.cell
def _(X, ct_imp_ohe):
    ct_imp_ohe.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 8.3:  Missing values in the new data

    # BREAK HERE, PAGE 119, TO BE CONTINUED
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
