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
    # import numpy as np

    # from sklearn.impute import SimpleImputer
    # from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    # from sklearn.feature_extraction.text import CountVectorizer
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.compose import make_column_transformer
    # from sklearn.pipeline import make_pipeline


    return os, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 18:  Class imbalance
    - Finished XXX September YY, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - A small amount of imbalance (like in the Titanic dataset) tends not to matter.
    - As the amount of class imbalance increases, more specialized techniques need to be applied

    #### 18.1 Workflow recap
    A common issue when working on a classification problem is known as
    class imbalance, where classes are not equally represented in the
    dataset.

    Class imbalance is inherent to many domains. For example, fraudulent transactions in a credit card dataset would have a natural class imbalance since only a tiny fraction of transactions are likely to be fraudulent.

    Class imbalance can occur in binary classification problems where there are only two classes and in multiclass problems with more than two classes. In binary classification, the class that
    has more samples is called the “majority class”, and the class that has fewer samples is called the “minority class”.

    So why does class imbalance even matter? In brief, Machine Learning models often have a harder time predicting the minority class because there are fewer examples of this class to learn from. In
    other words, your model won’t be able to learn as much about the patterns in the minority class, and thus it may have a hard time differentiating between the classes.
    Keep in mind that some class imbalance is present in most datasets. In the Titanic dataset, for example, the majority class represents 62% of the samples and the minority class represents 38%. There’s even more imbalance in the census dataset, with a split of 76% to 24%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now read in the titanic dataset:
    """)
    return


@app.cell
def _(os, pl):
    cols = [
        'Parch', 'Fare', 'Embarked', 'Sex', 
        'Name', 'Age', 'Cabin', 'SibSp'
        ]

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
    return (y,)


@app.cell
def _(pl, y):
    y.select(pl.col('Survived').value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    now load the census data
    """)
    return


@app.cell
def _(os, pl):
    if not os.path.exists('assets/census_dataset.csv'):
        print("Downloading census dataset...")
        census = pl.read_csv('http://bit.ly/censusdataset')
        census.write_csv('assets/census_dataset.csv')
    else:
        print("Loading census dataset from local file...")
        census = pl.read_csv('assets/census_dataset.csv')

    census_cols = ['workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
    census_X = census.select(pl.col(census_cols))
    census_y = census.select(pl.col('class'))
    return (census_y,)


@app.cell
def _(census_y, pl):
    census_y.select(pl.col('class').value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A small amount of class imbalance (like in the Titanic dataset) tends not to matter, and you should
    just use all of the usual techniques that you’ve learned in this book. But as the amount of class
    imbalance increases, more specialized techniques need to be applied to the problem, and those
    techniques are the focus of this chapter.

    #### 18.2 Preparing the mammography dataset
    This chapter will use a dataset of mammography scans designed to detect the presence of cancer.

    The dataset has 6 features as well as a target
    column called “class”. Each sample is an object of interest that was extracted from a scan, and each object was translated into these features using a computer vision algorithm.

    The class column has two possible values, -1 and 1. -1 means the object did not indicate the presence of cancer, and 1 means the object did indicate the presence of cancer. These labels were assigned by
    a human expert and thus represent the “ground truth” labels for the dataset.
    """)
    return


@app.cell
def _(os, pl):
    if not os.path.exists('assets/scanrecords.csv'):
        print("Downloading scan records dataset...")
        scan = pl.read_csv('http://bit.ly/scanrecords')
        scan.write_csv('assets/scanrecords.csv')
    else:
        print("Loading scan records dataset from local file...")
        scan = pl.read_csv('assets/scanrecords.csv')

    scan.head()
    return (scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    -1 (meaning “non-cancerous”) is the majority class, and 1 (meaning“cancerous”) is the minority class. This is severe class imbalance because the minority class makes up only 2% of the dataset.
    """)
    return


@app.cell
def _(pl, scan):
    scan.select(pl.col('class').value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although sklearn doesn’t require it, we’re going to map the strings in the class column to the integers 0 and 1, in which 0 means “non-cancerous” and 1 means “cancerous”.
    """)
    return


@app.cell
def _(pl, scan):
    scan_1 = (
        scan
        .with_columns(
            pl.col('class')
            .str.strip_chars("'")  # Remove surrounding single quotes from the class column
            .str.to_integer(dtype=pl.Int8) # , strict=False)
        )
    )
    scan_1
    return (scan_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that the dataset is ready, define two objects, scan_X and scan_y, to represent
    the feature matrix and the target.
    """)
    return


@app.cell
def _(scan_1):
    scan_X = scan_1.drop('class')
    scan_y = scan_1.select('class')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 18.3 Evaluating a model with a train/test split
    #Break on page 262, where 18.2 ends and 18.3 begins
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
