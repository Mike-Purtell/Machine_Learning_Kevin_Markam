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
    import polars.selectors as cs
    import numpy as np

    import os

    from sklearn.preprocessing import StandardScaler

    # from sklearn.feature_selection import SelectPercentile, chi2

    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer    
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    # from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.feature_selection import SelectFromModel
    # from sklearn.ensemble import ExtraTreesClassifier
    # from sklearn.feature_selection import RFE
    return (
        CountVectorizer,
        LogisticRegression,
        OneHotEncoder,
        SimpleImputer,
        StandardScaler,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 14:  Feature standardization
    - Finished Saturday September 19, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - TBD

    #### 14.1 Standardizing numerical features
    Some Machine Learning models benefit from feature standardization. That’s because the objective function of those models assumes that all features are centered around zero and have a variance of the same order of magnitude. If that assumption is incorrect, a given feature might dominate the objective function and the model won’t be able to learn from all of the features, thus reducing its performance.

    This chapter will experiment with standardizing features to see if that improves our model
    performance.

    We’ll start with the most common approach, using StandardScaler and only standardize features that were originally numerical. We’ll import it from the preprocessing module and create an instance called scaler using the default parameters. For each feature, it will subtract the mean and divide by the standard deviation, which centers the data around zero and scales it.
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
    return


@app.cell
def _(StandardScaler):
    scaler = StandardScaler()
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


@app.cell
def _():
    return


@app.cell
def _():
    params = {
        'columntransformer__countvectorizer__ngram_range': [(1, 1), (1, 2)],
        'columntransformer__pipeline__onehotencoder__drop': [None, 'first'],
        'columntransformer__simpleimputer__add_indicator': [True, False],
        'logisticregression__l1_ratio': [0.0, 1.0],  # 0.0 is L2 (Ridge), 1.0 is L1 (Lasso)
        'logisticregression__C': [0.1, 1, 10] 
    }
    return


if __name__ == "__main__":
    app.run()
