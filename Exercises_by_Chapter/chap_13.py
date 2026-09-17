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

    from sklearn.feature_selection import SelectPercentile, chi2

    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer    
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="`sklearn.utils.parallel.delayed` should be used"
    )
    return (
        CountVectorizer,
        GridSearchCV,
        LogisticRegression,
        OneHotEncoder,
        SelectPercentile,
        SimpleImputer,
        chi2,
        cross_val_score,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 13:  Feature Selection
    - Finished Wednesday September 16, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - TBD

    #### 13.1 Introduction to feature selection
    Feature selection is the process of removing uninformative features from your model. These are features that are not helping your model to make better predictions. In other words, uninformative features are adding “noise” to your model, rather than “signal”.

    There are a few reasons you might want to add feature selection to your workflow:

    - Model accuracy can often be improved by removing uninformative features.
    - Models are generally easier to interpret when they include fewer features.
    - When you have fewer features, models will take less time to train and it may cost less to gather
    the data that is required to train them.

    There are many valid methods for feature selection, including human intuition, domain knowledge, and data exploration. In this chapter, we’re going to do feature selection using automated methods that we can include in our Pipeline.

    There are three types of automated methods that we’ll cover in this chapter: intrinsic methods, filter methods, and wrapper methods.

    For the purposes of simplicity and training speed, we’ll use our logistic regression Pipeline as the starting point for the next few chapters. However, everything you’re learning could also be applied to the random forest Pipeline or the ensemble Pipeline.


    #### 13.2 Intrinsic methods: L1 regularization
    An intrinsic feature selection method is one in which feature selection happens automatically as part of the model building process. These are also known as implicit methods or embedded methods.
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
    return X, ct, logreg, pipe, y


@app.cell
def _():
    params = {
        'columntransformer__countvectorizer__ngram_range': [(1, 1), (1, 2)],
        'columntransformer__pipeline__onehotencoder__drop': [None, 'first'],
        'columntransformer__simpleimputer__add_indicator': [True, False],
        'logisticregression__l1_ratio': [0.0, 1.0],  # 0.0 is L2 (Ridge), 1.0 is L1 (Lasso)
        'logisticregression__C': [0.1, 1, 10] 
    }

    return (params,)


@app.cell
def _(GridSearchCV, X, params, pipe, y):
    grid = GridSearchCV(pipe, param_grid=params, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X, y.to_series())
    return (grid,)


@app.cell
def _(grid):
    grid.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the C and penalty parameters of logistic regression. The L1 penalty is the type of regularization that was used, and the C value indicates the amount of regularization.

    In general, regularization shrinks model coefficients in order to minimize overfitting to the training data and improve the model’s ability to generalize to new data. One notable aspect of L1 regularization is that as the amount of regularization increases, some coefficients will be shrunk all the way to zero, which means they will be excluded from the model. In other words, L1 regularization automatically does feature selection.

    To see an example of this, let’s take a look at the coefficients of the best model found by grid search, which is stored in the best_estimator_ attribute. Notice that the second coefficient is zero, which means that the L1 regularization caused that feature to be removed from the model. Also notice that there are 3671 features.
    """)
    return


@app.cell
def _(grid):
    grid.best_estimator_.named_steps['logisticregression'].coef_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then check how many of the coefficients are zero (3103). This means that L1 regularization removed those features, leaving only 568 features. As the amount of regularization increases, so does the number coefficients that have been shrunk to zero, which removes more features from the model. In the case of logistic regression, you increase the amount of regularization by decreasing the value of C.
    """)
    return


@app.cell
def _(grid):
    sum(grid.best_estimator_.named_steps['logisticregression'].coef_[0] == 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets compare this to our logistic regression Pipeline that was not tuned buy grid search. Notice that is uses L2 regularization. Notice penalty shows as deprecated. In the book using older sklearn, penalty is 'l2'
    """)
    return


@app.cell
def _(pipe):
    pipe.named_steps['logisticregression'].get_params()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Although L2 regularization does shrink coefficients, we can confirm that it does not shrink them all the way to zero, and thus it does not perform feature selection.
    """)
    return


@app.cell
def _(pipe):
    sum(pipe.named_steps['logisticregression'].coef_[0] == 0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Keep in mind that although L1 regularization produced a better-performing model, that will not always be the case. Best to try both types of regularization and see which one works better.

    Here are some advantages and disadvantages of intrinsic feature selection methods:
    - Speed and simplicity: Since feature selection is implicitly performed during model fitting, no additional feature selection process needs to be added to the workflow. This saves a lot of computational time.
    - Disadvantage: it’s model-dependent: The model that is best for your particular problem may not perform intrinsic feature selection.

    #### 13.3 Filter methods: Statistical test-based scoring
    The next type of feature selection we’ll cover is filter methods. Here’s how they work:
    - Every single feature is scored to quantify its potential relationship with the target column.
    - Features are ranked by their scores, and only the top scoring features are provided to the model.

    Thus, they’re called filter methods because they filter out what they believe to be the least informative features and then pass on the more informative features to the model.
    As you’ll see in this section, filter methods vary in terms of the processes they use to score the features.

    Our starting point for this section will be the logistic regression Pipeline that has not been tuned by grid search. The reason for this is because we want to tune all of the Pipeline steps simultaneously, rather than tuning the transformers and model first and then adding feature selection.

    In other words, the presence of a feature selection process may alter the optimal parameters for
    the transformers and the model, and thus we need to tune all three steps at once. Right now it’s a
    two-step Pipeline, but there will be three steps once we add feature selection to the Pipeline.
    """)
    return


@app.cell
def _(pipe):
    pipe
    return


@app.cell
def _(X, cross_val_score, pipe, y):
    cross_val_score(
        pipe,
        X,
        y.to_series(),
        cv=5,
        scoring='accuracy',
    ).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first filter method we’ll use is SelectPercentile. SelectPercentile scores features using univariate statistical tests:
    - You specify a statistical test, and it uses that test to score each feature independently.
    - Then, it passes on to the model a certain percentage (that you specify) of the best scoring features.

    Thus, the assumption behind SelectPercentile is that a statistical test can assess the strength of the relationship between a feature and the target, and that if a feature appears to be independent of the target, then it is uninformative for the purpose of classification.

    Let’s see how SelectPercentile works. After importing SelectPercentile and chi2 from the feature_selection module, we’ll create an instance of SelectPercentile called selection.

    First, we’ll pass it the statistical test. In this case we’re using chi-squared, but other tests are available in scikit-learn.

    Then, we’ll pass it the percentile. We’re arbitrarily using 50 to keep 50% of the features, but this is a parameter you should tune. And to be clear, lower values for this parameter keep fewer features, so a value of 10 would only keep 10% percent of the features.
    """)
    return


@app.cell
def _(SelectPercentile, chi2, ct, logreg, make_pipeline):
    selection = SelectPercentile(chi2, percentile=50)
    fs_pipe = make_pipeline(ct, selection, logreg)
    fs_pipe
    return (fs_pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because we’ve included feature selection within the Pipeline, we can continue to cross-validate the entire process to see the impact of feature selection on model accuracy. When we run cross-validation on the new Pipeline, the score has improved to 0.819.
    """)
    return


@app.cell
def _(X, cross_val_score, fs_pipe, y):
    cross_val_score(
        fs_pipe,
        X,
        y.to_series(),
        cv=5,
        scoring='accuracy',
    ).mean()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It’s worth noting that there’s an alternative to SelectPercentile called SelectKBest. SelectKBest is nearly identical, except that you specify a number of features to keep rather than a percentage.

    SelectPercentile vs SelectKBest:
    - SelectPercentile: Specify percentage of features to keep
    - SelectKBest: Specify number of features to keep
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 13.4 Filter methods: Model-based scoring

    # BREAK ON PAGE 200, section 13.4
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
