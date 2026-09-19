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

    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer    
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import cross_val_score

    from sklearn.preprocessing import MaxAbsScaler

    from sklearn.model_selection import GridSearchCV


    return (
        CountVectorizer,
        GridSearchCV,
        LogisticRegression,
        MaxAbsScaler,
        OneHotEncoder,
        SimpleImputer,
        StandardScaler,
        cross_val_score,
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

    - Standardization of numerical features is not always necessary or helpful.
    - models that benefit most from standardization are those that consider the distance between features, such as K-Nearest Neighbors and Support Vector Machines.
    - feature standardization does not benefit tree-based models such as random forests.

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
    return X, imp, imp_ohe, logreg, vect, y


@app.cell
def _(
    StandardScaler,
    imp,
    imp_ohe,
    logreg,
    make_column_transformer,
    make_pipeline,
    vect,
):
    scaler = StandardScaler()
    imp_scaler = make_pipeline(imp, scaler)

    ct_imp_scaler = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp_scaler, ['Age', 'Fare', 'Parch']),
    )

    scaler_pipe = make_pipeline(ct_imp_scaler, logreg)
    scaler_pipe

    return (scaler_pipe,)


@app.cell
def _(X, cross_val_score, scaler_pipe, y):
    cross_val_score(
        scaler_pipe,
        X,
        y.to_series(),
        cv=5,
        scoring='accuracy'
    ).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That might be surprising, because regularized linear models often benefit from feature standardization.
    However, our particular logistic regression solver (liblinear) happens to be robust to unscaled data,
    and thus there was no benefit in this case.
    The takeaway here is that you shouldn’t always assume that standardization of numerical features is
    necessary.

    ####  14.2 Standardizing all features
    The previous lesson standardized the numerical features only. Other approach is to
    standardize all features after transformation, even if they were not originally numerical.

    We will add standardization as the second step in the Pipeline, in between the
    ColumnTransformer and the model. But our ColumnTransformer outputs a sparse matrix, and
    StandardScaler would destroy the sparseness by centering the data, likely resulting in a memory
    issue.

    We will use an alternative scaler called MaxAbsScaler. We’ll import it from the
    preprocessing module and create an instance called scaler. MaxAbsScaler divides each feature
    by its maximum value, which scales each feature to the range -1 to 1. Zeros are never changed,
    and thus sparsity is preserved.
    """)
    return


@app.cell
def _(
    MaxAbsScaler,
    imp,
    imp_ohe,
    logreg,
    make_column_transformer,
    make_pipeline,
    vect,
):
    scaler_max_abs = MaxAbsScaler()
    ct_max_abs = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age', 'Fare']),
        ('passthrough', ['Parch'])
    )
    scaler_max_abs_pipe = make_pipeline(ct_max_abs, scaler_max_abs, logreg)
    scaler_max_abs_pipe
    return (scaler_max_abs_pipe,)


@app.cell
def _(X, cross_val_score, scaler_max_abs_pipe, y):
    cross_val_score(
        scaler_max_abs_pipe,
        X,
        y.to_series(),
        cv=5,
        scoring='accuracy'
    ).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cross-validate accuracy is 0.811, exactly the same as our baseline accuracy.
    This result is not surprising, because MaxAbsScaler has no effect on the columns output by
    OneHotEncoder and only a tiny effect on the columns output by CountVectorizer. Our
    approach is mostly just affecting the numerical columns.

    This example doesn’t benefit from standardization, but there are many cases where it will
    help. If you want to experiment with standardization, try out both of the
    approaches used in this chapter and pick the one that works better.

    #### 14.3: How to see what scaling was applied to each feature?
    If you’re interested in seeing what scaling was applied to each feature, you can fit the Pipeline and
    then examine the scale_ attribute of the maxabsscaler step.

    For example, the last three entries in the array correspond to the scaling for Age, Fare, and Parch.
    These are simply the maximum values of Age, Fare, and Parch in X. As a reminder, this is the scaling
    that will be applied to the features in X_new when making predictions.
    """)
    return


@app.cell
def _(X, scaler_max_abs_pipe, y):
    scaler_max_abs_pipe.fit(X, y.to_series())
    scaler_max_abs_pipe.named_steps['maxabsscaler'].scale_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you might expect, the scale_ attribute is also available when using StandardScaler.

    #### 14.4: How to I turn off feature standardization within a grid search?
    Although grid search is usually just used to tune parameter values, you can actually use grid search
    to turn on and off particular Pipeline steps. Thus you could use a grid search to decide whether or
    not feature standardization should be included.
    To demonstrate this, let’s create a small dictionary called scaler_params:
    - The first entry tunes the C parameter of logistic regression. The dictionary key is the step name,
    then two underscores, then the parameter name. The dictionary values are the possible values
    for that parameter.
    - The second entry is different: Rather than tuning the parameter of a Pipeline step, we’re
    tuning the Pipeline step itself. In this case, the dictionary key is simply the step name assigned
    by make_pipeline. The possible values are 'passthrough', which means skip this Pipeline
    step, or a MaxAbsScaler instance, which means keep MaxAbsScaler in the Pipeline.

    We’ll create scaler_grid using the scaler_pipe and scaler_params objects, and then run the grid
    search as normal.
    """)
    return


@app.cell
def _(GridSearchCV, MaxAbsScaler, X, scaler_max_abs_pipe, y):
    scaler_params = {}
    scaler_params['logisticregression__C'] = [0.1, 1, 10]
    scaler_params['maxabsscaler'] = ['passthrough', MaxAbsScaler()]
    scaler_grid = GridSearchCV(
        scaler_max_abs_pipe, 
        scaler_params, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    scaler_grid.fit(X, y.to_series())
    #scaler_grid.best_params_
    return (scaler_grid,)


@app.cell
def _(pl, scaler_grid):
    results_1 = pl.DataFrame(
        {
            "C": scaler_grid.cv_results_["param_logisticregression__C"].tolist(),
            "param_maxabsscaler": [
                "passthrough"
                if value == "passthrough"
                else type(value).__name__
                for value in scaler_grid.cv_results_[
                    "param_maxabsscaler"
                ].tolist()
            ],
            "mean_test_score": scaler_grid.cv_results_["mean_test_score"].tolist(),
            "rank_test_score": scaler_grid.cv_results_["rank_test_score"].tolist(),
        },
        strict=False,
    )

    results_1
    return


@app.cell
def _(scaler_grid):
    scaler_grid.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 14.5: Which models benefit from standardization?
    Feature standardization is most useful for models that considers the distance between
    features, such as K-Nearest Neighbors and Support Vector Machines.

    It also tends to be useful with any models that incorporate regularization, such as linear or logistic
    regression with an L1 or L2 penalty, though we saw earlier in the chapter that this doesn’t apply to all solvers.

    Notably, feature standardization will not benefit any tree-based models such as random forests.
    """)
    return


if __name__ == "__main__":
    app.run()
