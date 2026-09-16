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
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

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
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        SimpleImputer,
        cross_val_score,
        cs,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 12:  Ensembling multiple models
    - Started Tuesday September 15, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - Random forest runs very slowly, this notebook takes over 10 minutes to run

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
    return X, ct, logreg, pipe, y


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
    return rf, rf_pipe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cross-validate to generate a baseline accuracy, which is 0.811. This is nearly identical to the baseline accuracy of our logistic regression Pipeline. It’s likely that we can improve it through hyperparameter tuning.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 11.2 Tuning random forests with randomized search
    When tuning random forests, will tune the same transformer parameters used in chapter 10, but will use different parameters for the model. It’s important to tune the transformers and the model at the same time. It may turn out that the best data transformations for a random forest model are different than the best data transformations for a logistic regression model. Rather than typing out a parameter dictionary from scratch, we can start by creating a copy of the params dictionary called rf_params.
    """)
    return


@app.cell
def _():
    rf_params = {
        'columntransformer__pipeline__onehotencoder__drop': [None, 'first'],
        'columntransformer__countvectorizer__ngram_range': [(1, 1), (1, 2)],
        'columntransformer__simpleimputer__add_indicator': [False, True],
    }
    rf_params
    return (rf_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Random forests, with many tuneable parameters, create a computationally expensive grid search when trying to tune all of them. Random forests are comparatively slower to train than logistic regression. Best to use a two-step approach:
    - Randomized search: Test a variety of parameters and values, then examine the results for trends
    - Grid search: Use an optimized set of parameters and values based on what you learned from step 1

    Confirm the Pipeline step name is randomforestclassifier
    """)
    return


@app.cell
def _(rf_pipe):
    rf_pipe.named_steps.keys()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start with four parameters from RandomForestClassifier
    """)
    return


@app.cell
def _(rf_params):
    rf_params['randomforestclassifier__n_estimators'] = [100, 300, 500, 700]
    rf_params['randomforestclassifier__min_samples_leaf'] = [1, 2, 3]
    rf_params['randomforestclassifier__max_features'] = ['sqrt', None]
    rf_params['randomforestclassifier__bootstrap'] = [True, False]
    rf_params
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create an instance of RandomizedSearchCV with rf_pipe and rf_params objects.
    """)
    return


@app.cell
def _(RandomizedSearchCV, X, rf_params, rf_pipe, y):
    rf_rand = RandomizedSearchCV(
        rf_pipe,
        rf_params,
        cv=5,
        scoring='accuracy',
        n_iter=10,  # 100,   changed from 100 to 10 to speed up
        random_state=1,
        n_jobs=-1
    )
    rf_rand.fit(X, y.to_series())
    return (rf_rand,)


@app.cell
def _(rf_rand):
    # notice best score if 0.825, better than our baseline score of 0.811
    rf_rand.best_score_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 11.3 Further tuning with grid search
    Look at top 20 results
    """)
    return


@app.cell
def _(cs, pl, rf_rand):
    results = (
        pl.DataFrame(rf_rand.cv_results_)
        .select(
            cs.starts_with('param_'),
            cs.starts_with('mean_test'),
            cs.starts_with('rank')
        )
        .rename(
            {
                col: col.split('__')[-1] 
                for col in rf_rand.cv_results_.keys() 
                if '__' in col}
        )
        .sort('rank_test_score').head(20)
    )
    results
    return


@app.cell
def _(rf_params):
    rf_params['randomforestclassifier__n_estimators'] = [300, 500, 700, 900]
    rf_params['randomforestclassifier__min_samples_leaf'] = [2, 3, 4, 5]
    rf_params['randomforestclassifier__max_features'] = [None]
    rf_params['randomforestclassifier__bootstrap'] = [True]
    rf_params
    return


@app.cell
def _(GridSearchCV, X, rf_params, rf_pipe, y):
    rf_grid = GridSearchCV(
        rf_pipe,
        rf_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    rf_grid.fit(X, y.to_series())
    return (rf_grid,)


@app.cell
def _(rf_grid):
    rf_grid.best_score_
    return


@app.cell
def _(rf_grid):
    rf_grid.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 11.4 How to tune two models with single grid search
    We set up two separate Pipelines called pipe and rf_pipe, each used by a different model. That made it easy to grid search each Pipeline with its own parameters. You can tune two different models using a single grid search:

    - create a new Pipeline using the Pipeline class instead of the make_pipeline function.

    - create a new parameter dictionary called params1. For the simplicity of this example,
    we’re only going to tune one parameter from the preprocessor step and two parameters from the
    classifier step.

    Additionally, we’re going to add one more entry to the dictionary to indicate that the classifier we
    want to use with this parameter set is logistic regression. Notice that this is a logistic regression
    object, not a string, and also notice that we put it in brackets to make it a list. This might seem
    strange, but it will make more sense momentarily.


    The reason we’re doing this is so that we can provide custom names for the steps. In this case, we’ll call the Pipeline object both_pipe, and we’ll call the step names 'preprocessor' and 'classifier'. We’ll set the classifier to be logistic regression, though this is just a placeholder as you’ll see shortly.
    """)
    return


@app.cell
def _(logreg):
    params1 = {}
    params1['preprocessor__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
    params1['classifier__penalty'] = ['l1', 'l2']
    params1['classifier__C'] = [0.1, 1, 10]
    params1['classifier'] = [logreg]
    params1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create a parameter dictionary called params2. Tune one parameter from the preprocessor step and two parameters from the classifier step. Notice that the classifier parameters are random forest parameters, not logistic regression parameters. Add one more entry to the dictionary to indicate that the classifier we want to use with this parameter set is random forests. During the grid search, this will override the logistic regression classifier we specified when creating the Pipeline.
    """)
    return


@app.cell
def _(logreg):
    both_params1 = {}
    both_params1['preprocessor__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
    both_params1['classifier__penalty'] = ['l1', 'l2']
    both_params1['classifier__C'] = [0.1, 1, 10]
    both_params1['classifier'] = [logreg]
    both_params1
    return (both_params1,)


@app.cell
def _(rf):
    both_params2 = {}
    both_params2['preprocessor__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
    both_params2['classifier__n_estimators'] = [300, 500]
    both_params2['classifier__min_samples_leaf'] = [3, 4]
    both_params2['classifier'] = [rf]
    both_params2
    return (both_params2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create a list called both_params that includes both of these parameter sets.
    """)
    return


@app.cell
def _(Pipeline, both_params1, both_params2, ct, logreg):
    both_pipe = Pipeline([('preprocessor', ct), ('classifier', logreg)])
    both_params = [both_params1, both_params2]
    both_params
    return both_params, both_pipe


@app.cell
def _(GridSearchCV, X, both_params, both_pipe, y):
    both_grid = GridSearchCV(
        both_pipe, 
        both_params, 
        cv=5,
        scoring='accuracy', 
        n_jobs=-1
    )
    both_grid.fit(X, y.to_series())
    return (both_grid,)


@app.cell
def _(both_grid):
    both_grid.best_params_
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 11.5 How to tune tow models with a single randomized search
    Starting in scikit-learn version 0.22, RandomizedSearchCV can search multiple parameter dictionaries.This enables randomized searching of multiple models (similar to grid search with multiple models).This example is passed both_pipe and both_params to RandomizedSearchCV,  and run for 10 iterations.
    """)
    return


@app.cell
def _(RandomizedSearchCV, both_params, both_pipe):
    both_rand = RandomizedSearchCV(
        both_pipe,
        both_params,
        cv=5,
        scoring='accuracy',
        n_iter=10,
        random_state=1,
        n_jobs=-1
    )

    return (both_rand,)


@app.cell
def _(X, both_rand, y):
    both_rand.fit(X, y.to_series())
    return


app._unparsable_cell(
    r"""
    # results = (pd.DataFrame(both_rand.cv_results_)
    # .filter(regex='param_.+_|mean_test'))
    # results.columns = results.columns.str.split('__').str[-1]
    # results

    results = (
        pl.DataFrame(both_rand.cv_results_, strict=False)
        .select(cs.contains('param_.+_|mean_test')
        .rename({col: col.split("__")[-1] for col in result_cols})
    )

    print(results)

    """,
    name="_"
)


@app.cell
def _(both_rand, pl):
    pl.DataFrame(both_rand.cv_results_, strict=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
