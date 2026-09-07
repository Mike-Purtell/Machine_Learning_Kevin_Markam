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
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV

    import polars as pl
    import polars.selectors as cs

    import os

    return (
        CountVectorizer,
        GridSearchCV,
        LogisticRegression,
        OneHotEncoder,
        SimpleImputer,
        cross_val_score,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 10:  Evaluating and tuning a pipeline
    - Started Sunday September 6, 2026  - first page is 137
    - Break on Sunday September 6, at page 143
    - Completed xxxday September y, 2026 - last page is 169 (very big chapter)

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
    #### 10.1 Evaluating a pipeline with cross-validation
    This long chapter is a deep dive into efficient Pipelint tuning for maximum accuracy.
    With full data set in use, will run cross_validation_score on the entire pipeline instead of just running it on the model. Will pass it X and y, specify number of cross-validation folds. Will choose 5 cross-validation folds (only 3 were used when the data size was limited to 10) in order to minimize amount of computation.Finally will specify evaluation metric for classification accuracy.
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
    return X, pipe, y


@app.cell
def _(X, cross_val_score, pipe, y):
    print(cross_val_score(pipe, X, y.to_series(), scoring ='accuracy').mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Steps of 5-fold cross-validation on a Pipeline:
    1. Split data into 5 folds (A, B, C, D, E)
      - ABCD is training set
      - E is testing set
    2. Pipeline is fit on training set
      - ABCD is transformed
      - Model is fit on transformed data
    3. Pipeline makes predictions on testing set
      - E is transformed (using step 2 transformations)
      - Model makes predictions on transformed data
    4. Calculate accuracy of those predictions
    5. Repeat the steps above 4 more times, with a different testing set each time
    6. Calculate the mean of the 5 scores

    Notice that cross_val_score splits the data in step 1 before performing the transformations in steps 2 and 3. As a result, the imputation values for Age and Fare and the vocabulary for CountVectorizer are all computed 5 different times. Each time, these values are computed using the uniqe training set only, and then applied to both the training

    cross_val_score splits the data first before performing the transformations to prevent data leakage. If transformations were performed on the full dataset before splitting there would be  data leakage, since information about the testing set would be “leaked” into the model training process.

    Proper cross-validation:
    - Data is split (step 1) before transformations (steps 2 and 3)
    - Imputation values and vocabulary are computed using training set only
    - Prevents data leakage

    Improper cross-validation:
    - Transformations are performed before data is split
    - Imputation values and vocabulary are computed using full dataset
    - Causes data leakage

    #### 10.2 Tuning a Pipeline with grid search
    With baseline accuracy for our Pipeline known, next step is to tune the hyperparameters for the model and the transformers. Previously we have been using the default parameters for most objects in the Pipeline.  Tuning should result in a more accurate model.

    Keep in mind these terms
    - Hyperparameters: Values that you set, such as C value for logistic regression
    - Parameters: Values learned from the data, such as coefficients of logistic regression model

    Hyperparameter tuning refers to the process of tuning a model or tuning a Pipeline that contains a model and transformers

    Parameter refers to anything passed to a Class, including the C and random_state values passed to the LogisticRegression class, and the strategy value passed to the SimpleImputer class.

    WIll be using GridSearchVD to perform hyperparamter tuning by defining the values to try for each parameter, and letting it cross-validate every possible combination of these values.

    GridSearchCV can tune the entire Pipeline at once, including both the model and the transformers. This has two huge benefits over just tuning a modll.
    - Tunes the model and transformers simultaneously
    - Prevents data leakage

    #### 10.3 Tuning the model

    LogisticRegression tuning parameters:
    - l1_ratio: Type of regularization
      - 1 (L1)
      - 0 (L2, default)
    - C: Amount of regularization
      - 0.1
      - 1 (default)
      - 10

    Get the names of the Pipellne steps from the named_steps attributee
    """)
    return


@app.cell
def _(pipe):
    pipe.named_steps.keys()
    return


@app.cell
def _(mo):
    mo.md(r"""
    To use GridSearchCV, we need a dictionary where each entry represents a parameter and the values we want to try for that parameter

    - The key for each entry is the Pipeline step name, followed by two underscores, followed by the parameter name. Thus the key for the first entry is 'logisticregression__penalty', and the key for the second entry is 'logisticregression__C'. Using two underscores is what allows GridSearchCV to distinguish between the step name and the parameter name. Using a single underscore would be ambiguous, since a step name or parameter name can have an underscore within it.

    - The value for each entry is a list of the values you want to try for that parameter. Thus the value for the first entry is a list of 'l1' and 'l2', and the value for the second entry is a list of 0.1, 1, and 10.

    We’ll create an empty dictionary called params, add these two entries, and then print it out just to
    make sure that it looks correct.
    """)
    return


@app.cell
def _():
    params = {
        'logisticregression__l1_ratio': [0.0, 1.0],
        'logisticregression__C': [0.1, 1, 10],
    }
    params
    return (params,)


@app.cell
def _(GridSearchCV, X, params, pipe, y):
    grid = GridSearchCV(pipe, param_grid=params, cv=5, scoring='accuracy', verbose=0)
    grid.fit(X, y.to_series())
    return (grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results of the grid search are stored in an attribute called cv_results_, which we’ll convert to a DataFrame. We’ll use a filter to only keep the columns we need, and rename the parameter columns to make them easier to read.

    The DataFrame contains 6 rows because cross-validation ran 6 times, which is every possible combination of the 2 values of penalty and the 3 values of C that we specified.
    """)
    return


@app.cell
def _(grid, pl):
    result_cols = pl.DataFrame(grid.cv_results_, strict=False).columns

    df_results = (
        pl.DataFrame(grid.cv_results_, strict=False)
        .select([
            col for col in result_cols
            if col.startswith("param_") or "mean_test" in col or "rank" in col
        ])
        .rename({col: col.split("__")[-1] for col in result_cols})
        .sort("rank_test_score")
    )
    df_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Best accuracy (0.818) happened when C was 10 and penalty was 'l1'. Baseline accuracy with no tuning was 0.811. Accuracy improvement is here seems very small, 0.007, or 0.7% but could be huge in some contexts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 10.4 Tuning the transformers

    #BREAK AT PAGE 143
    pick up again at 10.4, Tuning the transformers
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
