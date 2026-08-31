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
    # Chapter 4: ColumnTransformer and Pipeline
    Completed on Sunday August 30, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - column tranformer to apply different steps to different columns
    - pipelines: apply same workflow to training data and new data
    - what to do with columns that are not transformed
       - drop
       - passthrough all
       - passthrough some (by list, slice, from numpy)
    - column transformer (ct) get_features_names_out
    - chaining steps with pipelines, example is column transformer, logistic regression
    - pipeline prediction, or pipe.predict(X). On small sample of 10, all 4 females survived, all 6 males did not
    - scale unspecified columns, it scaler = MaxAbsScaler()
    - pipe.fit
    - column tranformer (ct) fit_tranform method does not support polars
    - select columns by datatype, it datetime, category, or multiple types
    - select by pattern name
    - When to use ColumnTransformer or make_column_transformer
    - When to use Pipeline or make_pipeline
    - How to examine steps of a Pipeline
    """)
    return


@app.cell
def _():
    import polars as pl
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import KBinsDiscretizer
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.compose import make_column_selector
    from sklearn.compose import ColumnTransformer



    return (
        ColumnTransformer,
        LogisticRegression,
        MaxAbsScaler,
        OneHotEncoder,
        Pipeline,
        make_column_selector,
        make_column_transformer,
        make_pipeline,
        pl,
        sklearn,
    )


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Problems from Chapter 3
    - Need to stack categorical features next to numerical features
    - Need to apply the same preprocessing to new data
    - Solutions:
      - ColumnTransformer: Apply different preprocessing steps to different columns
      - Pipeline: Apply the same workflow to training data and new data
    """)
    return


@app.cell
def _(df):
    cols = ['Parch', 'Fare', 'Embarked', 'Sex']
    X = df.select(cols)
    X
    return (X,)


@app.cell
def _(OneHotEncoder, X, make_column_transformer):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ct = make_column_transformer(  #tuple
        (ohe,                      # tranformer object
        ['Embarked', 'Sex']        # list of columns to apply tranformer to
        ), 
        remainder='drop'
    )
    ct.fit_transform(X)
    return ct, ohe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Matrix shown above, columns 1-3 are Embarked, columns 4-5 are Sex. Run run with passthrough option to include columns not mentioned in the transformer step
    """)
    return


@app.cell
def _(X, make_column_transformer, ohe):
    ct_passthrough = make_column_transformer(  #tuple
        (ohe,                                  # tranformer object
        ['Embarked', 'Sex']                    # list of columns to apply tranformer to
        ), 
        remainder='passthrough'
    )
    ct_passthrough.fit_transform(X)
    return (ct_passthrough,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Matrix shown above, columns 1-3 are Embarked, columns 4-5 are Sex, 6 is Parch, 7 is fare. Confirm below with get_feature_names
    """)
    return


@app.cell
def _(ct):
    list(ct.get_feature_names_out())
    return


@app.cell
def _(ct_passthrough):
    list(ct_passthrough.get_feature_names_out())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next cell shows how specify list of columns to passthrough without taking all of them
    """)
    return


@app.cell
def _(X, make_column_transformer, ohe):
    ct_revised = make_column_transformer(
        (ohe,  ['Embarked', 'Sex']),          # first tuple - same as before
        ('passthrough', ['Parch', 'Fare'])    # second tuple, passthrough explicit columns
    )
    ct_revised.fit_transform(X)
    return (ct_revised,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.2 Chaining steps with Pipeline
    **Very Important** pipeline building
    """)
    return


@app.cell
def _(LogisticRegression, X, ct_revised, df, make_pipeline):
    logreg = LogisticRegression(solver='liblinear', random_state=1) # from chapter 2
    y = df.select('Survived').to_numpy()
    pipe = make_pipeline(
        ct_revised, 
        logreg
    )
    pipe.fit(X, y)
    return logreg, pipe, y


@app.cell
def _(pl):
    df_new = pl.read_csv('http://bit.ly/MLnewdata', n_rows=10)
    df_new
    return (df_new,)


@app.cell
def _(df_new):
    X_new = df_new.select(['Parch', 'Fare', 'Embarked', 'Sex'])
    X_new
    return (X_new,)


@app.cell
def _(X_new, pipe):
    pipe.predict(X_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pipeline prediction in this case follows two steps:
    1. ColumnTransformer applies the same transformations to X_new
    2. Fitted LogisticRegression model makes predictions on the transformed version of X_new


    Predictions with this dataset of 10 have all 4 females surviving and none of the 6 males surviving
    """)
    return


@app.cell
def _(X_new, df_new, pipe):
    df_new.with_columns(PRED = pipe.predict(X_new)) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.4 How do I drop some columns and passthrough others?
    """)
    return


@app.cell
def _(X, make_column_transformer, ohe):
    # drop the Fare column and passthrough the rest
    ct_4p4 = make_column_transformer(
        (ohe,  ['Embarked', 'Sex']),  # one hot encode Embarked and Sex, makes 5 columns
        ('drop', ['Fare']),           # passthrough Parch, drop Fare, one hot encode Embarked and Sex
        remainder='passthrough'
    )
    ct_4p4.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.5 How do I transform the unspecified columns?
    We know how to drop or pass through the unspecified columns in a ColumnTransformer, but let’s
    pretend we wanted to apply a transformation to all of the unspecified columns. This is actually
    simple to do by passing a transformer to the remainder parameter.

    For example, we might want to scale all of the unspecified columns. One option is MaxAbsScaler,
    which divides each feature by its maximum value and thus scales it to the range -1 to 1. We’ll import
    it from the preprocessing module and then create an instance.

    Then, we can pass the scaler to the remainder parameter.

    When we run the fit_transform method, you can see that the first 5 columns were created from
    Embarked and Sex, and the final column is the scaled version of the Parch column. (We’ll talk more
    about feature scaling in chapter 14.)

    Max parch value of 2 becomes 1, see row 9
    parch value of 1 becomes 0.5, see row 8
    """)
    return


@app.cell
def _(MaxAbsScaler, X, make_column_transformer, ohe):
    scaler = MaxAbsScaler()
    ct_scaler = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        ('drop', ['Fare']),
        remainder=scaler  # divide remainder column (Parch) by its max value
    )
    ct_scaler.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.6 How to select columns from NumPy array
    Start by converting the X and X_new DataFrames into NumPy arrays called X_array and
    X_new_array.
    """)
    return


@app.cell
def _(X, X_new):
    X_array = X.to_numpy()
    X_new_array = X_new.to_numpy()
    return X_array, X_new_array


@app.cell
def _(X_array):
    X_array
    return


@app.cell
def _(X_array, make_column_transformer, ohe):
    _ct = make_column_transformer(
        (ohe, [2, 3]),              # explicitly select columns 2 and 3
        remainder='passthrough')
    _ct.fit_transform(X_array)
    return


@app.cell
def _(X_array, make_column_transformer, ohe):
    _ct = make_column_transformer(
        (ohe, slice(2, 4)),       # select columns 2 and 3 with slice notation
        remainder='passthrough')
    _ct.fit_transform(X_array)
    return


@app.cell
def _(X_array, make_column_transformer, ohe):
    _ct = make_column_transformer(
        (ohe, [False, False, True, True]),  # select columns 2 and 3 with boolean mask
        remainder='passthrough')
    _ct.fit_transform(X_array)
    return


@app.cell
def _(
    X_array,
    X_new_array,
    logreg,
    make_column_transformer,
    make_pipeline,
    ohe,
    y,
):
    # same code as previous cell, but with a pipeline added
    _ct = make_column_transformer(
        (ohe, [False, False, True, True]),  # select columns 2 and 3 with boolean mask
        remainder='passthrough')
    _ct.fit_transform(X_array)

    _pipe = make_pipeline(
        _ct,
        logreg
    )
    _pipe.fit(X_array, list(y.flatten()))
    _pipe.predict(X_new_array)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 4.7 How to select columns by data type
    If there were many more columns, and we wanted to one-hot encode all object
    columns and passthrough all numeric columns without listing them out.
    How would we do that?

    Easiest way is with make_column_selector function, new to scikit-learn version 0.22

    Polars has a cs function that may work, pandas likely has something as well.
    """)
    return


@app.cell
def _(make_column_selector):
    select_object = make_column_selector(dtype_include=object)
    select_number = make_column_selector(dtype_include='number')
    return select_number, select_object


@app.cell
def _(X):
    X
    return


@app.cell
def _(X, make_column_transformer, ohe, select_number, select_object):
    _ct = make_column_transformer(
        (ohe, select_object),
        ('passthrough', select_number))
    _ct.fit_transform(X.to_pandas())     # fit_transform support pandas does not support polars
    return


@app.cell
def _(X, make_column_selector, make_column_transformer, ohe, select_object):
    exclude_object = make_column_selector(dtype_exclude=object)
    _ct = make_column_transformer(
        (ohe, select_object),
        ('passthrough', exclude_object)  # only passthrough numeric columns, exclude object columns
    )
    _ct.fit_transform(X.to_pandas())     # fit_transform support pandas does not support polars
    return


@app.cell
def _(make_column_selector):
    #  other datatype select options
    select_datetime = make_column_selector(dtype_include='datetime')
    select_category = make_column_selector(dtype_include='category')
    return


@app.cell
def _(make_column_selector):
    # select multiple objects
    select_multiple = make_column_selector(dtype_include=[object, 'category'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.8 Select columns by column name pattern
    Consider when there are many columns, and the columns we wanted had the same
    pattern in their names. For example, when desired columns start with the same word.

    make_column_selector function allows column selection by regular expression
    pattern. This example selects columns capital letters E or S.
    """)
    return


@app.cell
def _(X, make_column_selector, make_column_transformer, ohe):
    select_ES = make_column_selector(pattern='E|S')  # looks like regex to me
    _ct = make_column_transformer(
        (ohe, select_ES),
        remainder='passthrough')
    _ct.fit_transform(X.to_pandas())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.9 When to use ColumnTransformer or make_column_transformer
    Which one should **you** use?
    Author prefers make_column_transformer, for code that is easier to read or write.
    Most convenient to use default transformer names, without having to come up with
    a name for each transformer. There are exceptions were defining transformer
    names is useful.

    Note that ColumnTransformer class enables transformer weights. You can emphasize
    the output of some transformers more than others. Use case not clear, but if
    transformer weights are used, you must use the ColumnTransformer class, and
    cannot use make_column_transformer

    || ColumnTransformer|make_column_transformer|
    |--|:--:|:--:|
    Allow custom name|Yes| No|
    Allow transformer weights|Yes| No|
    """)
    return


@app.cell
def _(ColumnTransformer, ohe):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _ct
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.10 When to use Pipeline or make_pipeline
    Which one should **you** use?
    So far Pipelines were created using the make_pipeline function.
    This section shows how to use the Pipeline class, and compare it to
    make_pipeline.
    First import the Pipeline and create an instance.

    When creating an instance, the main difference to notice is we passing in a list of
    tuples to the Pipeline constructor. Each tuple has two elements, the
    to assign to the Pipeline step, and the model or transformer
    to include in the Pipeline.

    The first tuple the preprocessing step with ColumnTransformer, assigned
    the name “preprocessor”. The second tuple is the model building step using
    logistic regression, assigned the name “classifier”. These names show when
    printing out the Pipeline.

    || ColumnTransformer|make_column_transformer|
    |--|:--:|:--:|
    Allow custom name|Yes| No|
    Allow transformer weights|Yes| No|
    """)
    return


@app.cell
def _(ColumnTransformer, Pipeline, logreg, ohe):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )

    _pipe = Pipeline(
        [
            ('preprocessor', _ct),    # step name
            ('classifier', logreg)    # model or transformer
        ]
    )
    print(_pipe)  # more or less matches the book example
    _pipe         # as marimo notebook, interactive object structure is printed in a more readable way
    print(f'{_pipe.named_steps.keys() =}')  # print the named steps of the pipeline

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now make pipeline with make_pipeline function
    """)
    return


@app.cell
def _(ColumnTransformer, logreg, make_pipeline, ohe):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )

    _pipe = make_pipeline(_ct, logreg)
    print(f'{_pipe = }')

    print(f'{_pipe.named_steps.keys() =}')  # print the named steps of the pipeline
    _pipe

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Author prefers make_pipeline for code that is easier to read and write.
    The default step names offer convenience of not having to make up names for each step.
    But custom names provide clarity, especially for grid search of a Pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.11 How to examine steps of a Pipeline
    Sometimes you want to examine the steps of a fitted Pipeline.
    """)
    return


@app.cell
def _(ColumnTransformer, X, logreg, make_pipeline, ohe, y):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _pipe = make_pipeline(_ct, logreg)

    _pipe.fit(X, list(y))
    print(f'{_pipe = }')
    return


@app.cell
def _(ColumnTransformer, X, logreg, make_pipeline, ohe, y):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _pipe = make_pipeline(_ct, logreg)

    _pipe.fit(X, y)
    print(f'{_pipe.named_steps.keys() = }')
    return


@app.cell
def _(ColumnTransformer, X, logreg, make_pipeline, ohe, y):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _pipe = make_pipeline(_ct, logreg)

    _pipe.fit(X, list(y))
    print(f'{_pipe.named_steps["columntransformer"] = }')
    return


@app.cell
def _(ColumnTransformer, X, logreg, make_pipeline, ohe, y):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _pipe = make_pipeline(_ct, logreg)

    _pipe.fit(X, list(y))
    print(f'{list(_pipe.named_steps["logisticregression"].coef_) = }')
    return


@app.cell
def _(ColumnTransformer, X, logreg, make_pipeline, ohe, y):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _pipe = make_pipeline(_ct, logreg)

    _pipe.fit(X, list(y))
    print(f'{list(_pipe.named_steps["columntransformer"].get_feature_names_out()) = }')
    return


@app.cell
def _(ColumnTransformer, X, logreg, make_pipeline, ohe, y):
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _pipe = make_pipeline(_ct, logreg)

    _pipe.fit(X, list(y))
    print(f'{_pipe.named_steps["logisticregression"].coef_ = }')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _ct = ColumnTransformer(
        [('OHE', ohe, ['Embarked', 'Sex']),
        ('pass', 'passthrough', ['Parch', 'Fare'])]
    )
    _pipe = make_pipeline(_ct, logreg)

    _pipe.fit(X, list(y))
    print(f'{list(_pipe[1]].coef_) = }')
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
