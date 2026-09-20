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
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.model_selection import GridSearchCV


    return (
        CountVectorizer,
        FunctionTransformer,
        LogisticRegression,
        OneHotEncoder,
        SimpleImputer,
        cross_val_score,
        cs,
        make_column_transformer,
        make_pipeline,
        np,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 15:  Feature engineering with custom transformers
    - Finished Sunday September 20, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - TBD

    #### 15.1 Why not use dataframe tools for feature engineering?
    Let’s say that you need custom features for your model. You
    believe your model could learn more from a particular feature if it was represented in a different way or combined with another feature.

    Often, feature engineering is done using dataframe tools on the original dataset, and then passing the updated dataset to sklearn. However, you can actually do feature engineering within scikit-learn using
    custom transformers.

    It’s more work to do feature engineering within sklearn, but it means that all of your data transformations can be included in a Pipeline, with considerable benefits. The transformations can be tuned using a grid search, can be applied to new data without any
    extra work, and when done correctly, there’s no possibility of data leakage.
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
    cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age', 'Cabin',
    'SibSp']

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
    return X, X_new, df, imp, imp_constant, imp_ohe, logreg, vect, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 15.2: Transformer 1: Rounding numerical values
    For the next several sections, will only use the first 10 rows of df
    """)
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Assume Fare would be a better feature if it were rounded it up to the next integer.
    """)
    return


@app.cell
def _(df, pl):
    df.select(pl.col('Fare').ceil()).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to do this transformation within sklearn, we add a ceil function in a sklearn transformer using the FunctionTransformer class.
    We simply pass the ceil function to FunctionTransformer, and it returns a transformer object, which we’ll call ceiling.
    """)
    return


@app.cell
def _(FunctionTransformer, np):
    ceiling = FunctionTransformer(np.ceil)
    return (ceiling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because ceiling is a transformer, you can pass Fare to its fit_transform method, which performs
    the same transformation as before. This is the simplest example of feature engineering within
    scikit-learn.
    """)
    return


@app.cell
def _(ceiling, df, pl):
    ceiling.fit_transform(df.select(pl.col('Fare')).head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Like any transformer, ceiling can be included in a ColumnTransformer. we’ll create a ColumnTransformer instance that only includes the ceiling transformer, and will add more transformers throughout the chapter.

    Finally, we’ll pass df to its fit_transform method, which confirms that it works. As we’ve seen, ColumnTransformer always outputs a NumPy array or a sparse matrix. In this case, the output is a NumPy array.
    """)
    return


@app.cell
def _(ceiling, df, make_column_transformer):
    ct_1 = make_column_transformer(
        (ceiling, ['Fare'])
    )
    ct_1.fit_transform(df.head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 15.2 Clipping numerical values
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the 2nd transformation, assume Age would be a better feature if limited to the range 5 to 60. Round up all ages under 5 to 5, and round down all ages over 60 to 60. We can do this using NumPy’s clip function. Note that it has two required arguments, a_min and a_max, which define the limits.

    In this case, the only value that changed was row 7, in which 2.0 became 5.0.
    """)
    return


@app.cell
def _(df, np, pl):
    np.clip(df.select(pl.col('Age')).head(10).to_series(), a_min=5, a_max=60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Convert NumPy’s clip function to a transformer called clip. We need to pass the a_min and a_max arguments to the FunctionTransformer’s kw_args parameter.
    """)
    return


@app.cell
def _(FunctionTransformer, df, np, pl):
    clip = FunctionTransformer(np.clip, kw_args={'a_min':5, 'a_max':60})
    clip.fit_transform(df.select(pl.col('Age')).head(10))
    return (clip,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now add the clip transformer to the ColumnTransformer, and confirm that the ColumnTransformer works as expected.
    """)
    return


@app.cell
def _(ceiling, clip, df, make_column_transformer):
    ct_2 = make_column_transformer(
        (ceiling, ['Fare']),
        (clip, ['Age']),
    )
    ct_2.fit_transform(df.head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 15.4 Transformer 3: extracting string values
    For our third transformation, assume that the first letter of Cabin indicates the deck they were staying on, which we believe might be predictive. Consider using string slice methods to extract the first character.
    """)
    return


@app.cell
def _(df, pl):
    print(df.select(pl.col('Cabin').str.slice(0, 1)).head(10).shape)
    return


@app.cell
def _(np):
    def first_letter(df):
        arr = np.asarray(df, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        values = []
        for value in arr[:, 0]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                values.append(None)
            else:
                values.append(str(value)[0])

        return np.asarray(values, dtype=object).reshape(-1, 1)

    return (first_letter,)


@app.cell
def _(FunctionTransformer, first_letter):
    letter = FunctionTransformer(first_letter)
    return (letter,)


@app.cell
def _(ceiling, clip, df, letter, make_column_transformer):
    ct_3 = make_column_transformer(
        (ceiling, ['Fare']),
        (clip, ['Age']),
        (letter, ['Cabin']),
    )
    ct_3.fit_transform(df.head(10))  
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 15.5 Rules for transformer functions
    We talked about input and output shapes. Here are summary rules for functions to be used in a ColumnTransformer:
    - best that your function accept 2D input, ie multiple columns in the ColumnTransformer, good for scalabilty.
    - your function is required to return 2D output for use in a
    ColumnTransformer. If your function produces a 1D array, reshape to 2D before returning it.

    #### 15.6 Transformer 4: Combining 2 feature
    For the fourth transformation, assume a passenger’s total number of family members aboard, SibSp plus Parch, is more predictive than either feature individually.
    To create this feature, we can use the DataFrame’s sum method over the 1 axis. However, this outputs
    a 1D object, which will not work with a ColumnTransformer.
    """)
    return


@app.cell
def _(df, pl):
    print(
        df
        .select(pl.col('SibSp', 'Parch'))
        .with_columns(FAMILY_SIZE = pl.col('SibSp') + pl.col('Parch'))
        .select(pl.col('FAMILY_SIZE'))
        .head(10)
    )
    return


@app.cell
def _(df, np, pl):
    def sum_cols(df):
        arr = np.asarray(df, dtype=float)
        return arr.sum(axis=1, keepdims=True)

    sum_cols(df.select(pl.col('SibSp', 'Parch')).head(10))
    return (sum_cols,)


@app.cell
def _(FunctionTransformer, sum_cols):
    total = FunctionTransformer(sum_cols)
    return (total,)


@app.cell
def _(ceiling, clip, df, letter, make_column_transformer, total):
    ct_4 = make_column_transformer(
        (ceiling, ['Fare']),
        (clip, ['Age']),
        (letter, ['Cabin']),
        (total, ['SibSp', 'Parch']),
    )
    ct_4.fit_transform(df.head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 15.7: Revising the transformers
    We have built four custom transformers and tested them out on 10 rows by using .head(10) Now, lets apply them to our entire dataset, along with all of our other transformations.

    Let’s handle issue number two, which is the missing values in Fare and Age. Remember we used a Pipeline to combine imputation and one-hot encoding for Embarked and Sex. We’ll do something similar for Fare and Age:
    - For Fare, we’ll create a Pipeline called imp_ceiling that does imputation before ceiling.
    - For Age, we’ll create a Pipeline called imp_clip that does imputation before clipping.
    """)
    return


@app.cell
def _(ceiling, clip, imp, make_pipeline):
    imp_ceiling = make_pipeline(imp, ceiling)
    imp_clip = make_pipeline(imp, clip)
    return imp_ceiling, imp_clip


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now tackle issue number three, which is the most complicated.
    Will slice the first letter of Cabin and then take the value_counts
    of the result to see why:
    - It contains missing values, so imputation will be required.
    - Letters are strings, so one-hot encoding will be required.
    """)
    return


@app.cell
def _(X, pl):
    X.select(
        pl.col('Cabin')
        .str.slice(0, 1)
        .value_counts()
        # .sort("count", descending=True)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that the G and T categories are quite rare, which can cause problems with cross-validation.
    For any rare category, it’s possible for all values of that category to show up in the same testing fold during cross-validation.
    If that happens, the rare category won’t be learned by the OneHotEncoder
    during the fit step, and will be treated as an unknown category during the transform step. By default, the OneHotEncoder will error when it encounters an unknown category, and thus cross-validation will also throw an error.

    Although the problem is complicated, the solution is simple: We need to set the handle_unknown parameter to 'ignore'.
    """)
    return


@app.cell
def _(OneHotEncoder):
    ohe_ignore = OneHotEncoder(handle_unknown='ignore')
    return (ohe_ignore,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To resolve all of these issues with the Cabin column, we’ll create a three-step Pipeline:
    - Step 1 is the letter transformer, which extracts the first letter.
    - Step 2 is imp_constant, which imputes the constant value “missing”.
    - Step 3 is ohe_ignore, which one-hot encodes the results.
    """)
    return


@app.cell
def _(
    FunctionTransformer,
    first_letter,
    imp_constant,
    make_pipeline,
    ohe_ignore,
):
    def first_letter_all(df):
        return first_letter(df)

    letter_all = FunctionTransformer(first_letter_all)
    letter_imp_ohe = make_pipeline(letter_all, imp_constant, ohe_ignore)
    return (letter_imp_ohe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that the three big issues in this lesson have been resolved, we’re finally ready to update our primary ColumnTransformer:
    - Embarked, Sex, and Name are transformed exactly as they were previously.
    - Fare will use imp_ceiling instead of imp.
    - Age will use imp_clip instead of imp.
    - Cabin, will use letter_imp_ohe.
    - SibSp and Parch will use total.
    """)
    return


@app.cell
def _(
    imp_ceiling,
    imp_clip,
    imp_ohe,
    letter_imp_ohe,
    make_column_transformer,
    total,
    vect,
):
    ct_5 = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp_ceiling, ['Fare']),
        (imp_clip, ['Age']),
        (letter_imp_ohe, ['Cabin']),
        (total, ['SibSp', 'Parch']),
    )
    return (ct_5,)


@app.cell
def _(X, ct_5):
    ct_5.fit_transform(X)
    return


@app.cell
def _(ct_5, logreg, make_pipeline):
    pipe_5 = make_pipeline(ct_5, logreg)
    pipe_5
    return (pipe_5,)


@app.cell
def _(mo):
    mo.md(r"""
    When we cross-validate the Pipeline, the accuracy is 0.827, which is higher than our baseline accuracy of 0.811. And it’s very likely that its accuracy could be further improved through hyperparameter tuning.
    """)
    return


@app.cell
def _(X, cross_val_score, pipe_5, y):
    cross_val_score(
        pipe_5,
        X,
        y.to_series(),
        cv=5,
        scoring='accuracy',
    ).mean()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Finally, we’ll fit the Pipeline to X and y and use it to make predictions for X_new.
    """)
    return


@app.cell
def _(X, X_new, pipe_5, y):
    pipe_5.fit(X, y.to_series())
    pipe_5.predict(X_new)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### 15.8 How do I fix incorrect data types within a pipeline?
    Let’s say that you have the following DataFrame.
    """)
    return


@app.cell
def _(pl):
    demo = pl.DataFrame({
        'A': ['10', '20', '30'],
        'B': ['40', '50', None],
        'C': [70, 80, 90],
        'D': ['x', 'y', 'z']
    })
    demo
    return (demo,)


@app.cell
def _(mo):
    mo.md(r"""
    It may look like the first three columns are all integers, but columns A and B are actually object columns because the numbers are being stored as strings, which is a common problem in real datasets.

    These data types need to be fixed in order for a sklearn model to understand them.
    """)
    return


@app.cell
def _(demo, pl):
    demo_fixed = (
        demo
        .with_columns(
            pl.col(['A', 'B', 'C']).cast(pl.UInt8),
        )
    )
    demo_fixed
    return


@app.cell
def _(mo):
    mo.md(r"""
    To incorporate this into a sklearn Pipeline, define a custom function called make_integer that converts DataFrame columns to integers.
    """)
    return


@app.cell
def _(cs, pl):
    def make_integer(df):
        return df.with_columns(
            cs.all().cast(pl.UInt8),
        )

    return (make_integer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The number 60 in column B has been replaced with an empty string. The pipeline handles that gracefully.
    """)
    return


@app.cell
def _(FunctionTransformer, demo, make_integer):
    integer = FunctionTransformer(make_integer)
    integer.fit_transform(demo[['A', 'B']])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Break on page 236, at the end of 15.8, start of 15.9
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
