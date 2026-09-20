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
    import pickle

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
    from sklearn.preprocessing import PolynomialFeatures


    return (
        CountVectorizer,
        FunctionTransformer,
        LogisticRegression,
        OneHotEncoder,
        PolynomialFeatures,
        SimpleImputer,
        cross_val_score,
        cs,
        make_column_transformer,
        make_pipeline,
        np,
        os,
        pickle,
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

    - Stateless tranformations do not learn any information during the fit step. These include functions like ceiling, clip, letter(gets first letter of a string, and total (horizontal sum of columns))
    - FunctionTransformer can only be used with Stateless transformations

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
    #### 15.9: How do I create features from datetime data?
    Demonstrate how to create date-based features. Read a tiny dataset of reported UFO sightings into a DataFrame.
    """)
    return


@app.cell
def _(pl):
    ufo = (
        pl.read_csv(
            'http://bit.ly/ufosample',
        )
        .with_columns(
            Date = pl.col('Date').str.to_date(format='%m/%d/%Y')
        )
    )
    ufo
    return (ufo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can access properties of the Date column using the dt
    accessor. For example, we can easily access the day of the month using the day attribute.
    """)
    return


@app.cell
def _(pl, ufo):
    ufo.select(pl.col('Date').dt.day())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to use the day of the month as a feature, the first step is to create
    a custom function called day_of_month.
    """)
    return


@app.cell
def _(pl, ufo):
    def day_of_month(df):
        return df.select(pl.col('Date').dt.day())

    day_of_month(ufo)
    return (day_of_month,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now convert the day_of_month function into a transformer called day, and check that it
    works as well.
    """)
    return


@app.cell
def _(FunctionTransformer, day_of_month, ufo):
    day = FunctionTransformer(day_of_month)
    day.fit_transform(ufo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 15.8: How do I create feature interaction?
    When there is an interaction between two or more features, one common technique is to
    create “interaction terms” or “feature interactions” that your model can learn from. This is generally done by multiplying the values of each pair of features and then using those as new features.

    Creating interaction features is useful when the combined impact of a pair of features is different
    from the impact of the features when considered independently. For example, let’s pretend that
    features A and B each have a small positive impact on the target, but when combined, they have a
    much larger positive impact on the target than you would expect. In that case, it would be useful to
    create the interaction feature of A × B.

    Let’s see how we can create feature interactions in sklearn. We’ll assume that we’ve decided to
    create interactions between Fare, SibSp, and Parch. Here are the first three and last three rows of
    each of those features.
    """)
    return


@app.cell
def _(X, pl):
    X.select(pl.col('Fare', 'SibSp', 'Parch'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our first step is to import the PolynomialFeatures class from the preprocessing module, and then
    create an instance called poly. We’ll set the include_bias parameter to False to avoid creating
    a column of ones in the output, and we’ll set the interaction_only parameter to True to avoid
    creating the square of each feature.
    """)
    return


@app.cell
def _(PolynomialFeatures):
    poly = PolynomialFeatures(include_bias=False, interaction_only=True)
    return (poly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When we run the fit_transform method and pass it those three columns, it outputs six columns:
    - First three columns of the output are the original three columns: Fare, SibSp, and Parch.
    - Next three columns are our interaction terms: Fare × SibSp, Fare × Parch, and SibSp ×
    Parch.
    """)
    return


@app.cell
def _(X, pl, poly):
    poly.fit_transform(X.select(pl.col('Fare', 'SibSp', 'Parch')))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we wanted to include these feature interactions in our model, we would simply include poly as
    one of the transformers in our ColumnTransformer.
    One obvious question is: How should you choose which feature interactions to create?
    - Ideally, you would use expert knowledge to guide your decision of which interactions to create.
    - Otherwise explore the data to decide which interactions to create.
    - For small number of features, create all possible interactions, then use feature selection to remove less ones.

    For a large number of features, it’s impractical to create all possible
    interactions. This would increase the risk of false positive feature interactions that appear to have a relationship with the target but could be occurring due to random chance (hallucinations?).

    Tree-based models can learn feature interactions on their own through recursive splitting. When using a tree-based prediction model, then you don’t need to manually create feature interactions.

    Keep in mind that while linear models can’t explicitly learn feature interactions, they can sometimes replace the information supplied by the interaction terms, in which case the
    interactions are unnecessary.

    **You should always evaluate the model with interactions
    against the model without interactions, and only include them if they’re improving the model’s
    performance.**

    #### 15.11 How do I save a Pipeline with custom transformers?
    If you save a Pipeline using pickle or joblib, and the Pipeline includes custom transformers, then
    the saved Pipeline can only be loaded into a new environment if the functions it depends on are
    defined in the new environment.
    For example, let’s import pickle and use it to save our current Pipeline.
    """)
    return


@app.cell
def _(pickle, pipe_5):
    with open('pipe.pickle', 'wb') as f:
        pickle.dump(pipe_5, f)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s pretend that we’re in a brand new environment and want to make predictions for X_new using our saved Pipeline. Because the Pipeline includes custom transformers which use the
    first_letter and sum_cols functions, those two functions need to be defined in the new environment.

    Those functions depend on polars, so polars would also need to be imported into the new environment. With these conditions satisfied, we can load our saved Pipeline into the pipe_from_pickle object.
    """)
    return


@app.cell
def _(pickle):
    with open('pipe.pickle', 'rb') as f_read:
        pipe_from_pickle_read = pickle.load(f_read)
    return (pipe_from_pickle_read,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also need to create the X_new object in our environment.
    """)
    return


@app.cell
def _(pl):
    read_cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age', 'Cabin',
    'SibSp']
    df_new_read = pl.read_csv('http://bit.ly/MLnewdata')
    X_new_read = df_new_read.select(pl.col(read_cols))
    return (X_new_read,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can make predictions using the saved Pipeline
    """)
    return


@app.cell
def _(X_new_read, pipe_from_pickle_read):
    pipe_from_pickle_read.predict(X_new_read)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If this seems too cumbersome, one alternative is to use a Python library called cloudpickle, which extends the functionality of pickle to allow you to save user-defined functions.

    Install and import cloudpickle, then save the Pipeline using cloudpickle instead of pickle. Notice that the cloudpickle code is exactly the same as the pickle code, except you use the dump function from cloudpickle instead of from pickle.

    **skipped the cloudpickle examples**

    #### 15.12: Can FunctionTransformer be used with any transformation?
    FunctionTransformer should only be used with stateless transformations, where the transformation doesn’t learn any information during the fit step.

    All custom transformations in this chapter were stateless: rounding up to the next integer, limiting values to a range,
    extracting the first letter, and adding two columns. They didn’t learn anything about the training data that
    later needed to be applied to testing data. They
    work exactly the same on the testing data regardless of what the
    training data looked like.

    This is in contrast to stateful transformations, which do learn information from the fit step that need to be applied to both training and testing data. We’ve seen many stateful transformations in this
    book:
    - OneHotEncoder learns the categories from the training data, and those same categories need to be applied to the testing data.
    - CountVectorizer learns the vocabulary from the training data, and that vocabulary needs to be used when building the document-term matrix for the testing data.
    - SimpleImputer learns the values to impute from the training data, and those values are applied to the testing data.
    - MaxAbsScaler learns the scale of each feature from the training data, and that scaling is applied to the testing data.

    FunctionTransformer should never be used to implement stateful transformations. Depending on the situation, you would either run into an error or you would silently cause data leakage. In that
    case, you would need to write your own class in order to create a proper transformer.
    """)
    return


if __name__ == "__main__":
    app.run()
