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
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    # from sklearn.impute import SimpleImputer
    # from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    # from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    # from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score


    return (
        LogisticRegression,
        OneHotEncoder,
        OrdinalEncoder,
        cross_val_score,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 17:  High-Cardinality categorical features
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


    #### 17.1 Recap of nominal and ordinal features
    Two types of categorical features have been covered in the book:
    - Nominal features have categories that are unordered, such as Embarked and Sex.
    - Ordinal features have categories with an inherent logical ordering, such as Pclass.

    Advice given so far for encoding nominal and ordinal features:
    - Nominal features: use OneHotEncoder. It will add one column for each
    category.
    - Ordinal features:  these are already encoded numerically, leave them as-is.
    - Ordinal features encoded as strings: use an OrdinalEncoder. It will add a single column with the category ordering that you define.

    OneHotEncoder is the preferred approach for a nominal feature, using Embarked as an example.

    Embarked has 3 categories, so OneHotEncoder adds 3 features. From each of the 3 features,
    the model can learn the relationship between the target value and whether or not a given passenger embarked at that port. For example, the model might learn from the first feature that passengers  who embarked at C (Cherbourg France) have a higher survival rate than passengers who didn’t embark at C.

    If you were to instead use OrdinalEncoder with Embarked, it would output 1 feature. This is
    problematic because it would imply an ordering of the categories that doesn’t inherently exist. For example, if passengers who embarked at C (Cherbourg, France) and S (Southampton, England) had high survival rates, and passengers who embarked at Q (Queenstown, Ireland) had low survival rates, there would be no way for a linear model to learn this relationship.

    This chapter explores whether this advice still holds for high-cardinality
    categorical features, which are categorical features with many unique values.

    #### 17.2 Preparing the cencus dataset
    We’ll use a dataset of US census data from 1994. First,  read the dataset into a new DataFrame called census.
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

    census
    return (census,)


@app.cell
def _(census, pl):
    census.select(pl.col(pl.Utf8)).unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can see from the previous cell that education, occupation, and native-country all have more than 10 unique values. There’s no hard rule to define a high-cardinality feature, but all 3 of these could be considered high-cardinality because they each have many unique values.

    You can’t tell from this display, but the 8 features are all nominal features, with the exception of
    education since it does have a logical ordering. However, we’re going to be treating education as
    nominal for this experiment.

    The column labeled “class” is actually our target. This column indicates whether the person has an income of more or less than $50,000 a year.
    We can view the class proportions by normalizing the output of value_counts.
    """)
    return


@app.cell
def _(census, pl):
    census.select(pl.col('class').value_counts())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can view the class proportions by normalizing the output of value_counts.
    """)
    return


@app.cell
def _(census, pl):
    census.select(pl.col('class').value_counts(normalize=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When defining our X DataFrame, which I’m calling census_X, we’re only going to use the 8 categorical
    columns, which I’ve listed out manually. And we’ll use class as our y Series, which I’m calling
    census_y.
    """)
    return


@app.cell
def _(census, pl):
    census_cols = ['workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
    census_X = census.select(pl.col(census_cols))
    census_y = census.select(pl.col('class'))
    return census_X, census_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 17.3 Setting up the encoders
    We will test the effectiveness of OneHotEncoder and OrdinalEncoder with these 8 features. For this experiment, we would normally just create instances using the default arguments.
    """)
    return


@app.cell
def _(OneHotEncoder, OrdinalEncoder):
    ohe = OneHotEncoder()
    oe = OrdinalEncoder()
    return (oe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that we created an instance of OrdinalEncoder without defining the category ordering. This is because we’re treating all of the features as nominal, and nominal features have no logical ordering.
    As a result, OrdinalEncoder would simply learn the categories for each feature in alphabetical order,
    which we can confirm by fitting the OrdinalEncoder and checking the categories_ attribute.
    """)
    return


@app.cell
def _(census_X, oe):
    oe.fit(census_X).categories_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That being said, we will have a problem with encoding due to our highest
    cardinality feature, native-country. Let’s see why.
    """)
    return


@app.cell
def _(census_X, pl):
    (
        census_X.select(
            pl.col('native-country')
            # .groupby('native-country').agg(pl.len())
            #.value_counts()
        )
        .group_by('native-country').agg(pl.len().alias('count'))
        .sort('count', descending=True)

    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One of the categories (Holand-Netherlands) appears only once in the dataset. Rare category values can cause problems with cross-validation, and will definitely create a problem because that sample is guaranteed to appear in the test fold but not in a training fold during one of the runs of cross-validation. That will cause an error
    for both OneHotEncoder and OrdinalEncoder.
    In the case of OneHotEncoder, the solution is simply to set the handle_unknown parameter to 'ignore'.
    """)
    return


@app.cell
def _(OneHotEncoder, OrdinalEncoder):
    ohe_ignore = OneHotEncoder(handle_unknown='ignore')
    oe_ignore = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    return oe_ignore, ohe_ignore


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Starting in sklearn 0.24, OrdinalEncoder also has a handle_unknown parameter that could be used for this situation. Older versions (including the book examples) define the categories in advance for each feature using a list comprehension that iterates through the feature columns and extracts the unique values from each column. This is shown in the book examples with an older version sklearn, but thankfully not needed here. That solution is messy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 17.4: Encoding nominal features for a linear model
    Now that we’ve set up our OneHotEncoder, called ohe_ignore, and our OrdinalEncoder, called oe_cats, let’s see what happens when we pass census_X to fit_transform and then check the shape.

    As expected, the OneHotEncoder creates a lot of columns due to the high-cardinality features, whereas the OrdinalEncoder creates only one column for each of the eight features.
    """)
    return


@app.cell
def _(census_X, ohe_ignore):
    ohe_ignore.fit_transform(census_X).shape
    return


@app.cell
def _(OrdinalEncoder, census_X):
    # book uses OrdinalEncoder(categories=cat) with older sklearn
    oe_cats = OrdinalEncoder()   
    oe_cats.fit_transform(census_X).shape

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let’s actually test the advice that I’ve given, which is that OneHotEncoder should be used for nominal features, to see if this advice still holds for high-cardinality features.

    The simplest method for doing this is to create two Pipelines. One of them uses OneHotEncoder
    and the other uses OrdinalEncoder, and both end in a logistic regression model.
    """)
    return


@app.cell
def _(LogisticRegression, make_pipeline, oe_ignore, ohe_ignore):
    logreg = LogisticRegression(max_iter=1000)
    ohe_logreg = make_pipeline(ohe_ignore, logreg)
    oe_logreg = make_pipeline(oe_ignore, logreg)
    return oe_logreg, ohe_logreg


@app.cell
def _(census_X, census_y, cross_val_score, ohe_logreg):
    cross_val_score(
        ohe_logreg,
        census_X,
        census_y.to_series(),
        cv=5,
        scoring='accuracy'
    ).mean()
    return


@app.cell
def _(census_X, census_y, cross_val_score, oe_logreg):
    cross_val_score(
        oe_logreg,
        census_X,
        census_y.to_series(),
        cv=5,
        scoring='accuracy'
    ).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two Pipelines take around the same amount of time to run, but the accuracy of the
    OneHotEncoder Pipeline is 0.833, which is significantly better than the 0.755 accuracy of the
    OrdinalEncoder Pipeline. This would suggest that at least for a linear model like logistic
    regression, OneHotEncoder should be used for nominal features, even when the features have high
    cardinality.
    """)
    return


@app.cell
def _():
    # BREAK on Page 255, end of 17.4, start of 17.5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
