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
    # Chapter 3: Master Machine Learning
    ### Encoding categorical features
    Completed on Saturday August 29, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - learned OneHotEncoding with scikit-learn. It is similar to pandas(get_dummies) or polars(to_dummies) which both perform one-hot-encoding
    - Learned transformer methods, fit, transform, and fit_transform
    - Learned how to perform one hot endoding with multiple features
    - Learned when to use transform instead of fit_transform
    - Learned what to do when testing data includes a new category
    - Learned when to drop one or more of the one-hot encoded categories
    - Studied the difference between OrdinalEncoder and LabelEncoder
    - Learned how to manage numeric features, and trade offs of forcing ordinality by binning
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

    return KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, pl, sklearn


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
    Scikit-learn models expect features to be numeric, so Embarked and Sex can’t be
    passed to a model without numeric endoding. The process used here is one-hot
    encoding or dummy encoding
    """)
    return


@app.cell
def _(df):
    X = df.select(['Parch', 'Fare', 'Embarked', 'Sex'])
    X
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **sparse** parameter of OneHotEncoder was changed to **sparse_ouptut** in scikit-learn 1.4+. Book examples run with scikit-learn 0.23.2 use **sparse**. My PC has scikit-learn rev 1.9 so I must use **sparse_output**.

    Each row contains a single '1'. This is called one-hot encoding because in each row there is one “hot” level, meaning one non-zero level.

    Each unique row pattern (ie 001, 100) represent a unique value of Embarked, where S is for Southampton England(starting point), C is for Cherbourg France (1st stop), Q for Queenstown in County Cork Ireland (2nd and final stop ever for the Titanic))

    All rows for C have a 1 in the first column, 0 for the others. Q has a 1 in the second column only, S has a 1 in the third column only.

    This is the output of OneHotEncoder
    - 1, 0, 0 means 'C'
    - 0, 1, 0 means 'Q'
    - 0, 0, 1 means 'S'
    """)
    return


@app.cell
def _(OneHotEncoder, X):
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit_transform(X.select('Embarked'))
    return (ohe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Think of the categories_ attribute as the column header of the 10 by 3 array. Categories are always listed in alphabetic order, so first column is C followed by Q and S.  Notice that many attributes in scikit-learn end in an _, as shown here
    """)
    return


@app.cell
def _(ohe):
    ohe.categories_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Passengers who embarked at Queenstown Ireland were less affluent than passengers who embarked from Southampton England or Cherbourg France. Almost all Irish‑embarking passengers were third‑class emigrants, while Southampton and Cherbourg had a mix of first‑, second‑, and third‑class travelers.
    """)
    return


@app.cell
def _(df):
    y = df.select('Survived').to_series()
    y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### One-hot encoding of multiple features
    We used OneHotEncoder with the Embarked column, but we need to encode
    both Embarked and Sex. OneHotEncoder can encode multiple features at once,
    by passing a 2-column DataFrame to fit_transform. Previously
    we passed a 1-column DataFrame. Encoding of 2-columns outputs 5 columns, with first 3 columns
    representing Embarked, last 2 columns representing Sex.
    """)
    return


@app.cell
def _(X, ohe):
    ohe.fit_transform(X.select('Embarked', 'Sex'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So for this example, the first row with 00101 means embarked from S, male. Second row
    with 10010 means they embarked from C, female.
    """)
    return


@app.cell
def _(ohe):
    ohe.categories_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### When to use transform instead of fit_transform?
    Following example use transform to encode 2 categories
    """)
    return


@app.cell
def _(ohe, pl):
    demo_train = pl.DataFrame({'letter':['A', 'B', 'C', 'B']})
    ohe.fit_transform(demo_train)
    return (demo_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    demo_train has 2 occurances of 'B',that show in the 2nd and 4th rows of fit_transform output
    """)
    return


@app.cell
def _(pl):
    demo_test = pl.DataFrame({'letter':['A', 'C', 'A']})
    demo_test
    return (demo_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run fit_transform on the testing data. Output array includes 2 columns because demo_test only has 2 categories.
    The first column represents the A category, and the second column represents the C category.
    """)
    return


@app.cell
def _(demo_test, ohe):
    ohe.fit_transform(demo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output array only includes two columns, because the testing data only included two categories.
    The first column represents the A category, and the second column represents the C category.

    This is problematic, because if we trained a model using the 3-column feature matrix, and then tried
    to make predictions on the 2-column feature matrix, it would error due to a shape mismatch. That
    makes sense because if you train a model such as logistic regression using 3 features, it will learn 3
    coefficients, and it expects to use all 3 of those coefficients when making predictions.

    The solution is to run fit_transform on the training data, and only run transform on the testing
    data. Let’s take a look at the output arrays.

    Notice that the categories are represented the same way in both arrays: The first column represents
    A, the second column represents B, and the third column represents C.
    """)
    return


@app.cell
def _(demo_train, ohe):
    ohe.fit_transform(demo_train)
    return


@app.cell
def _(demo_test, ohe):
    ohe.transform(demo_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lessons learned**
    1. Run fit_transform on training data:
    • fit: Learn 3 categories (A, B, C)
    • transform: Create feature matrix with 3 columns

    2. Run transform on testing data:
    • transform: Create feature matrix with 3 columns

    When using any transformer, use the **fit_transform** method first on the
    training data and just use **transform** method on the testing data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.5: What happens if the testing data includes a new category?
    """)
    return


@app.cell
def _(demo_train):
    demo_train
    return


@app.cell
def _(demo_train, ohe):
    ohe.fit_transform(demo_train)
    return


@app.cell
def _(ohe):
    ohe.categories_
    return


@app.cell
def _(pl):
    demo_test_unknown = pl.DataFrame({'letter':['A', 'C', 'D']})
    demo_test_unknown
    return (demo_test_unknown,)


@app.cell
def _(OneHotEncoder):
    # next line throws an error because 'D' was not seen in the training data
    # ohe.transform(demo_test_unknown) 

    # one solution is to specify columns manually. This is only useful when all 
    # possible categories are known in advance, which is not always the case. Other
    # solution is to use the `handle_unknown` parameter of the OneHotEncoder, and
    # set it to 'ignore', to ignore unknown categories during transformation.
    ohe_unknown_cats=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    return (ohe_unknown_cats,)


@app.cell
def _(demo_train, ohe_unknown_cats):
    ohe_unknown_cats.fit_transform(demo_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The third sample below is encoded as all zeros because D is an unknown category.
    """)
    return


@app.cell
def _(demo_test_unknown, ohe_unknown_cats):
    print(f'{ohe_unknown_cats.categories_ =}')
    print(f'{demo_test_unknown = }')
    ohe_unknown_cats.transform(demo_test_unknown)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This might seem strange but is actually reasonable since the training data
    had no information about the relationship between the D category and the target
    value.
    A major limitation of this is that all unknown categories will be encoded the same
    way. An E value in the testing data would also be encoded as all zeros.

    Advice from the author:
    1. When starting a project, keep the handle_unknown parameter set to default value of 'error'. This will alert you if new categories exist in the testing data.
    2. When new categories are found, determine the full set of categories
    through research, then define the categories manually when creating the OneHotEncoder
    instance.
    3. When the full set of categories can't be known, then set the handle_unknown parameter to
    'ignore'. However, you are urged retrain your model ASAP with data that includes
    any new categories.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.6: Should I drop one of the one-hot encoded categories?
    """)
    return


@app.cell
def _(demo_train):
    demo_train
    return


@app.cell
def _(demo_train, ohe):
    ohe.fit_transform(demo_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When one-hot encoding, it’s somewhat common to drop the first column of the output array because
    it contains redundant information and because it avoids collinearity between features.
    """)
    return


@app.cell
def _(OneHotEncoder, demo_train):
    ohe_drop_first = OneHotEncoder(sparse_output=False, drop='first')
    ohe_drop_first.fit_transform(demo_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dropping the first column will work regardless of the number of categories, but you’re only ever
    allowed to drop a single column. And it doesn’t actually matter which column you drop, though the
    convention is to drop the first column.

    question is: Should you drop the first column?

    If you know that perfectly collinear features will cause problems, such as when feeding the resulting
    data into a neural network or an unregularized regression, then it’s a good idea to drop the first
    column. However, for most scikit-learn models, perfectly collinear features will not cause any
    problems, and thus dropping the first column will not benefit the model.

    There are significant downsides to dropping the first column:

    - Dropping the first column is incompatible with ignoring unknown categories, which is the
    handle_unknown='ignore' option from the previous lesson, since the dropped
    category and unknown categories would both be encoded as all zeros. You are allowed to do
    this starting in scikit-learn 1.0, but I still don’t recommend it.
    - Dropping the first column can introduce bias into the model if you standardize your features,
    such as with StandardScaler, or if you use a regularized model, such as logistic regression,
    since the dropped category will be exempt from standardization and regularization.

    **Recommendation**: drop the first column only if perfectly collinear
    features will cause problems, otherwise keep the first column
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.7: Should I drop one of the one-hot encoded categories?
    Pclass, which stands for passenger class, is an ordinal feature. Although it’s already numeric, the
    numbers 1, 2, and 3 represent the categories 1st class, 2nd class, and 3rd class. It’s considered
    ordinal data because there’s a logical ordering to the categories.

    Our intuition is that there may be a relationship between Pclass values increasing and survival rate
    decreasing, because passengers in the lower-numbered classes may have gotten priority access to
    lifeboats.
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thus if we were going to include Pclass in the model, we would keep the existing numeric encoding
    so that the model can learn the relationship between Pclass and Survived with a single feature.
    You could use one-hot encoding with Pclass instead, but the model wouldn’t be able to learn that
    relationship as effectively because that information would be spread out across three features.

    Options for encodeing Pclass are:
    - Ordinal encoding: Create 1 feature
    - One-hot encoding: Create 3 features
    """)
    return


@app.cell
def _(pl):
    df_ordinal = pl.DataFrame({
        'Class': ['third', 'first', 'second', 'third'],
        'Size': ['S', 'S', 'L', 'XL']
    })
    df_ordinal
    return (df_ordinal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For ordinal data use the OrdinalEncoder class. First, import it from the
    preprocessing module. Then, create an instance of OrdinalEncoder, and define the
    logical order of the categories.

    A list of lists is passed to the categories parameter. The first inner list is
    the categories for the Class feature, and the second inner list is the
    categories for the Size feature. The two lists are in that order because that is
    the order for passing features to the fit_transform method.

    One important note is that “M” category for Size is included even though it wasn’t present in this
    DataFrame because it is expected in the dataset at some point.
    """)
    return


@app.cell
def _(OrdinalEncoder, df_ordinal):
    oe = OrdinalEncoder(categories=[['first', 'second', 'third'], ['S', 'M', 'L', 'XL']])
    oe.fit_transform(df_ordinal)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the Class feature, “first” was encoded as 0, “second” was encoded as 1, and “third” was encoded
    as 2.
    For the Size feature, “S” was encoded as 0, “L” was encoded as 2, and “XL” was encoded as 3. And if
    “M” appears in the data at some point, it will be encoded as 1.
    Again, we encoded each input feature as a single column so that the model can learn the relationship
    between the target and an increase or decrease in each feature.

    Contrast this with the output from OneHotEncoder with these same two features.
    Output array will have 7 columns, because Class has 3 categories and Size
    has 4 categories. These 7 columns contain the same information as the 2 columns output by
    OrdinalEncoder, but the model would have a comparatively harder time learning from the 7 columns
    since the information is expressed in a less compact form.
    """)
    return


@app.cell
def _(OneHotEncoder, df_ordinal):
    ohe_7_cols = OneHotEncoder(
            sparse_output=False, 
            categories=[
                ['first', 'second', 'third'], 
                ['S', 'M', 'L', 'XL']
            ]
    )
    ohe_7_cols.fit_transform(df_ordinal)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Author's advice for encoding categorical data:
    - **Ordinal feature stored as numbers**: Leave as-is
    - **Ordinal feature stored as strings**: Use OrdinalEncoder
    - **Nominal feature**: Use OneHotEncoder

    OrdinalEncoder, unlike OneHotEncoder, does not directly allow for new categories in the
    testing data that were not seen during training. For that functionality is
    use a handle_unknown parameter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.8: What’s the difference between OrdinalEncoder and LabelEncoder?
    1. OrdinalEncoder allows you to define the order of the categories,
    whereas LabelEncoder uses alphabetical order of values.

    2. OrdinalEncoder can be used to encode multiple features at once,
    whereas LabelEncoder can only encode one column of data at once.

    | |Ordinal Encoder|Label Encoder |
    |--|:--:|:--:|
    Category order definable?|Yes|No
    Multiple features encodable?|Yes|No

    Outdated uses for LabelEncoder:
    - Encoding string-based labels for some classifiers
    - Encoding string-based features for OneHotEncoder
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3.8: What’s the difference between OrdinalEncoder and LabelEncoder?
    When there are continuous numeric feature such as Fare, pass them directly to
    your Machine Learning model, no endoding is necessary
    """)
    return


@app.cell
def _(df):
    df.select('Fare')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A strategy with numeric features is to “discretize” or “bin” them
    into categorical ranges. In scikit-learn this is done with KBinsDiscretizer.
    """)
    return


@app.cell
def _(KBinsDiscretizer, df):
    kb = KBinsDiscretizer(n_bins=3, strategy='quantile', encode='ordinal')
    kb.fit_transform(df.select('Fare'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Should numeric features be discretized? Theoretically, discretization
    can benefit linear models by helping them to learn non-linear trends. However, the general
    recommendation is to not use discretization,  for these reasons:

    1. Discretization removes nuances that makes it harder for a model to
    learn data trends.
    2. Discretization reduces the variation in the data, which may flag trends
    that aren't real.
    3. Benefits of discretization depend on the parameters. Arbitrary descisions risk
    overfitting the training data. Making those decisions during a tuning process
    adds complexity and processing time. Neither of these options are attractive.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### more on hot key encoding
    This table (from AI) differentiates hot key encoding of scikit-learn vs. equivalent dataframe solutions (pandas/get_dummies(), polars/to_dummies()).

    | Feature|pandas, polars| OneHotEncoder (scikit-learn)
    |--|--|--|
    |Stateful| No | Yes|
    |Output type| Dataframe | Numpy/sparse matrix|
    |Best for | EDA, quick transforms | ML pipelines, production|
    |Ease of use| Very easy | more setup|
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
