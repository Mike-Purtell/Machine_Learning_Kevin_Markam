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
app = marimo.App(width="medium", layout_file="layouts/chap_7.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import make_union

    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.impute import KNNImputer
    from sklearn.impute import MissingIndicator

    # Iterative Imputer is expermental, requires enabling this feature explicitly
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    import polars as pl
    import polars.selectors as cs

    return (
        CountVectorizer,
        IterativeImputer,
        KNNImputer,
        LogisticRegression,
        MissingIndicator,
        OneHotEncoder,
        SimpleImputer,
        make_column_transformer,
        make_pipeline,
        make_union,
        pl,
    )


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 7: Handling missing values
    Completed on Saturday September 5, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - Learned how to impute missing value with sklearn. KNN (nearest neighbors is most interesting to me)
    - Learned about missingness, where missing values have relationship with other data. Example is Age, where older people are more reluctant to share their age.
    - Some of the missing value strategies are best managed at the dataframe level.
    - Machine learning models often will not tolerate missing values
    - Interesting way to add columns to indicate that value was imputed from missing data
    """)
    return


@app.cell
def _(pl):
    # Initialize our dataframes for training and new data
    df = pl.read_csv('http://bit.ly/MLtrain', n_rows=10)
    df_new = pl.read_csv('http://bit.ly/MLnewdata', n_rows=10)
    return df, df_new


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Example of missing data
    Look at Cabin (7 missing values out of 10) and Age (1 missing value)

    Missing values vs unknown categories:
    - Missing value: Value encoded as NaN or None
    - Unknown category: Category not seen in the training data
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    OneHotEncoder,
    df,
    make_column_transformer,
    make_pipeline,
):
    cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']
    X = df.select(cols)
    y = df.select('Survived')
    # add Age to a ColumnTransformer as a passthrough columns
    ohe = OneHotEncoder()
    vect = CountVectorizer()
    logreg = LogisticRegression(solver='liblinear', random_state=1)
    ct = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        ('passthrough', ['Parch', 'Fare', 'Age']))  
    pipe = make_pipeline(ct, logreg)

    # this doesn't work - scikit learn models don't accept missing data
    # pipe.fit(X, y)   #  in this case, 1 value missing from Age
    return X, cols, ohe, vect, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 7.2 Three ways to handle missing values

    1. Drop rows with missing values
    This dataframe approach is impractical. If there are a large number of missing values, it will discard to much of the training data.
    """)
    return


@app.cell
def _(X):
    X.drop_nulls()    # drops rows with any missing values, 9 rows remaining   
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2. Drop columns with missing values
    Notice the age column is gone
    """)
    return


@app.cell
def _(X, pl):
    X.select(
        [
            pl.col(column)
            for column in X.columns
            if X[column].null_count() == 0
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3. Impute missing values
    Fill in missing values based on what is known from existing data. Here are the tradeoffs

    - Benefit: Keeps more samples and features
    - Cost: Imputed values may not match the true values

    Factors to consider before imputing:
    - How important are the samples?
    - How important are the features?
    - What percentage of values would need to be imputed?

    There are many strategies for handling of missing data, to be covered later in this chapter. From the dataframe world, missing values can be replaced by mean, median, fill_forward, fill_backward, interpolation, and many other ways. Best way depends on the context.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 7.3 Missing value imputation
    This lesson performs missing value imputation on the Age column, so that it can be included in the model. First method is the SimpleImputer

    Simple imputation strategies:
    - Mean value (I believe this the default strategy)
    - Median value
    - Most frequent value
    -User-defined value
    """)
    return


@app.cell
def _(SimpleImputer, X):
    imp = SimpleImputer()
    # notice that missing values in the 'Age' column are imputed
    # The missing value is replaced with 28.111 (mean value of non-missing values)
    imp.fit_transform(X.select('Age'))
    return (imp,)


@app.cell
def _(imp):
    imp.statistics_
    return


@app.cell
def _(X, imp, make_column_transformer, ohe, vect):
    ct_impute = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age']),
        ('passthrough', ['Parch', 'Fare'])
    )
    ct_impute.fit_transform(X)
    return (ct_impute,)


@app.cell
def _(LogisticRegression, X, ct_impute, make_pipeline, y):
    pipe_impute = make_pipeline(
        ct_impute,
        LogisticRegression()
    )
    pipe_impute.fit(X, y)
    return (pipe_impute,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before we make predictions, we’re going to examine the Pipeline to confirm the imputation value for Age that was learned from X:
    - First, we access the columntransformer step of the Pipeline using the named_steps attribute.
    - Then, from that Pipeline step, we access the simpleimputer transformer using the
    named_transformers_ attribute.
    - Finally, from that transformer, we access the statistics_ attribute.

    This confirms what we saw previously, which is that the imputer learned a value of 28.11 from the Age column. This query does not show which column was imputed, but that can be found by inspected earlier code, or using interactive pipeline diagram above this cell.
    """)
    return


@app.cell
def _(pipe_impute):
    (
        pipe_impute
            .named_steps['columntransformer']
            .named_transformers_['simpleimputer']
            .statistics_
    )
    return


@app.cell
def _(cols, df_new):
    X_new = df_new.select(cols)
    X_new
    return (X_new,)


@app.cell
def _(X_new, pipe_impute):
    pipe_impute.predict(X_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that X_new had no missing values in the Age column, so nothing was imputed during prediction. If X_new had missing values in Age, the imputed value would have been the mean of Age in X, which is 28.11, not the mean of Age in X_new. This is very important because a transformer is only allowed to learn from the training data, and then apply what it learned to both the training and new data.

    What do transformers learn from the training data?
    - OneHotEncoder: Learns categories
    - CountVectorizer: Learns vocabulary
    - SimpleImputer: Learns imputation value

    #### 7.4 Using “missingness” as a feature
    """)
    return


@app.cell
def _(SimpleImputer, X):
    imp_indicator = SimpleImputer(add_indicator=True)
    imp_indicator.fit_transform(X.select('Age'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why add a missing indicator?
    - Useful when the data is not missing at random
    - Can encode the relationship between “missingness” and the target value

    For example, if missing Age was more common with older passengers reluctant to share their ages, and older passengers were more likely to have survived, then there is a relationship between occurances of missing Age and the likelihood of survival. The missingness itself can be a useful feature, that can be included as a model feature using a missing indicator.

    #### 7.5 How do I perform a multivariate imputation?
    SimpleImputer does univariate imputation, which only looks at the feature being imputed when deciding what values to impute. Thus when imputing missing values for Age, SimpleImputer only considers the values in the Age column.

    Multivariate imputation takes other features into account when deciding what
    values to impute.

    For example, if high Parch and a low Fare is common for kids and Age is missing for a this kind of row, you should impute a low Age rather than the mean of Age, which is what SimpleImputer would do.

    Multivariate imputation is available in scikit-learn via the IterativeImputer and KNNImputer classes, and in this lesson I’ll explain how both of them work.

    Notice in the example below that the imputed age reduced from 28.111 to 24.237.

    Notes about IterativeImputer:
    - Only works with numerical features
    - You have to decide which features to include
    - You can include multiple features with missing values
    - You can choose the regression model
    """)
    return


@app.cell
def _(IterativeImputer, X):
    imp_iterative = IterativeImputer()
    imp_iterative.fit_transform(X.select('Parch', 'Fare', 'Age'))
    return (imp_iterative,)


@app.cell
def _(imp_iterative, make_column_transformer, ohe, vect):
    ct_imp_iterative = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp_iterative, ['Parch', 'Fare', 'Age']))
    ct_imp_iterative
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### KNNImputer
    Example below has n_neighbors set to 2. So KNN find the 2 rows with closest match to Parch and Fare, and calculates the mean value of them. In this case, the 2 rows with closest match on Parch and Fare are the 3rd row where age of 26 and the 5th row where age of 35. The mean value for Age in these 2 nearest rows is 30.5 as shown below.

    How KNNImputer works:
    1. Find the row in which Age is missing
    2. Find the n_neighbors nearest rows in which Age is not missing
    3. Calculate the mean of Age from the nearest rows
    """)
    return


@app.cell
def _(KNNImputer, X):
    imp_knn = KNNImputer(n_neighbors=2)
    imp_knn.fit_transform(X.select('Parch', 'Fare', 'Age'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 7.6 Best practices for missing value imputation
    Missing value imputation may be dubious from a statistics point of view, but has been shown experimentally to improve prediction under certain conditions:

    Missing Completely At Random (MCAR):
    - No relationship between missingness and underlying data
    - Example: Booking agent forgot to gather Age
    - For small datasets, IterativeImputer is more effective than mean imputation.
    - For large datasets, IterativeImputer and mean imputation work equally
    well, though IterativeImputer has a much higher computational cost.
    - Missing indicator offers little value since the missingness is random.

    Missing Not At Random (MNAR):
    - Relationship between missingness and underlying data
    - Example: Older passengers declined to give their Age
    - Mean imputation is more effective than IterativeImputer
    - Add a missing indicator
    - Use a powerful, non-linear model for prediction

    Missing due to a structural deficiency:
    - Data omitted for a specific purpose
    - Example: Staff members did not pay a Fare
    - Impute a logical and reasonable user-defined value (for example use 0 for comps)
    - Add a missing indicator

    Best practices differ depending upon the type of missing data. But if you’re going to impute missing values, it’s important to thoroughly understand your data before choosing an imputation strategy.

    Histogram-based gradient boosting trees in scikit-learn have built-in support for missing values. These are handled them internally at a lower computation cost than complex imputation strategies like IterativeImputer, and perform just as well or better than imputation across a variety of missing value scenarios.

    For large datasets with many of missing values, it’s worth using a histogram based
    gradient boosting tree as the prediction model and excluding the imputation step. It is also worthwhile to compare its performance with models that do require an imputation step.

    #### 7.7 Difference between ColumnTranformer and Feature Union
    """)
    return


@app.cell
def _(SimpleImputer, X):
    imp_indicator_7p7 = SimpleImputer(add_indicator=True)
    imp_indicator_7p7.fit_transform(X.select('Age'))
    return


@app.cell
def _(X, imp):
    imp.fit_transform(X.select('Age'))
    return


@app.cell
def _(MissingIndicator, X):
    indicator = MissingIndicator()
    indicator.fit_transform(X.select('Age'))    
    return (indicator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Example below shows how to use a union of a simple imputer and a missing indicator to handle missing values in the 'Age' column. The value of '1' next to 28.11 indicates the value was missing and imputed.
    """)
    return


@app.cell
def _(X, imp, indicator, make_union):
    union = make_union(imp, indicator)
    union.fit_transform(X.select('Age'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same results as above could have been achieved usinga a ColumnTransformer, by passing age to 2 separate tranformations.
    """)
    return


@app.cell
def _(X, imp, indicator, make_column_transformer):
    _ct = make_column_transformer(
        (imp, ['Age']),        # imputes missing values in the 'Age' column
        (indicator, ['Age']),  # adds indicator for missing values in'Age' column
    )
    _ct.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    FeatureUnion vs ColumnTransformer:
    - FeatureUnion:
      - Single input column
      - Applies multiple different transformations to that column in parallel
    - ColumnTransformer:
      - Multiple input columns
      - Applies a different transformation to each column in parallel

    ColumnTransformer is far more flexible than FeatureUnion and is recommended for vast majority of transformations. When multiple transformations in parallel are requred on the same column, use a FeatureUnion or a ColumnTransformer.
    """)
    return


if __name__ == "__main__":
    app.run()
