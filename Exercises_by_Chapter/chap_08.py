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
app = marimo.App(width="medium", layout_file="layouts/chap_08.slides.json")


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
        LogisticRegression,
        OneHotEncoder,
        SimpleImputer,
        make_column_transformer,
        make_pipeline,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 8:  Fixing Common workflow problems
    Completed on Sunday September 6, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    -  Scikit‑learn ≥0.24 (my version) treats missing values as just another category; 0.23.3 (book version) raises an error when missing values are detected.
    - How to manage missing values in categorical data - pay attention to the context, solutions vary
    - How to manage missing values in new data
    - How to see feature names in column transformer - remember that these are Numpy and referenced by an integer index, not be column name string
    - Methods to use with a pipeline
    #### 8.1 Two new problems

    So far, this book has used only the first 10 rows of the Titanic dataset to make it easy to see how each workflow step transforms the data. Starting with this chapter, the full dataset is used. Real‑world issues will show up and we will learn how to handle them properly. Begin by loading the training data into df and the new data into df_new. Note that df_new has one fewer column because it doesn’t include the target variable, Survived.
    """)
    return


@app.cell
def _(pl):
    # Initialize our dataframes for training and new data removed n_rows paramter to use the full datast
    df = pl.read_csv('http://bit.ly/MLtrain')
    df_new = pl.read_csv('http://bit.ly/MLnewdata')
    print(f'{df.shape = }')             # 11 columns, 891 rows
    print(f'{df_new.shape = }')     # 10 columns, 418 rows
    return df, df_new


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### count the number of missing values by column for in each dataframe
    """)
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell
def _(df_new):
    df_new.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Features with missing values:
    - Problematic:
      - Embarked: Missing values in df
      - Fare: Missing value in df_new
    - Not problematic:
      - Cabin: Not currently using
      - Age: Already being imputed
    #### 8.2 Problem 1: Missing values in a categorical feature

    Six feature columns we use are:
    """)
    return


@app.cell
def _(df):
    cols = ['Parch',  'Fare', 'Embarked', 'Sex', 'Name', 'Age']
    X = df.select(cols)
    y = df.select('Survived')
    return X, cols, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I’m not seeing the missing‑value errors the book describes for Embarked. That’s because older scikit‑learn (0.23.3) raised errors when OneHotEncoder saw missing values, while the version I’m using (0.24+) treats them as a separate category and runs without complaint.
    """)
    return


@app.cell
def _(
    CountVectorizer,
    OneHotEncoder,
    SimpleImputer,
    X,
    make_column_transformer,
):
    ohe = OneHotEncoder()
    vect = CountVectorizer()
    imp = SimpleImputer()
    ct = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age']),
        ('passthrough', ['Parch', 'Fare'])
    )
    ct.fit_transform(X)
    return imp, ohe, vect


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even though my scikit‑learn version can handle missing values in OneHotEncoder, I’m still following the book’s approach and fixing them with SimpleImputer. Imputing a few missing entries is better than treating them as an “unknown” category.
    We start by creating a SimpleImputer with strategy="constant". Because we’re using a Polars DataFrame, we set missing_values=None. The book doesn’t specify this parameter, because its default is NaN, which is what pandas uses for missing values.
    """)
    return


@app.cell
def _(SimpleImputer):
    imp_constant = SimpleImputer(
        strategy = 'constant',
        fill_value = 'missing',
        missing_values = None   # to show missing values as 'missing', had to add this parameter to imp_constant
    )
    imp_constant
    return (imp_constant,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Imputation strategies for categorical features:
    - Most frequent value
    - User-defined value

    Create a two-step pipeline that only contains transformers. First step is imputation using our new imputer, and second step is one-hot encoding. Call this pipelinge imp_ohe to show of the two steps it contains.
    """)
    return


@app.cell
def _(X, imp_constant, make_pipeline, ohe):
    imp_ohe = make_pipeline(imp_constant, ohe)
    imp_ohe.fit_transform(X.select('Embarked'))
    return (imp_ohe,)


@app.cell
def _(imp_ohe):
    imp_ohe[1].categories_
    return


@app.cell
def _(imp, imp_ohe, make_column_transformer, vect):
    ct_imp_ohe = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age']),
        ('passthrough', ['Parch', 'Fare'])
    )
    return (ct_imp_ohe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A few notes about the imp_ohe pipeline:

    - A ColumnTransformer can only contain transformer objects.  imp_ohe qualifies because all of its steps are transformers.

    - It’s fine to apply imp_ohe to the Sex column. Since it has no missing values, the imputer is a no‑op and the data simply flows into the one‑hot encoder.

    Also notice that the output matrix is much wider now. The Name column introduces a huge number of unique tokens.
    """)
    return


@app.cell
def _(X, ct_imp_ohe):
    ct_imp_ohe.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 8.2:  Missing values in the new data
    This section will deal with missing values in the Fare Column. From earlier in this notebook, Fare has no missing values in df, but has 1 missing values in df_new. Our modeling pipeline will have to account for it.

    Solution is to impute the missing value in the Fare column.  The ColumnTransfer already contains an imputer (imp) for mean imputation on the Age column.  We will apply that to Fare as well.
    """)
    return


@app.cell
def _(X, imp, imp_ohe, make_column_transformer, vect):
    ct_impute_fare = make_column_transformer(
        (imp_ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        (imp, ['Age', 'Fare']),
        ('passthrough', ['Parch'])
    )
    ct_impute_fare.fit_transform(X)
    return (ct_impute_fare,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fare column does not have any missing values in X, thus the imputer did not impute
    any values for Fare during the fit_transform. However, the imputer learned the mean of Fare in X, which is
    the imputation value that will be applied to Fare in X_new during prediction.

     Let's update the modeling Pipeline to include the revised ColumnTransformer, and fit it
    on X and y. You can see from the diagram that there’s now a transformer Pipeline within the
    ColumnTransformer, which is within the modeling Pipeline.
    """)
    return


@app.cell
def _(LogisticRegression, X, ct_impute_fare, make_pipeline, y):
    logreg = LogisticRegression(solver='liblinear', random_state=1)
    pipe = make_pipeline(ct_impute_fare, logreg)
    pipe.fit(X, y)
    return logreg, pipe


@app.cell
def _(cols, df_new, pipe):
    X_new = df_new.select(cols)
    pipe.predict(X_new)
    return (X_new,)


@app.cell
def _(mo):
    mo.md(r"""
    #### 8.4 How do I see the feature names output by the ColumnTransformer?
    When we pass X to the ColumnTransformer’s fit_transform method, it outputs a matrix with 1518
    columns. How can we find out the names of these columns?
    """)
    return


@app.cell
def _(X, ct_impute_fare):
    ct_impute_fare.fit_transform(X)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Starting in scikit-learn 1.1, the get_feature_names_out method is available for all transformers, so retrieving the feature names will no longer error.

    Changes to get_feature_names:
    - Starting in version 1.0: get_feature_names replaced with get_feature_names_out
    - Starting in version 1.1: get_feature_names_out available for all transformers
    """)
    return


@app.cell
def _(ct_impute_fare):
    ct_impute_fare.get_feature_names_out()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To stay in sync with the book, this section will roughly show how to get the feature names for scikit-learn versions prior to 1.1. First step, print the transformer configuration:
    """)
    return


@app.cell
def _(ct_impute_fare):
    print(ct_impute_fare.transformers_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first transformer is a Pipeline of SimpleImputer and OneHotEncoder. OneHotEncoder has a
    get_feature_names method, which we can access by selecting the pipeline transformer and then
    its onehotencoder step. get_feature_names outputs 6 features, which we know are the first 6
    features in the matrix because this is the first transformer in the ColumnTransformer.
    """)
    return


@app.cell
def _(ct_impute_fare):
    (
        ct_impute_fare
        .named_transformers_['pipeline']
        .get_feature_names_out()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second transformer is a CountVectorizer. It also has a get_feature_names method, which
    we can access by selecting the countvectorizer transformer. We could print out all of the feature
    names, but instead we’ll pass it to the len function, which indicates that the next 1509 features in
    the matrix came from CountVectorizer
    """)
    return


@app.cell
def _(ct_impute_fare):
    len(ct_impute_fare.named_transformers_['countvectorizer'].get_feature_names_out())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Features output by each transformer:
    - Pipeline: 6 features (Embarked and Sex)
    - CountVectorizer: 1509 features (Name)
    - SimpleImputer: 2 features (Age and Fare)
    - passthrough: 1 feature (Parch)

    #### 8.5 Why did we create a Pipeline inside of the ColumnTransformer?
    The Embarked column contained missing values and needed onehot
    encoding, so we created a two-step Pipeline called imp_ohe. The first step of this Pipeline is
    imputation of a constant value, and the second step is one-hot encoding.

    We included the imp_ohe Pipeline in the ColumnTransformer, and applied it to both the Embarked
    and Sex columns. Here’s what it would look like if the ColumnTransformer only contained the
    imp_ohe Pipeline.

    When you run the fit_transform method, Embarked turns into 4 columns and Sex turns into 2
    columns, and the results are stacked side-by-side.
    """)
    return


@app.cell
def _(X, imp_constant, make_column_transformer, make_pipeline, ohe):
    _imp_ohe = make_pipeline(imp_constant, ohe)
    _ct = make_column_transformer((_imp_ohe, ['Embarked', 'Sex']))
    _ct.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the Sex column didn’t contain any missing values and only needed one-hot encoding, we
    could have achieved the exact same result by applying imp_ohe to Embarked and separately applying
    ohe to Sex. The fit_transform does indeed output the same results as above, though I personally prefer the
    first ColumnTransformer. (I Agree with this point. Even though Titanic is a static data set with no new info expected, realworld data often grows, meaning the columns with no missing data now may have missing values as future data is added.)
    """)
    return


@app.cell
def _(X, imp_ohe, make_column_transformer, ohe):
    _ct = make_column_transformer(
        (imp_ohe, ['Embarked']),
        (ohe, ['Sex']))
    _ct.fit_transform(X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One common question is whether you can avoid using the imp_ohe Pipeline entirely by making
    a ColumnTransformer like this instead, in which the imputation of a constant value is applied to
    Embarked, and one-hot encoding is applied to both Embarked and Sex.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall the key differences between a Pipeline and a ColumnTransformer:
    - In a Pipeline, the output of one step is the input to the next step. This is precisely why
    we created the imp_ohe Pipeline: We needed the output of the imputer to become the input
    to the one-hot encoder.
    - A ColumnTransformer does not have steps. Instead, it has transformers that operate
    in parallel, and the output of each transformer is stacked beside the other transformer outputs.

    #### 8.6 Which imputation strategy is best with categorical features?
    When imputing missing values for a categorical feature, you can either impute the most frequent value or a constant user-defined value. In this lesson, I’m going to discuss how you might choose between these two strategies.

    Imputing a constant value essentially treats the missing values as a new category, which I believe is the better choice regardless of whether the values are missing at random or not at random. Imputing a constant value is especially important if the majority of values are missing, since imputing the most frequent value in that case would more than double the size of the category that was imputed, which would be quite misleading to the model.

    That being said, imputing the most frequent value is much more acceptable when you have only a small number of missing values for a given feature, since the imputation won’t have much of an impact on the model anyway.

    It’s important to note that if you impute a constant value for a categorical feature, and that feature has missing values in the new data but not the training data, then you’ll need to set the OneHotEncoder’s handle_unknown parameter to 'ignore'. That’s because the missing values category won’t be learned during the OneHotEncoder’s fit step, and thus unknown values seen during the transform step need to be ignored in order to avoid an error.

    The alternative here is to impute the most frequent value for that feature, in which case you can leave the handle_unknown parameter set to its default value of 'error'.

    Possible problem with imputing a constant value:
    - Condition: The feature only has missing values in the new data
    - Solution: Set handle_unknown to 'ignore' for the OneHotEncoder
    - Alternative: Impute the most frequent value, and leave handle_unknown set to 'error'

    ####  8.7 Should I impute missing values before all other transformations?

    Here’s the Pipeline that we’ve built throughout the book. The strategy I’ve used so far is to include all data transformations within a single ColumnTransformer, including any missing value imputation, and then use that ColumnTransformer as the first step in a two-step Pipeline.
    """)
    return


@app.cell
def _(pipe):
    pipe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An alternative approach would be to create a three-step Pipeline in which the first step is missing value imputation, the second step includes all other data transformations, and the third step is the model. Let’s try it out to see if this is a better approach.

    Impute missing values as a first step?

    - Current Pipeline:
      - Step 1: All data transformations
      - Step 2: Model
    - Alternative Pipeline:
      - Step 1: Missing value imputation
      - Step 2: All other data transformations
      - Step 3:

    Here is the first ColumnTransformer, which only does missing value imputation. Constant value imputation is applied to Embarked, mean imputation is applied to Age and Fare, and the other columns are passed through because they don’t contain any missing values in the training or new data. This will be the first step in the Pipeline.
    """)
    return


@app.cell
def _(imp, imp_constant, make_column_transformer):
    ct1 = make_column_transformer(
        (imp_constant, ['Embarked']),
        (imp, ['Age', 'Fare']),
        ('passthrough', ['Sex', 'Name', 'Parch'])
    )
    ct1
    return (ct1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now include these newly created the ColumnTransformers in a three-step Pipeline, and fit the Pipeline to X and y. Note the column references by number instead of by name.
    """)
    return


@app.cell
def _(make_column_transformer, ohe, vect):
    ct2 = make_column_transformer(
        (ohe, [0, 3]),
        (vect, 4),
        ('passthrough', [1, 2, 5])
        # Numpy arrayrequires integer indices. column names like below don't work
        # ('passthrough', ['Sex', 'Name', 'Parch'])  
    )
    return (ct2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we’ve created the ColumnTransformers, we can include them in a three-step Pipeline and fit the Pipeline to X and y.
    """)
    return


@app.cell
def _(X, X_new, ct1, ct2, logreg, make_pipeline, y):
    _pipe = make_pipeline(ct1, ct2, logreg)
    _pipe.fit(X, y)
    _pipe.predict(X_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three-step Pipeline like this is a valid approach. Author finds the original two-step Pipeline easier to write and to read, and thus I prefers it.

    #### 8.8  What methods can I use with a Pipeline?
    The rules for Pipelines are that all steps other than the final step must be a transformer, and the
    final step can be a model or a transformer.

    If a Pipeline ends in a model, such as our pipe object, you can use the Pipeline’s fit and predict methods:
    - If you run the fit method, all steps before the final one run **fit_transform**, and the final step runs **fit**.
    - If you run the predict method, all steps before the final one run **transform**, and the final step runs **predict**.

    If a Pipeline ends in a transformer, such as our imp_ohe object, you generally use the Pipeline’s fit_transform and transform methods, but you can also use the fit method:
    - If you run the fit_transform method, all steps run fit_transform.
    - If you run the transform method, all steps run transform.
    - If you run the fit method, all steps before the final one run fit_transform, and the final step
    runs fit.

    Although this is a lot of information to take in, developing this level of understanding will definitely
    make it easier for you to test and debug your future Pipelines.
    """)
    return


if __name__ == "__main__":
    app.run()
