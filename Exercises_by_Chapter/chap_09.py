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
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline

    import polars as pl
    import polars.selectors as cs

    import os

    return (
        CountVectorizer,
        LogisticRegression,
        OneHotEncoder,
        SimpleImputer,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 9:  Workflow review #2
    Completed on Sunday September 6, 2026

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
    - TBD
    #### 9.1 Workflow recap

    In this chapter, we’re going to review the workflow that we’ve built so far and also discuss the concept of data leakage.

    The import block above has imported the three transformer classes we’re using, one modeling class, and two composition functions, and the polars datafame library.

    Below is all of the code that's necessary to recreate our workflow up to this point.
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
    logreg = LogisticRegression(solver='liblinear', random_state=1)

    # create a 2-step pipeline, fit it to X and y, and make predictions on X_new
    pipe = make_pipeline(ct, logreg)
    pipe.fit(X, y)
    pipe.predict(X_new)

    return (pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 9.2 Comparing ColumnTransformer and Pipeline

    Let's review our workflow so far:

    The **ColumnTransformer** received 6 columns of input from the X DataFrame:
    - Embarked and Sex were passed to the transformer Pipeline of SimpleImputer and OneHotEncoder.
      - SimpleImputer step imputed a constant value and output 2 columns
      - OneHotEncoder step transformed those 2 columns into 6 columns.
    - The Name column was passed to the CountVectorizer producing 1509 columns.
    - Age and Fare columns passed to the SimpleImputer, which imputed the mean and output 2 columns.
    - Parch column was passed through unmodified.
    - Finally, it horizontally concatenated the 6 + 1509 + 2 + 1 columns side-by-side, resulting in a total of 1518 columns.

    The modeling **Pipeline** has 2 steps:

    - ColumnTransformer that received 6 columns and transformed them into 1518 columns.
    - LogisticRegression model received 1518 columns of input and used them for fitting or predicting.

    Here’s a diagram that illustrates our workflow so far, which we will briefly review.
    """)
    return


@app.cell
def _(mo):
    diagram = mo.mermaid(
        """
        flowchart TB
            X["X\\n6 columns"] --> CT["ColumnTransformer"]

            CT -->|"Embarked, Sex"| imp1["SimpleImputer\\n2 columns"]
            imp1 --> ohe["OneHotEncoder\\n6 columns"]

            CT -->|"Name"| cv["CountVectorizer\\n1509 columns"]

            CT -->|"Age, Fare"| imp2["SimpleImputer\\n2 columns"]

            CT -->|"Parch"| passthrough["passthrough\\n1 column"]

            ohe --> concat["1518 columns"]
            cv --> concat
            imp2 --> concat
            passthrough --> concat

            concat --> lr["LogisticRegression"]
        """
    )
    diagram
    return


@app.cell
def _(mo):
    mo.md(r"""
    Review differences between a ColumnTransformer and a Pipeline

    ColumnTransformer
    - pulls out subsets of columns and transforms them independently, and then concatenates the results side-by-side.
    - It only ever does data transformations.
    - It does not have steps, because each subset of columns is transformed independently.

    Pipeline

    - A series of steps that occur in order, and the output of each step becomes the
    input to the next step.
    - The last step of a Pipeline can be a model or a transformer, whereas all other steps must be transformers.

    We can display the Pipeline to see it's diagram
    """)
    return


@app.cell
def _(pipe):
    pipe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 9.3 Why not use dataframe libaries for transformations at the dataframe level?
    The book does all of the data transformations in scikit-learn, but could have done many of them at the dataframe level, and then passed the transformed data into scikit-learn? There are several reasons for not tranforming at the dataframe level:
    - CountVectorizer, one of the most useful techniques for encoding text data, is not available in pandas, however it is available in polars.

    - You could do some or most of the transformations with polars or pandas and then
    use scikit-learn for CountVectorizer, but this adds complexity to the workflow. Especially true when combining dense matrices from the dataframes with sparse matrices from CountVectorizer. It is more efficient to do all of the transformations with scikit-learn.
    - One-hot encoding with dataframe get_dummies functions could add many columns to your DataFrame making it more difficult to navigate. Scikit-learn for one-hot encoding keeps the source DataFrame intact.
    - Missing value imputation with polars or pandas results in data leakage. Using scikit-learn for imputation avoids data leakage problems.
    - doing all transformations with scikit-learn enables cross-validate and tuning of your entire Pipeline rather than just your model. This improves model peformance.


    Item | scikit-learn| dataframe (polars/pandas|
    |--|--|--|
    Encoding text data|CountVectorizer|not available
    One-hot encoding|OneHotEncoder|use dummies function
    Imputing missing values|Simple Imputer and others|fill_nam or fillna
    Cross validation and tuning|Entire Pipeline|Model Only

    #### 9.4 Preventing data leakage
    Data leakage occurs when test data slips into the training process. When that happens, evaluation scores become unreliable, leading to poor hyperparameter choices and inflated expectations of real‑world performance. Because it’s impossible to know whether the leak skews results slightly or massively, the safest approach is to avoid data leakage altogether.

    In summary, here is why data leakage is problematic:

    - Your model evaluation scores will be less reliable
    - You might make bad decisions when tuning hyperparameters
    - You will overestimate how well your model will perform on new data

    Imputing missing values at the dataframe level causes data leakage because the imputation is computed using all rows, both training and test. Cross‑validation is supposed to mimic the future by keeping test folds unseen during training. But if you impute before splitting, the model ends up using information from the test portion to fill in missing values, breaking that simulation. Imputation values must come only from the training data to keep evaluation honest

    In other words, imputing on the full dataset is like peeking into the future and then using that future knowledge during training, which is absolutely not allowed.

    You might think you can avoid leakage by splitting the data first and then imputing missing values with pandas. That works only if you always evaluate with a single train/test split. It breaks down with cross‑validation, because the training rows change for every fold. Managing imputation manually at the dataframe level for each fold is impractical, which is why imputation must live inside the scikit‑learn pipeline.

    So far, we’ve talked about data leakage mainly in the context of missing‑value imputation, but plenty of other dataframe‑level transformations can leak information too.

    Feature scaling done on the full dataset will leak data, and even one‑hot encoding can leak unless the categories are fixed and known ahead of time. More broadly, any transformation that uses information from other rows when transforming a given row will cause leakage if it’s done at the dataframe level.

    Here are 3 ways that scikit-learn prevents data leakage

    - 1.scikit‑learn transformers separate fit from transform, which lets you learn transformation parameters from the training data only and then apply those same parameters to both the training and test sets.

    - 2. The pipeline’s fit and predict methods wrap all the internal fit_transform and transform calls, ensuring each step runs at the correct time

    - 3. cross_val_score splits the data prior to performing data transformations, which I’ll
    explain in detail in the next chapter.
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
