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


@app.cell
def _():
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    import polars as pl

    return (
        CountVectorizer,
        LogisticRegression,
        OneHotEncoder,
        make_column_transformer,
        make_pipeline,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 6: Encoding Text Data
    Completed on Friday September 4, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - Value of vectorizing text data like names
    - patterns could be found across common first names and last names
    - type of name is not kept, for example a name like Patricia Thomas, does not assign any attribute to 'Thomas', such as last name. If there were a Thomas Jefferson, his 'Thomas' and Patricia's Thomas would have the same impact on modeling and prediction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 Vectorizing text data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Example of text data
    Name could be predictive of survival if part of important family or if uses
    certain titles (such as Mrs.)
    """)
    return


@app.cell
def _(pl):
    df = pl.read_csv('http://bit.ly/MLtrain', n_rows=10)
    df.select('Name')
    df_new = pl.read_csv('http://bit.ly/MLnewdata', n_rows=10)
    return df, df_new


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Best to split up names instead of keeping full names. Use CountVectorizer class to
    convert text into a matrix of token counts. In summmary:
    - OneHotEncoder: Each full name treated as a category (not recommended)
    - CountVectorizer: Each word in a name treated independently (recommended)
    """)
    return


@app.cell
def _(CountVectorizer, df):
    vect = CountVectorizer()  # instantiate vect
    dtm = vect.fit_transform(df.select('Name').to_series()) # dtm = Document Term Matrix
    dtm
    return dtm, vect


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### CountVectorizer vs other transformers:
    - CountVectorizer: 1-dimensional input (Series)
    - Other transformers: 2-dimensional input (DataFrame)
    """)
    return


@app.cell
def _(vect):
    print(vect.get_feature_names_out())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Default settings for CountVectorizer:
    - Convert all words to lowercase
    - Remove all punctuation
    - Exclude one-character words
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    About the document-term matrix:
    - 10 rows and 40 columns
    - Rows represent rows from training data, columns represent words
    - Rows are “documents”, feature names are “terms”
    - Sparse matrix
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How to examine a document-term matrix:
    1. Use toarray method to make it dense
    2. Convert dense matrix into a DataFrame
    3. Use feature names as column headings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Build column transformer
    pass schema the list of column names, processed as shown here
    """)
    return


@app.cell
def _(dtm, pl, vect):
    pl.DataFrame(dtm.toarray(), schema=list(vect.get_feature_names_out()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    “Bag of Words” representation:
    - Ignores word order
    - Only counts how many times a word appears
    """)
    return


@app.cell
def _():
    ### 6.2 Including text data in the model
    return


@app.cell
def _(df):
    cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name']
    X = df.select(cols)
    X
    return (X,)


@app.cell
def _(OneHotEncoder, make_column_transformer, vect):
    ohe = OneHotEncoder()
    ct = make_column_transformer(
        (ohe, ['Embarked', 'Sex']),
        (vect, 'Name'),
        ('passthrough', ['Parch', 'Fare'])  
    )
    return (ct,)


@app.cell
def _(X, ct):
    ct.fit_transform(X)
    return


@app.cell
def _(ct):
    list(ct.get_feature_names_out())
    return


@app.cell
def _(LogisticRegression, X, ct, df, make_pipeline):
    logreg = LogisticRegression(solver='liblinear', random_state=1) # from chapters 2, 4
    pipe = make_pipeline(ct, logreg)
    y = df.select('Survived').to_numpy()
    pipe.fit(X,y)
    return (pipe,)


@app.cell
def _(df_new, pipe):
    X_new = df_new.select(['Parch', 'Fare', 'Embarked', 'Sex', 'Name'])
    pipe.predict(X_new)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.3 Why is document-term matrix stored as sparse
    """)
    return


@app.cell
def _(pl, vect):
    text = ['Machine Learning is fun', 'I am learning Machine Learning']
    pl.DataFrame(
        vect.fit_transform(text).toarray(),
        schema=list(vect.get_feature_names_out())
    )
    return (text,)


@app.cell
def _(text, vect):
    _dtm = vect.fit_transform(text)
    print(_dtm
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Preferred matrix representation:
    - Most elements are zero: Sparse matrix
    - Most elements are non-zero: Dense matrix
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 6.4 What happens if the testing data includes new words?
    """)
    return


@app.cell
def _(text, vect):
    _dtm = vect.fit_transform(text)
    vect.get_feature_names_out()
    return


@app.cell
def _(vect):
    text_new = ['Data Science is FUN!']
    vect.transform(text_new).toarray()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CountVectorizer methods:
    - fit: Learn the vocabulary
    - transform: Create the document-term matrix using that vocabulary

    By comparing the output to the feature names, the vectorizer only learned 'fun' and 'is' from the testing data. 'data' and 'science' were ignored because they were not part of the training data.

    Ignoring unknown words makes sense. If a word was not part of training, then nothing is known about the relationship between that word and the target variable. Similar to setting the OneHotEncoder’s handle_unknown parameter to 'ignore', since that ignores unknown categories encountered during the transform step by encoding them as all zeros.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    6.5 Q&A: How to vectorize multiple columns of text
    Look at Name and Ticket columns
    """)
    return


@app.cell
def _(df):
    df.select(['Name', 'Ticket'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    vect.fit_transform(df.select('Name').to_series())
    """)
    return


@app.cell
def _(df, vect):
    vect.fit_transform(df.select('Ticket').to_series())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What we want is to stack these matrices side-by-side as a 10 by 53 matrix.
    One idea would be to pass both columns as a DataFrame to CountVectorizer, which is how we
    would transform multiple columns with OneHotEncoder. However, you’ll see that the output is a
    2 by 2 matrix, which is not what we had hoped for. This is because CountVectorizer expects
    1-dimensional input, and we passed it 2-dimensional input instead.
    """)
    return


@app.cell
def _(df, vect):
    vect.fit_transform(df.select(['Name', 'Ticket']).to_pandas())
    return


@app.cell
def _(df, make_column_transformer, vect):
    _ct = make_column_transformer(
        (vect, 'Name'),
        (vect, 'Ticket')
    )
    _ct.fit_transform(df)
    _ct
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 6.6 One-hot encode or vectorize categorical features?
    What if you have categorical features with only one word, such as Embarked and Sex. We’ve been using OneHotEncoder, but should we use CountVectorizer instead? Let’s try it out and see what happens.
    """)
    return


@app.cell
def _(df):
    df.select(['Embarked', 'Sex'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    CountVectorizer on the Sex column produces the exact same output as the
    OneHotEncoder.
    """)
    return


@app.cell
def _(df, vect):
    vect.fit_transform(df['Sex']).toarray()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From previous lesson, CountVectorizer won’t do what you expect if you try to encode multiple columns at once, since it expects 1-dimensional input.
    """)
    return


@app.cell
def _(df, vect):
    vect.fit_transform(df.select(['Embarked', 'Sex']).to_pandas()).toarray()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Default settings for CountVectorizer doesn't allow one-character tokens. Would have to modify those settings to use it with Embarked.
    """)
    return


@app.cell
def _(df, vect):
    vect.fit_transform(df.select('Embarked').to_pandas()).toarray()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, OneHotEncoder lets you decide how you want to handle categories that weren’t seen during
    training (using the handle_unknown parameter), whereas CountVectorizer will always ignore
    words that it didn’t see during training.


    In summary, OneHotEncoder is the better encoding mechanism for any data that you would consider
    categorical, since it can encode multiple columns at once, it allows one-character category names by
    default, and it provides more options for handling unknown categories.

    Advantages of OneHotEncoder for categorical data:
    - Encodes multiple columns at once
    - Allows one-character category names
    - Gives more options for handling unknown categories
    """)
    return


if __name__ == "__main__":
    app.run()
