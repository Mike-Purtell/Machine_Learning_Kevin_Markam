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
    import polars as pl
    import polars.selectors as cs
    import numpy as np

    import os

    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction.text import CountVectorizer    
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import VotingClassifier

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="`sklearn.utils.parallel.delayed` should be used"
    )
    return (
        CountVectorizer,
        GridSearchCV,
        LogisticRegression,
        OneHotEncoder,
        RandomForestClassifier,
        SimpleImputer,
        VotingClassifier,
        cross_val_score,
        make_column_transformer,
        make_pipeline,
        os,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 12:  Ensembling multiple models
    - Started Tuesday September 15, 2026

    Item | Book uses|I use|
    |--|--|--|
    python|unknown|3.14
    dataframes|pandas|polars
    scikit-learn|0.23.2 (2025)|1.9.0 (June 2026)
    IDE|unknown|VS Code
    Notebooks|unknown|Marimo

    **My takeaways:**

    - TBD

    #### 12.1 Introduction to ensembling
    We’ve been using two different models, logistic regression and random forests. The process of  ensembling produces a combined model, known as an ensemble, that is more accurate than any of the individual models.
    The process for ensembling is:
    - For a regression, calculate the average of predictions from individual
    regressors to use as your prediction.
    - For a classification, average the predicted probabilities output by the
    classifiers, or let the classifiers vote on which class to predict

    Ensembling works because:
    - “One-off” errors made by each model will be discarded when ensembling
    - Ensemble has a lower variance than any individual model

    #### 12.2 Ensembling logistic regression and random forest
    """)
    return


@app.cell
def _(
    CountVectorizer,
    LogisticRegression,
    OneHotEncoder,
    SimpleImputer,
    cross_val_score,
    make_column_transformer,
    make_pipeline,
    os,
    pl,
):
    # list of 6 columns we will select from the dataset
    cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

    # read the training dataset and use it to define X and y
    training_data_path = './assets/MLtrain.csv'
    if not os.path.exists(training_data_path):
        print("Downloading training dataset...")
        df = pl.read_csv('http://bit.ly/MLtrain')
        df.write_csv(training_data_path)
    else:
        print("Loading training dataset from local file...")
        df = pl.read_csv(training_data_path)

    df = pl.read_csv(training_data_path)
    X = df[cols]
    y = df.select('Survived')

    # read the new dataset and define X_new
    new_data_path = './assets/MLnewdata.csv'
    if not os.path.exists(new_data_path):
        print("Downloading new dataset...")
        df_new = pl.read_csv('http://bit.ly/MLnewdata')
        df_new.write_csv(new_data_path)
    else:
        print("Loading new dataset from local file...")
        df_new = pl.read_csv(new_data_path)

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
        # l1_ratio=0.0,
        random_state=1,
    )

    # create a 2-step pipeline, fit it to X and y, and make predictions on X_new
    pipe = make_pipeline(ct, logreg)
    pipe.fit(X, y.to_series())
    print(cross_val_score(pipe, X, y.to_series(), scoring ='accuracy').mean())
    return X, X_new, ct, logreg, pipe, y


@app.cell
def _(RandomForestClassifier, X, cross_val_score, ct, make_pipeline, y):
    rf = RandomForestClassifier(random_state=1, n_jobs=-1)
    rf_pipe = make_pipeline(ct, rf)
    print(cross_val_score(rf_pipe, X, y.to_series(), scoring ='accuracy').mean())
    return rf, rf_pipe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create the ensemble using the VotingClassifier class, which we’ll import from the ensemble module. Then create an instance called vc and pass it a list of tuples, in which the first element of the tuple is a name and the second element is a classifier object. The options for the voting parameter are 'soft', in which predicted probabilities are averaged, and 'hard', in which only class predictions are taken into account. Try soft voting first, with n_jobs to -1 to enable parallel processing.
    """)
    return


@app.cell
def _(VotingClassifier, logreg, rf):
    vc_1 = VotingClassifier(
        [
            ('clf1', logreg),
            ('clf2', rf)
        ],
        voting='soft',
        n_jobs=-1
    )
    return (vc_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create a new pipeline called vc_pipe in which VotingClassifier is the second step instead of a model
    """)
    return


@app.cell
def _(ct, make_pipeline, vc_1):
    vc_pipe_1 = make_pipeline(ct, vc_1)
    vc_pipe_1
    return (vc_pipe_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 12.3 Combining predicted probabilities
    Examine how the VotingClassifier makes predictions when using soft voting, by looking at the predicted probabilities output by the logistic regression and random forest Pipelines for the first 3 samples in X_new. The left column is the predicted probability of class 0 (for each sample), and the right column is the predicted probability of class 1.
    """)
    return


@app.cell
def _(X, X_new, pipe, y):
    pipe.fit(X, y.to_series())
    pipe.predict_proba(X_new)[:3]
    return


@app.cell
def _(X, X_new, rf_pipe, y):
    rf_pipe.fit(X, y.to_series())
    rf_pipe.predict_proba(X_new)[:3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now use the VotingClassifier to output predicted probabilities, using fit and predict_proba, just like other classifiers. If you examine its predicted probabilities, you’ll see that it’s simply averaging the two sets of probabilities from logistic regression and random forests. For example, the 0.19 in the left column of the second sample is the average of 0.14 (from logistic regression) and 0.24 (from random forests).
    """)
    return


@app.cell
def _(X, X_new, vc_pipe_1, y):
    vc_pipe_1.fit(X, y.to_series())
    vc_pipe_1.predict_proba(X_new)[:3]
    return


@app.cell
def _(X_new, vc_pipe_1):
    vc_pipe_1.predict(X_new[:3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the 3 cases we just examined, logistic regression and random forests agreed on the class predictions. Let’s now examine a case in which the two models disagree. For sample 80, logistic regression predicts class 0 but without much confidence, whereas random forests predicts class 1 with more confidence.
    """)
    return


@app.cell
def _(X_new, pipe):
    pipe.predict_proba(X_new)[80]
    return


@app.cell
def _(X_new, rf_pipe):
    rf_pipe.predict_proba(X_new)[80]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When VotingClassifier averages the predicted probabilities for this sample, the class 1 value is higher than the class 0 value, thus it will predict class 1.
    """)
    return


@app.cell
def _(X_new, vc_pipe_1):
    vc_pipe_1.predict_proba(X_new)[80]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let’s move on to cross-validation to see how the VotingClassifier Pipeline with soft voting performs. Its score is 0.818, which is better than either model alone.
    """)
    return


@app.cell
def _(X, cross_val_score, vc_pipe_1, y):
    cross_val_score(vc_pipe_1, X, y.to_series(), cv=5, scoring='accuracy').mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 12.4 Combining class predictions
    Modify the VotingClassifier to use hard voting, which ignores predicted probabilities and takes a majority vote based on class predictions.
    """)
    return


@app.cell
def _(VotingClassifier, X, cross_val_score, ct, logreg, make_pipeline, rf, y):
    vc_2 = (
        VotingClassifier([
            ("clf1", logreg),
            ("clf2", rf)
        ],
        voting='hard',
        n_jobs=-1
        )
    )
    vc_pipe_2 = make_pipeline(ct, vc_2)
    cross_val_score(
        vc_pipe_2, 
        X, 
        y.to_series(), 
        cv=5, 
        scoring='accuracy'
    ).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why is this result misleading?
    - In the case of a tie, hard voting always chooses class 0
    - Thus hard voting is performing better than soft voting by chance

    #### 12.5 Choosing a voting strategy
    Soft voting:
    - Preferred if you have an even number of models (especially two)
    - Preferred if all models are well-calibrated
    - Only works if all models have the predict_proba method

    Hard voting:
    - Preferred if some models are not well-calibrated
    - Does not require the predict_proba method
    """)
    return


@app.cell
def _(VotingClassifier, ct, logreg, make_pipeline, rf):
    vc_3 = VotingClassifier([
        ('clf1', logreg), 
        ('clf2', rf)
        ], 
        voting='soft',
        n_jobs=-1
    )

    vc_pipe_3 = make_pipeline(ct, vc_3)

    return (vc_pipe_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 12.6 Tuning and ensemble with grid search
    VotingClassifier’s hyperparameters can be tuned using a grid search to improve its accuracy. The best parameters for the VotingClassifier Pipeline might be different than the parameters for either model tuned separately. Create a vc_params dictionary with only the ColumnTransformer parameters.
    """)
    return


@app.cell
def _():
    vc_params = {
        'columntransformer__pipeline__onehotencoder__drop': [None, 'first'],
        'columntransformer__countvectorizer__ngram_range': [(1, 1), (1, 2)],
        'columntransformer__simpleimputer__add_indicator': [False, True]
    }



    vc_params 
    return (vc_params,)


@app.cell
def _(vc_pipe_3):
    vc_pipe_3.named_steps.keys()
    return


@app.cell
def _(vc_pipe_3):
    print(vc_pipe_3.named_steps['votingclassifier'].named_estimators)
    return


@app.cell
def _(vc_params):
    vc_params['votingclassifier__clf1__penalty'] = ['l1', 'l2']
    vc_params['votingclassifier__clf1__C'] = [1, 10]
    vc_params['votingclassifier__clf2__n_estimators'] = [100, 300]
    vc_params['votingclassifier__clf2__min_samples_leaf'] = [2, 3]
    vc_params
    return


@app.cell
def _(GridSearchCV, X, vc_params, vc_pipe_3, y):
    vc_grid = GridSearchCV(
        vc_pipe_3, 
        vc_params, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    vc_grid.fit(X, y.to_series())
    return (vc_grid,)


@app.cell
def _(vc_grid):
    print(vc_grid.best_score_)
    return


@app.cell
def _(vc_grid):
    print(vc_grid.best_params_)
    return


@app.cell
def _(X_new, vc_grid):
    vc_grid.predict(X_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 12.7 When should I use ensembling?
    Ensembling generally improves model performance. It is useful when performance is the highest priority. Ensembling increases process complexity and is less interpretable than a single model.

    Recommendation for ensembling is to include at least 3 models in the ensemble, and that these models performing reasonably well on their own. Ideally the selected models generate their predictions using different processes.

    #### 12.8 How to apply different weights to models in an ensemble
    By default, each model within an ensemble has an equal weight. You can weight certain models more than others to give them more “voting power” for determining the predicted class labels or predicted probabilities. For example, logistic regression model could be given double the voting power of the random forest model by setting the weights parameter of the VotingClassifier.
    """)
    return


@app.cell
def _(VotingClassifier, X, ct, logreg, make_pipeline, rf, y):
    vc_4 = VotingClassifier([
        ('clf1', logreg),
        ('clf2', rf),
        ],
        voting='soft',
        weights=[2, 1],
        n_jobs=-1
    )
    vc_pipe_4 = make_pipeline(ct, vc_4)
    vc_pipe_4.fit(X, y.to_series())
    return (vc_pipe_4,)


@app.cell
def _(X_new, vc_pipe_4):
    vc_pipe_4.predict_proba(X_new)[:3]
    return


@app.cell
def _(X, cross_val_score, vc_pipe_4, y):
    cross_val_score(
        vc_pipe_4, 
        X, 
        y.to_series(), 
        cv=5,
        scoring='accuracy'
        ).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can search for optimal weights using a grid search. Here’s how to add that to the vc_params dictionary.
    """)
    return


@app.cell
def _(vc_params):
    vc_params['votingclassifier__weights'] = [(1, 1), (2, 1), (1, 2)]
    return


if __name__ == "__main__":
    app.run()
