import discogs_client
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import eli5
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def api_request(d, genres):

    tracklists = []
    for genre in genres:
        hits = d.search(genre=genre, country="UK")

        l = []
        for i, hit in enumerate(hits):
            print(i)
            if i == 1000:
                break
            l.append(hit.tracklist)
            time.sleep(1)

        tracklists.append(l)

    return tracklists


def format_data(data, genres):

    x, y = [], []
    for i, g in enumerate(data):
        for album in g:
            tl = []
            for song in album:
                tl.append(song.title)

            x.append(tl)
            y.append(genres[i])

    x = [" ".join(map(str, i)) for i in x]

    return x, y


def pipe(X_train, y_train):

    ps = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (ps.stem(w) for w in analyzer(doc))

    pipeline = Pipeline(
        [
            ("vec", CountVectorizer(analyzer=stemmed_words)),
            ("tfidf", TfidfTransformer()),
            ("clf", xgb.XGBClassifier()),
        ]
    )

    return pipeline.fit(X_train, y_train)


def scores(y_test, y_pred, g):

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, ax=ax, annot=True,
                xticklabels=np.sort(g), yticklabels=np.sort(g))

    ax.set_xlabel("Predicted Genre")
    ax.set_ylabel("True Genre")
    plt.tight_layout()
    fig.show()


def eli5_features(test, pipeline):

    clf = pipeline.named_steps["clf"]
    vec = pipeline.named_steps["vec"]
    transformer = Pipeline(pipeline.steps[:-1])

    with open("eli5_weights.html", "w") as f:
        f.write(eli5.show_weights(clf, vec=vec, top=50).data)

    with open("eli5_prediction.html", "w") as f:
        f.write(eli5.show_prediction(clf,
                                     transformer.transform(test),
                                     feature_names=vec.get_feature_names()).data)


if __name__ == "__main__":

    g = [
        "Rock",
        "Electronic",
        "Pop",
        "Jazz",
        "Latin",
        "Folk, World, & Country",
        "Funk / Soul",
        "Classical",
        "Hip Hop",
        "Reggae",
    ]

    d = discogs_client.Client("ExampleApplication/0.1", user_token="token")
    tl = api_request(d, g)  # this will take ~6 hours

    x, y = format_data(tl, g)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                random_state=30, stratify=y)
    pipeline = pipe(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    scores(y_test, y_pred, g)
    eli5_features(X_test[1044:1045], pipeline)
