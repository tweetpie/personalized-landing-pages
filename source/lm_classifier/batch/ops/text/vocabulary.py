import pandas as pd

from source.lm_classifier.constants import N_MOST_FREQ_WORDS


def create_frequent_words(tfidf):
    """
    Returns a list of the most frequent words
    """

    idf = pd.Series(dict(zip(tfidf.vocabulary_.keys(), tfidf.idf_))).sort_values(ascending=False)
    df = pd.Series(tfidf.vocabulary_).sort_values(ascending=True)

    items = [
        list(df.index[:N_MOST_FREQ_WORDS]),
        list(df.index[-N_MOST_FREQ_WORDS:]),
        list(idf.index[:N_MOST_FREQ_WORDS])
    ]
    frequent_words = sum(items, [])
    return frequent_words
