import nltk
import stopwords
import matplotlib.pyplot as plt
from nltk import SnowballStemmer, defaultdict
import importData
import re
import numpy as np
from sklearn import metrics

"""Apart from obvious procedures like removing punctuation and lowering all sentences 
we try different preprocessing techniques such as:
1. Stemming, (reducing words to their root form, we decrease the size of words dictionary).
2. Stop words removal, (we assume that words like 'the', 'and', 'a/an' etc. don't give much information).
3. N-grams, (considering contiguous sequences of n items from a given sample of text)."""


# Remove punctuation and lower sentences
def regex(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text


importData.train_df.head()
# consider only rating 1 and 10
bayes_df_train = importData.train_df[(importData.train_df.rating == '1') | (importData.train_df.rating == '10')]
bayes_df_test = importData.test_df[(importData.test_df.rating == '1') | (importData.test_df.rating == '10')]

stemmer = SnowballStemmer("english")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def stem_(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])


stemmed_train_df = bayes_df_train.copy()
stemmed_test_df = bayes_df_test.copy()
stemmed_train_df.text = stemmed_train_df.text.apply(lambda row: stem_(row))
stemmed_test_df.text = stemmed_test_df.text.apply(lambda row: stem_(row))


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop_words])


swr_train_df = bayes_df_train.copy()
swr_test_df = bayes_df_test.copy()
swr_train_df.text = swr_train_df.text.apply(lambda row: remove_stop_words(row))
swr_test_df.text = swr_test_df.text.apply(lambda row: remove_stop_words(row))


def stem_and_remove_stop_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])


stemmed_swr_train_df = bayes_df_train.copy()
stemmed_swr_test_df = bayes_df_test.copy()
stemmed_swr_train_df.text = stemmed_swr_train_df.text.apply(lambda row: stem_and_remove_stop_words(row))
stemmed_swr_test_df.text = stemmed_swr_test_df.text.apply(lambda row: stem_and_remove_stop_words(row))


# Score visualization
def print_score(preds, Y, name, prints=True):
    print(name)
    acc = np.mean(preds == Y)
    print(f"Acc: {acc}")
    M = metrics.confusion_matrix(preds, Y)
    N = np.sum(M)
    if prints:
        print('\nConfusion matrix:')
        print(M)
        print(f'\nTrue negative (rating = 1): {M[0][0]}')
        print(f'True positive (rating = 10): {M[1][1]}')
        print(f'False negative: {M[0][1]}')
        print(f'False positive: {M[1][0]}')
    return M, N, acc


def plot_bar(X, Y1, Y2, title, x_title, width=0.02, a=0, b=-1):
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}%'.format(height * 100),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig, ax = plt.subplots(figsize=(15, 5))
    rects1 = ax.bar(X[a: b] - width / 2, Y1[a: b], width, label='Train')
    rects2 = ax.bar(X[a: b] + width / 2, Y2[a: b], width, label='Test')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel(x_title)
    ax.set_xticks(X[a: b])
    ax.set_ylim([0, 0.7])
    ax.set_title(title)
    ax.legend(loc='lower right')
    autolabel(rects1)
    autolabel(rects2)


def plot(X, Y1, Y2, title, x_title):
    plt.plot(X, Y1, label='Train')
    plt.plot(X, Y2, label='Test')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.xlabel(x_title)
    plt.ylabel('Accuracy')


class MyCountVectorizer:
    def __init__(self, min_df=-1, max_df=1e18, binary=False):
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary

    def fit(self, df):
        words_cnt = defaultdict(int)
        col = df.columns[0]

        for i in range(len(df)):
            text = df.iloc[i][col]
            for word in text.split():
                words_cnt[word] += 1

        all_words = []
        for word, cnt in words_cnt.items():
            if self.min_df <= cnt <= self.max_df:
                all_words.append(word)

        self.all_words_ids = {w: i for i, w in enumerate(all_words)}
        self.width = len(all_words)

    def transform(self, df):
        col = df.columns[0]
        count_matrix = np.zeros([len(df), self.width], dtype=np.int32)

        for i in range(len(df)):
            text = df.iloc[i][col]
            words_cnt = defaultdict(int)

            for word in text.split():
                words_cnt[word] += 1

            for word, cnt in words_cnt.items():
                if word in self.all_words_ids:
                    pos = self.all_words_ids[word]
                    if self.binary:
                        count_matrix[i][pos] = 1
                    else:
                        count_matrix[i][pos] = cnt

        return count_matrix