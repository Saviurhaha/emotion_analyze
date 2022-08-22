import cvxopt
import numpy as np
import pandas as pd
import glob
import re
from collections import defaultdict
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import scipy.stats as sstats
from sklearn.metrics import confusion_matrix

train_pos_path = 'datasets/aclImdb/train/pos/*'
train_neg_path = 'datasets/aclImdb/train/neg/*'
train_pos = glob.glob(train_pos_path)
train_neg = glob.glob(train_neg_path)
test_pos_path = 'datasets/aclImdb/test/pos/*'
test_neg_path = 'datasets/aclImdb/test/neg/*'
test_pos = glob.glob(test_pos_path)
test_neg = glob.glob(test_neg_path)
train_df = []
test_df = []


# 1. GETTING THE DATA AND DIVIDE THE DATA TO TWO PARAMETER: RATING AND INDEX
# Read data sets from aclImdb
def read_data(path, message):
    res = []
    for p in tqdm(path, desc=message, position=0):
        with open(p, encoding="utf8") as f:
            text = f.read()
            beg = p.find('\\')
            idx, rating = p[beg + 1:-4].split('_')  # record the rating of each file
            res.append([text, rating])

    return res


# merge train_pos and train_neg to train_df
train_df += read_data(path=train_pos, message='Getting positive train data')
train_df += read_data(path=train_neg, message='Getting negative train data')

# merge test_pos and test_neg to test_df
test_df += read_data(path=test_pos, message='Getting positive test data')
test_df += read_data(path=test_neg, message='Getting negative test data')

train_df = pd.DataFrame(train_df, columns=['text', 'rating'])
test_df = pd.DataFrame(test_df, columns=['text', 'rating'])

print('Records: ', train_df.size)  # Records:  50000  25000*2(text and rating)
# Df.head() will treat the first row in the Excel table as the column name and output the next five rows by default
print(train_df.head())

# calculate the number of each rating because we using supervised learning, In the labeled train/test sets,
# a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with
# more neutral ratings are not included in the train/test sets. In the unsupervised set, reviews of any rating are
# included and there are an even number of reviews > 5 and <= 5.

# range: range(start, stop[, step]) not include stop!
for i in range(1, 11):
    print(f'Number of reviews with rating {i}: {train_df[train_df.rating == str(i)].shape[0]}')

# Number of reviews with rating 1: 5100
# Number of reviews with rating 2: 2284
# Number of reviews with rating 3: 2420
# Number of reviews with rating 4: 2696
# Number of reviews with rating 5: 0
# Number of reviews with rating 6: 0
# Number of reviews with rating 7: 2496
# Number of reviews with rating 8: 3009
# Number of reviews with rating 9: 2263
# Number of reviews with rating 10: 4732

# 2. PRE-PROCESSING
"""Apart from obvious procedures like removing punctuation and lowering all sentences 
we try different preprocessing techniques such as:
1. Stemming, (reducing words to their root form, we decrease the size of words dictionary).
2. Stop words removal, (we assume that words like 'the', 'and', 'a/an' etc. don't give much information)."""


# Remove the irregular symbol
def regex(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text


# Remove punctuation and lower all texts
train_df.text = train_df.text.apply(lambda row: regex(row))
test_df.text = test_df.text.apply(lambda row: regex(row))
print(train_df.head())

# consider only rating 1 and 10
bayes_df_train = train_df[(train_df.rating == '1') | (train_df.rating == '10')]
bayes_df_test = test_df[(test_df.rating == '1') | (test_df.rating == '10')]

# only stemming: reducing words to their root form
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))


def stem_(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])


stemmed_train_df = bayes_df_train.copy()
stemmed_test_df = bayes_df_test.copy()
stemmed_train_df.text = stemmed_train_df.text.apply(lambda row: stem_(row))
stemmed_test_df.text = stemmed_test_df.text.apply(lambda row: stem_(row))
print(stemmed_train_df.head())
print(stemmed_test_df.head())


# only stop words remove: removing the no meaning words
def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop_words])


swr_train_df = bayes_df_train.copy()
swr_test_df = bayes_df_test.copy()
swr_train_df.text = swr_train_df.text.apply(lambda row: remove_stop_words(row))
swr_test_df.text = swr_test_df.text.apply(lambda row: remove_stop_words(row))
print(swr_train_df.head())
print(swr_test_df.head())


# stemmed and stop words removing
def stem_and_remove_stop_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])


stemmed_swr_train_df = bayes_df_train.copy()
stemmed_swr_test_df = bayes_df_test.copy()
stemmed_swr_train_df.text = stemmed_swr_train_df.text.apply(lambda row: stem_and_remove_stop_words(row))
stemmed_swr_test_df.text = stemmed_swr_test_df.text.apply(lambda row: stem_and_remove_stop_words(row))


# Score visualization
# visualization by confusion matrix
# print true negative(TN), true positive(TP), false negative(FN), false positive(FP)
# print the figure of confusion matrix

def confusion_matrix_plot_matplotlib(y_truth, y_predict, cmap=plt.cm.PuRd):
    # labels = ["positive", "negative"]
    cm = confusion_matrix(y_truth, y_predict)
    plt.matshow(cm, cmap=cmap)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')
    # xlocations = np.array(range(len(labels)))
    # plt.xticks(xlocations, labels, rotation=90)
    # plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


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
    confusion_matrix_plot_matplotlib(preds, Y)

    return M, N, acc


# plotting
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


def auto_label(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}%'.format(height * 100),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot(X, Y1, Y2, title, x_title):
    plt.plot(X, Y1, label='Train')
    plt.plot(X, Y2, label='Test')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.xlabel(x_title)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


# For each training text, only the frequency of each word in the training text is considered
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


# Naive Bayes is based on prior and posterior probabilities
# In order to prevent the occurrence of zero probability, the Laplace coefficient (alpha) fitting is used
class NaiveBayes:
    def __init__(self, alpha=0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior_array = class_prior
        if class_prior:
            self.fit_prior = False

    # Priors: the prior probability size, if not given, the model calculates itself from the sample data
    # (using the maximum likelihood method)

    def fit(self, X, y):
        self.classes, prior = np.unique(y, return_counts=True)
        self.N = len(y)

        # Setting class prior
        if self.fit_prior:
            self.class_prior = {class_: np.log(prior[i] / self.N + 1e-100)
                                for i, class_ in enumerate(self.classes)}
        elif self.class_prior_array:
            self.class_prior = {class_: np.log(self.class_prior_array[i] + 1e-100)
                                for i, class_ in enumerate(self.classes)}
        else:
            self.class_prior = {class_: np.log(1 / len(self.classes) + 1e-100)
                                for class_ in self.classes}

        # Creating words dictionaries
        self.class_words_counts = {class_: defaultdict(lambda: 0)
                                   for class_ in self.classes}
        for i, text in enumerate(X):
            target = y[i]
            for word in text.split():
                self.class_words_counts[target][word] += 1

        # Creating probabilities dictionaries
        self.class_words_probs = {class_: defaultdict(lambda: np.log(self.alpha + 1e-100))
                                  for class_ in self.classes}
        for class_, dict_ in self.class_words_counts.items():
            for word, count in dict_.items():
                self.class_words_probs[class_][word] = np.log(count + 1e-100)

        self.class_words_amount = {class_: np.log(sum(self.class_words_counts[class_].values()))
                                   for class_ in self.classes}

    def get_class_log_probabilities(self, text):
        probs = {class_: 0 for class_ in self.classes}
        for class_ in self.classes:
            for word in text.split():
                probs[class_] += self.class_words_probs[class_][word]
                probs[class_] -= self.class_words_amount[class_]
            probs[class_] += self.class_prior[class_]
        return probs

    def predict(self, X, return_probabilities=False):
        preds = []
        preds_probs = []
        for text in X:
            prob = self.get_class_log_probabilities(text)
            # prob = {class_ : np.exp(pbb) for class_,pbb in prob.items()}
            preds_probs.append(prob)
            pred = max(prob, key=prob.get)
            preds.append(pred)

        if return_probabilities:
            return preds, preds_probs
        return preds


# alpha: Laplace smoothing coefficient, used for the case of fitting probability is 0, alpha is generally 1
X_train, y_train = np.array(bayes_df_train['text']), np.array(bayes_df_train['rating'])
X_test, y_test = np.array(bayes_df_test['text']), np.array(bayes_df_test['rating'])
NBc_res = []

# the predictions of test data set
print("------------------------NB: the accuracy of test data------------------------")
alpha = 1.0
NB = NaiveBayes(fit_prior=False, alpha=alpha)
NB.fit(X_train, y_train)
predictions, ppb = NB.predict(X_train, return_probabilities=True)
predictions, ppb = NB.predict(X_test, return_probabilities=True)
M, N, acc = print_score(predictions, y_test, f"TEST, alpha : {alpha}")
NBc_res.append(['Org test data\na=1.0', 'Acc', acc])
NBc_res.append(['Org test data\na=1.0', 'FN', M[0][1] / N])
NBc_res.append(['Org test data\na=1.0', 'FP', M[1][0] / N])

# the predictions of stemmed test data set
print("------------------------NB: the accuracy of stemmed test data------------------------")
NB = NaiveBayes(fit_prior=False, alpha=alpha)
NB.fit(np.array(stemmed_train_df['text']), np.array(stemmed_train_df['rating']))
predictions = NB.predict(np.array(stemmed_test_df['text']))
M, N, acc = print_score(predictions, np.array(stemmed_test_df['rating']),
                        f"TEST, alpha : {alpha}, stemmed")
NBc_res.append(['Stemmed test data\na=1.0', 'Acc', acc])
NBc_res.append(['Stemmed test data\na=1.0', 'FN', M[0][1] / N])
NBc_res.append(['Stemmed test data\na=1.0', 'FP', M[1][0] / N])

# the predictions of test data without stopwords
print("------------------------NB: the accuracy of test data without stopwords------------------------")
NB = NaiveBayes(fit_prior=False, alpha=alpha)
NB.fit(np.array(swr_train_df['text']), np.array(swr_test_df['rating']))
predictions = NB.predict(np.array(swr_test_df['text']))
M, N, acc = print_score(predictions, np.array(stemmed_test_df['rating']),
                        f"TEST, alpha : {alpha}, stop words removed")
NBc_res.append(['Test data\nwithout stopwords\na=1.5', 'Acc', acc])
NBc_res.append(['Test data\nwithout stopwords\na=1.5', 'FN', M[0][1] / N])
NBc_res.append(['Test data\nwithout stopwords\na=1.5', 'FP', M[1][0] / N])

# the predictions of stemmed test data without stop words
print("------------------------NB: the accuracy of stemmed test data without stop words------------------------")
NB = NaiveBayes(fit_prior=False, alpha=alpha)
NB.fit(np.array(stemmed_swr_train_df['text']), np.array(stemmed_swr_train_df['rating']))
predictions = NB.predict(np.array(stemmed_swr_test_df['text']))
M, N, acc = print_score(predictions, np.array(stemmed_test_df['rating']),
                        f"TEST, alpha : {alpha}, stemmed and stop words removed")
NBc_res.append(['Stemmed test data\nwithout stop words\na=1.5', 'Acc', acc])
NBc_res.append(['Stemmed test data\nwithout stop words\na=1.5', 'FN', M[0][1] / N])
NBc_res.append(['Stemmed test data\nwithout stop words\na=1.5', 'FP', M[1][0] / N])

NBc_res_df = pd.DataFrame(NBc_res, columns=['Data', 'Y', 'Value'])
sns.set(style="darkgrid")
ax = sns.barplot(x='Data', y='Value', hue='Y', data=NBc_res_df)
ax.figure.set_size_inches(15, 5)
ax.set_title('Naive Bayes')
auto_label(ax.patches)
ax.legend()
plt.tight_layout()
plt.show()

accuracy = []
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 10.0]

print("-------------------------------NB: The accuracy of different alpha--------------------------------")
for alpha in alphas:
    NB = NaiveBayes(fit_prior=False, alpha=alpha)
    NB.fit(X_train, y_train)
    predictions, ppb = NB.predict(X_test, return_probabilities=True)
    acc = np.mean(predictions == y_test)
    print(f'Alpha : {alpha}, test acc: {acc}')
    accuracy.append(acc)

plt.plot(alphas, accuracy)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy for different alphas')
plt.tight_layout()
plt.show()

# logistic regression: using sklearn
# Min df = 5 means "Ignore terms that appear in less than 5 documents"
cv = CountVectorizer(min_df=5)
cv.fit(stemmed_swr_train_df.text)
cv2 = CountVectorizer(min_df=5)
cv2.fit(bayes_df_train.text)
cv3 = CountVectorizer(min_df=5)
cv3.fit(stemmed_train_df.text)
cv4 = CountVectorizer(min_df=5)
cv4.fit(swr_train_df.text)

X_train_og = cv2.transform(bayes_df_train.text)
X_test_og = cv2.transform(bayes_df_test.text)

X_train_stem_swr = cv.transform(stemmed_swr_train_df.text)
X_test_stem_swr = cv.transform(stemmed_swr_test_df.text)

X_train_stem = cv3.transform(stemmed_train_df.text)
X_test_stem = cv3.transform(stemmed_swr_test_df.text)

X_train_swr = cv4.transform(swr_train_df.text)
X_test_swr = cv4.transform(swr_test_df.text)

y_train = np.array(list(map(int, bayes_df_train['rating']))) // 10
y_test = np.array(list(map(int, bayes_df_test['rating']))) // 10

LG2_res = []

# C: float, default: 1.0; Its value is equal to the reciprocal of the regularization intensity and is a positive
# floating point number. The smaller the number, the stronger the regularization.
# max_iter:Maximum number ofiterations
# Solver: The Solver parameter determines how we optimize the logistic regression loss function.
# There arefour algorithms to choose from
# lbfgs：A kind of quasi - Newton method, the loss function is iteratively optimized by
# using the second derivative matrix of the loss function, namely the Hazen matrix.
print("----------------Different parameter of Logistic Regression--------------")
Cs = [0.01, 0.05, 0.25, 0.5, 0.75, 1]
LR_accuracy = []
for c in Cs:
    lr = LogisticRegression(C=c, max_iter=500, solver='lbfgs')
    lr.fit(X_train_og, y_train)
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_test, lr.predict(X_test_og))))
    LR_accuracy.append(accuracy_score(y_test, lr.predict(X_test_og)))
plt.plot(Cs, LR_accuracy)
plt.xlim(0, 1)
plt.xlabel('c')
plt.ylabel('Accuracy')
plt.title('Accuracy for different c')
plt.tight_layout()
plt.show()

# the predictions of test data set
print("-----------------C=0.05, test data---------------------")
best_C = 0.05
best_lr = LogisticRegression(C=best_C, max_iter=500, solver='lbfgs')
best_lr.fit(X_train_og, y_train)
M, N, acc = print_score(best_lr.predict(X_test_og), y_test, 'TEST')
LG2_res.append(['Org test data\n', 'Acc', acc])
LG2_res.append(['Org test data\n', 'FN', M[0][1] / N])
LG2_res.append(['Org test data\n', 'FP', M[1][0] / N])

# the predictions of stemmed test data set
print("-----------------C=0.05, stemmed test data set--------------------")
lrs = LogisticRegression(C=best_C, max_iter=500, solver='lbfgs')
lrs.fit(X_train_stem, y_train)
M, N, acc = print_score(lrs.predict(X_test_stem), y_test, 'TEST')
LG2_res.append(['Stemmed test data', 'Acc', acc])
LG2_res.append(['Stemmed test data', 'FN', M[0][1] / N])
LG2_res.append(['Stemmed test data', 'FP', M[1][0] / N])

# the predictions of test data without stopwords
print("---------------C=0.05, test data with out stopwords-----------------------")
lrs = LogisticRegression(C=best_C, max_iter=500, solver='lbfgs')
lrs.fit(X_train_swr, y_train)
M, N, acc = print_score(lrs.predict(X_test_swr), y_test, 'TEST')
LG2_res.append(['Test data\nwithout stopwords', 'Acc', acc])
LG2_res.append(['Test data\nwithout stopwords', 'FN', M[0][1] / N])
LG2_res.append(['Test data\nwithout stopwords', 'FP', M[1][0] / N])

# the predictions of stemmed test data without stop words
print("-----------------C=0.05, stemmed test data without stop words---------------------")
lrs = LogisticRegression(C=best_C, max_iter=500, solver='lbfgs')
lrs.fit(X_train_stem_swr, y_train)
M, N, acc = print_score(lrs.predict(X_test_stem_swr), y_test, 'STEMMED TEST WITHOUT STOP WORDS')
LG2_res.append(['Stemmed test data\nwithout stop words\n', 'Acc', acc])
LG2_res.append(['Stemmed test data\nwithout stop words\n', 'FN', M[0][1] / N])
LG2_res.append(['Stemmed test data\nwithout stop words\n', 'FP', M[1][0] / N])

LG2_res_df = pd.DataFrame(LG2_res, columns=['Data', 'Y', 'Value'])
sns.set(style="darkgrid")
ax = sns.barplot(x='Data', y='Value', hue='Y', data=LG2_res_df)
ax.figure.set_size_inches(15, 5)
ax.set_title("Sklearn Logistic Regression")
auto_label(ax.patches)
plt.tight_layout()
plt.show()

# using sklearn decision tree
dt_res = []
print("-----------Decision tree of org test data-----------")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_og, y_train)
M, N, acc = print_score(model_dt.predict(X_test_og), y_test, 'TEST')
dt_res.append(['Org test data\n', 'Acc', acc])
dt_res.append(['Org test data\n', 'FN', M[0][1] / N])
dt_res.append(['Org test data\n', 'FP', M[1][0] / N])

print("------------Decision tree of stemmed test data-----------")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_stem, y_train)
M, N, acc = print_score(model_dt.predict(X_test_stem), y_test, 'TEST')
dt_res.append(['Stemmed test data', 'Acc', acc])
dt_res.append(['Stemmed test data', 'FN', M[0][1] / N])
dt_res.append(['Stemmed test data', 'FP', M[1][0] / N])

print("------------Decision tree of test data without stopwords------------")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_swr, y_train)
M, N, acc = print_score(model_dt.predict(X_test_swr), y_test, 'TEST')
dt_res.append(['Test data\nwithout stopwords', 'Acc', acc])
dt_res.append(['Test data\nwithout stopwords', 'FN', M[0][1] / N])
dt_res.append(['Test data\nwithout stopwords', 'FP', M[1][0] / N])

print("------------Decision tree of stemmed test data without stop words----------")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_stem_swr, y_train)
M, N, acc = print_score(model_dt.predict(X_test_stem_swr), y_test, 'TEST')
dt_res.append(['Stemmed test data\nwithout stop words\n', 'Acc', acc])
dt_res.append(['Stemmed test data\nwithout stop words\n', 'FN', M[0][1] / N])
dt_res.append(['Stemmed test data\nwithout stop words\n', 'FP', M[1][0] / N])

dt_res_df = pd.DataFrame(dt_res, columns=['Data', 'Y', 'Value'])
sns.set(style="darkgrid")
ax = sns.barplot(x='Data', y='Value', hue='Y', data=dt_res_df)
ax.figure.set_size_inches(15, 5)
ax.set_title("Sklearn Decision tree")
auto_label(ax.patches)
plt.tight_layout()
plt.show()


# using SVM
# C is the penalty factor, which is the tolerance for error.
# The higher C is, the less errors can be tolerated and the overfitting is easy.
# The smaller C is, the easier it is to underfit.
# If C is too large or too small, the generalization ability becomes poor
# Gamma is an argument that comes with the RBF function when it is selected as the kernel.
# Implicitly determines the distribution of data mapped into the new feature space
# The bigger the gamma, the fewer the support vectors, the smaller the gamma, the more support vectors.
# The number of support vectors influences the speed of training and prediction.
# th=0.00001
class SVM:
    def __init__(self, C, gamma, th=1e-5):
        self.C = C
        self.gamma = gamma
        self.th = th

    def fit(self, X, y, info=False):
        m, n = X.shape

        # kernel matrix
        if info:
            print(f'Calculating kernel matrix...', end='')
        K = rbf_kernel(X, X, gamma=self.gamma)

        # setup solver
        if info:
            print(f'Setting solver...', end='')
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(np.zeros(1))
        if info:
            print(f'starting solver...')
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol['x'])

        # getting support vectors
        self.S = (self.alphas > self.th).reshape(-1, )
        self.support_vextors = X[self.S]
        self.sv_target = y[self.S]

        # getting bias
        if info:
            print(f'Calculating bias...', end='')
        K_sv = rbf_kernel(self.support_vextors, self.support_vextors, gamma=self.gamma)
        B = self.sv_target.ravel() - np.sum(K_sv * self.sv_target.ravel() * self.alphas[self.S].ravel(), axis=1)
        self.bias = sstats.mode(B)[0][0]

        if info:
            print(f'done')

    def predict(self, X):
        K = rbf_kernel(X, self.support_vextors, gamma=self.gamma)
        preds = np.sign(np.sum(K * self.sv_target.ravel() * self.alphas[self.S].ravel(), axis=1) + self.bias)
        return preds


CV = MyCountVectorizer(min_df=5)
CV.fit(stemmed_swr_train_df)
X_train_SVM, X_test_SVM = CV.transform(stemmed_swr_train_df), CV.transform(stemmed_swr_test_df)
X_stemmed_test_SVM = CV.transform(stemmed_test_df)
X_swr_test_SVM = CV.transform(swr_test_df)
y_train = np.array([-1.0 if r == '1' else 1.0 for r in stemmed_swr_train_df['rating']]).reshape(-1, 1)
y_test = np.array([-1.0 if r == '1' else 1.0 for r in stemmed_swr_test_df['rating']]).reshape(-1, 1)
y_stemmed_test = np.array([-1.0 if r == '1' else 1.0 for r in stemmed_test_df['rating']]).reshape(-1, 1)
y_swr_test = np.array([-1.0 if r == '1' else 1.0 for r in swr_test_df['rating']]).reshape(-1, 1)
C = 1.25
gamma = 0.0025
svm_model = SVM(C, gamma)
svm_model.fit(X_train_SVM, y_train, info=True)

print("-------------------------SVM: the accuracy of stemmed and without stopwords train set---------------------")
svm_predictions_train = svm_model.predict(X_train_SVM)
M, N, acc = print_score(svm_predictions_train, y_train.ravel(),
                        f'Train data, SVM, rbf kernel, C={C}, gamma={gamma}')

print("--------------------------SVM: the accuracy of stemmed and without stopwords test set-----------------------")
svm_predictions = svm_model.predict(X_test_SVM)
M, N, acc = print_score(svm_predictions, y_test.ravel(),
                        f'Test data, SVM, rbf kernel, C={C}, gamma={gamma}')

print("--------------------------------SVM: the accuracy of stemmed test set-----------------------------------------")
svm_predictions_stemmed_test = svm_model.predict(X_stemmed_test_SVM)
M, N, acc = print_score(svm_predictions_stemmed_test, y_stemmed_test.ravel(),
                        f'Test data, SVM, rbf kernel, C={C}, gamma={gamma}')

print("---------------------------SVM: the accuracy of without stopwords test "
      "set-----------------------------")
svm_predictions_swr_test = svm_model.predict(X_swr_test_SVM)
M, N, acc = print_score(svm_predictions_swr_test, y_swr_test.ravel(),
                        f'Test data, SVM, rbf kernel, C={C}, gamma={gamma}')

# Testing different parameters
SVMc_res_train = []
SVMc_res_test = []
cvxopt.solvers.options['show_progress'] = False
print("------------------------------------SVM: different parameters-----------------------------------------")
for c in [1.0, 1.25, 1.5]:
    for gamma_ in [0.001, 0.0025, 0.005]:
        svm_test_model = SVM(c, gamma_)
        print(f'C: {c}, gamma: {gamma_}')
        svm_test_model.fit(X_train_SVM, y_train, info=False)
        svm_preds_train = svm_test_model.predict(X_train_SVM)
        svm_preds_test = svm_test_model.predict(X_test_SVM)
        M_train, N_train, acc_train = print_score(svm_preds_train, y_train.ravel(),
                                                  f'Train data, SVM, rbf kernel, C={c}, gamma={gamma_}', prints=False)
        M_test, N_test, acc_test = print_score(svm_preds_test, y_test.ravel(),
                                               f'Test data, SVM, rbf kernel, C={c}, gamma={gamma_}', prints=False)
        SVMc_res_train.append([f'C = {c}\ngamma = {gamma_}', 'Acc', acc_train])
        SVMc_res_train.append([f'C = {c}\ngamma = {gamma_}', 'FN', M_train[0][1] / N_train])
        SVMc_res_train.append([f'C = {c}\ngamma = {gamma_}', 'FP', M_train[1][0] / N_train])
        SVMc_res_test.append([f'C = {c}\ngamma = {gamma_}', 'Acc', acc_test])
        SVMc_res_test.append([f'C = {c}\ngamma = {gamma_}', 'FN', M_test[0][1] / N_test])
        SVMc_res_test.append([f'C = {c}\ngamma = {gamma_}', 'FP', M_test[1][0] / N_test])
        print()

SVMc_res_train_df = pd.DataFrame(SVMc_res_train, columns=['Data', 'Y', 'Value'])
SVMc_res_test_df = pd.DataFrame(SVMc_res_test, columns=['Data', 'Y', 'Value'])

sns.set(style="darkgrid")
ax = sns.barplot(x='Data', y='Value', hue='Y', data=SVMc_res_test_df)
ax.figure.set_size_inches(20, 5)
ax.set_title("Stemmed test data without stop words, SVM with rbf kernel")
ax.legend(loc='center right')
auto_label(ax.patches)
plt.tight_layout()
plt.show()
