from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from preProcess import bayes_df_train, bayes_df_test


class Naive_Bayes:
    def __init__(self, alpha=0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior_array = class_prior
        if class_prior:
            self.fit_prior = False

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


X_train, y_train = np.array(bayes_df_train['text']), np.array(bayes_df_train['rating'])
X_test, y_test = np.array(bayes_df_test['text']), np.array(bayes_df_test['rating'])
NBc_res = []
alpha = 1.5
NB = Naive_Bayes(fit_prior=False, alpha=alpha)
NB.fit(X_train, y_train)
accuracy = []
alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 10.0]

for alpha in alphas:
    NB = Naive_Bayes(fit_prior=False, alpha=alpha)
    NB.fit(X_train, y_train)
    predictions, ppb = NB.predict(X_test, return_probabilities=True)
    acc = np.mean(predictions == y_test)
    print(f'Alpha : {alpha}, test acc: {acc}')
    accuracy.append(acc)

plt.plot(alphas, accuracy)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy for different alphas')
