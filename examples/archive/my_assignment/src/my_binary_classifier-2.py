"""
Train and evaluate a binary classifier for sentiment detection.

Design

1. define a class BinaryClassifier using a MultinomialNB classifier from scikit-learn.
    - Initialize the class with parameters for the classifier.
2. add a method load_data to load the data from the file using pd.read_csv.
4. add a method preprocess_data to preprocess the data and split it into training and testing sets.
5. add a method train to train the classifier on the training data.
6. add a method evaluate to evaluate the model on the test data and print the confusion matrix.
7. add a static method to plot the confusion matrix.
8. define a main function to instantiate the class and call the methods in order.

Results:

[INFO] number of documents: 50000
[INFO] number of documents after dedublication: 49582
[INFO] instances of negative: 24698
[INFO] instances of positive: 24884
[[3979  930]
 [ 950 4058]]
[INFO] Relative Accuracy; 0.8104265402843602
[INFO] Accuracy in instances 8037

Author: Kristoffer Nielbo, John Dee and Jane Doe
Email: kln@cas.au.dk

"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


def set_random_seed(seed=0):
    """ Utility function to set random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)


class BinaryClassifier:
    """ A binary classifier for spam detection.
    
    Attributes:
        aplha (float): smoothing parameter for the classifier.
        class_prior (array-like, shape = [n_classes]): prior probabilities of the classes.
        fit_prior (bool): whether to learn class prior probabilities or not.
    """
    def __init__(self, alpha=1.0, class_prior=None, fit_prior=True, max_features=1000):
        self.classifier = MultinomialNB(alpha=alpha, class_prior=class_prior, fit_prior=fit_prior)
        self.cv = CountVectorizer(max_features=max_features)
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_data(self, filename, xcolumn='review', ycolumn='sentiment'):
        """ Load the data from a csv file.

        Args:
            filename (str): path to the csv file.
        """
        data = pd.read_csv(filename)
        print(f'[INFO] number of documents: {data.shape[0]}')
        data.drop_duplicates(inplace=True)
        print(f'[INFO] number of documents after dedublication: {data.shape[0]}') 
        corpus = data[xcolumn]
        self.classnames = list(set(data[ycolumn]))
        for label in self.classnames:
            print(f'[INFO] instances of {label}: {sum(data[ycolumn] == label)}')
        self.X = self.cv.fit_transform(corpus.values)
        self.y = data[ycolumn].values

    def preprocess_data(self, test_size=0.2):
        """ Preprocess the data and split it into training and testing sets.
        
        Args:
            test_size (float): size of the testing set.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)

    def train(self):
        """ Train the classifier on the training data
        """
        self.classifier.fit(self.X_train , self.y_train)

    def evaluate(self, filename='cm.png'):
        """ Evaluate the model on the test data and print the confusion matrix.

        Args:
            filename (str): path to the output file.
        """
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        self.plot_confusion_matrix(cm, self.classnames, filename)
        print(f'[INFO] Relative Accuracy; {accuracy_score(self.y_test, y_pred)}')
        print(f'[INFO] Accuracy in instances {accuracy_score(self.y_test, y_pred, normalize=False)}')

    @staticmethod
    def plot_confusion_matrix(cm, class_names=None, filename='cm.png'):
        """
        Generate a confusion matrix visualization using seaborn heatmap.
        
        Args:
            cm (array, shape = [n, n]): a confusion matrix of integer classes
            class_names (array, shape = [n]): String names of the integer classes
        """
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        if class_names is not None:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(filename)


def main():
    """ Instantiate the class and call the methods in order.
    """
    set_random_seed()
    bec = BinaryClassifier()
    bec.load_data('dat/imdb_reviews.csv')
    bec.preprocess_data()
    bec.train()
    bec.evaluate(filename='figs/cm.png')

if __name__ == '__main__':
    main()