"Rui Xu 005230642"
"second try"

"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.   
## I will use the "@" symbols to refer to some variables and functions. 
## For example, for the 3 lines of code below
## x = 2
## y = x * 2 
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE. 

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not 
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not
        majority_value = Counter(y).most_common(1)[0][0]
        majority_num = Counter(y).most_common(1)[0][1]
        p = majority_num*1.0/X.shape[0]
        if (majority_value == 1):
            p_survival = p
            p_not = 1-p
        else:
            p_not = p
            p_survival = 1-p
        self.probabilities_ = [p_survival,p_not]

        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_ 

        y = np.random.choice([1,0],size = X.shape[0],p = self.probabilities_)

        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set 
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction
    
    train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 
    
    for i in range(1,(ntrials+1)):
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = test_size, random_state = i) 
        clf.fit(xtrain, ytrain)
        y_pred_train = clf.predict(xtrain)
        y_pred_test = clf.predict(xtest)
        train_score = 1 - metrics.accuracy_score(ytrain, y_pred_train, normalize=True)
        test_score = 1 - metrics.accuracy_score(ytest, y_pred_test, normalize=True)
        train_scores = train_scores+[train_score]
        test_scores = test_scores+[test_score]
    
    train_error = np.mean(train_scores) ## average error over all the @ntrials
    test_error = np.mean(test_scores)
    

    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf2 = RandomClassifier()
    clf2.fit(X, y)
    y_pred2 = clf2.predict(X)
    train_error2 = 1 - metrics.accuracy_score(y, y_pred2, normalize=True)
    print('\t-- training error: %.3f' % train_error2)

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    # call the function @DecisionTreeClassifier
    clf3 = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)
    clf3.fit(X, y)
    y_pred3 = clf3.predict(X)
    train_error3 = 1 - metrics.accuracy_score(y, y_pred3, normalize=True)
    print('\t-- training error: %.3f' % train_error3)

    ### ========== TODO : END ========== ###



    # note: uncomment out the following lines to output the Decision Tree graph
    
    # save the classifier -- requires GraphViz and pydot
    from io import StringIO
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.38/bin'
    import pydotplus
    from sklearn import tree
    dot_data = StringIO()
    tree.export_graphviz(clf3, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors...')
    # call the function @KNeighborsClassifier
    clf4_3 = KNeighborsClassifier(n_neighbors = 3)
    clf4_3.fit(X, y)
    y_pred4_3 = clf4_3.predict(X)
    train_error4_3 = 1 - metrics.accuracy_score(y, y_pred4_3, normalize=True)
    print('\t-- training error of 3-Nearest Neighbors : %.3f' % train_error4_3)
    
    clf4_5 = KNeighborsClassifier(n_neighbors = 5)
    clf4_5.fit(X, y)
    y_pred4_5 = clf4_5.predict(X)
    train_error4_5 = 1 - metrics.accuracy_score(y, y_pred4_5, normalize=True)
    print('\t-- training error of 5-Nearest Neighbors : %.3f' % train_error4_5)
    
    clf4_7 = KNeighborsClassifier(n_neighbors = 7)
    clf4_7.fit(X, y)
    y_pred4_7 = clf4_7.predict(X)
    train_error4_7 = 1 - metrics.accuracy_score(y, y_pred4_7, normalize=True)
    print('\t-- training error of 7-Nearest Neighbors : %.3f' % train_error4_7)
    

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error
    clf_train, clf_test = error(MajorityVoteClassifier(), X, y)
    print('\t-- average training error of Majority Vote : %.3f' % clf_train)
    print('\t-- average test error of Majority Vote : %.3f' % clf_test)
    
    clf2_train, clf2_test = error(RandomClassifier(), X, y)
    print('\t-- average training error of Random : %.3f' % clf2_train)
    print('\t-- average test error of Random : %.3f' % clf2_test)
    
    clf3_train, clf3_test = error(DecisionTreeClassifier(criterion = 'entropy', max_depth = 4), X, y)
    print('\t-- average training error of Decision Tree : %.3f' % clf3_train)
    print('\t-- average test error of Decision Tree : %.3f' % clf3_test)
    
    clf4_train, clf4_test = error(KNeighborsClassifier(n_neighbors = 5), X, y)
    print('\t-- average training error of 5-Nearest Neighbors : %.3f' % clf4_train)
    print('\t-- average test error of 5-Nearest Neighbors : %.3f' % clf4_test)
    

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))
    cv_score = [] ## track accuracy for each value of $k, should have length 25 once you're done
    cv_error = []
    for i in k:
        clf_find = KNeighborsClassifier(n_neighbors = i)
        score = cross_val_score(clf_find, X, y, cv = 10, scoring = 'accuracy')
        cv_score = cv_score+[sum(score)/10.0]
        cv_error = cv_error+[1-(sum(score)/10.0)]
    plt.figure()
    plt.plot(k, cv_score)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()
    
    plt.figure()
    plt.plot(k, cv_error)
    plt.xlabel('k')
    plt.ylabel('error')
    plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    d = list(range(1,21))
    cv_train_error = []
    cv_test_error = []
    for i in d:
        clf_find2 = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
        test_score = cross_val_score(clf_find2, X, y, cv = 10, scoring = 'accuracy')
        cv_test_error = cv_test_error+[1-(sum(test_score))/10.0]
        clf_find2 = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
        clf_find2.fit(X,y)
        y_pred_all = clf_find2.predict(X)
        train_error = 1 - metrics.accuracy_score(y, y_pred_all, normalize=True)
        cv_train_error = cv_train_error+[train_error]    
    plt.figure()
    plt.plot(d, cv_train_error, color = "r", label = "average training error")
    plt.plot(d, cv_test_error, color = "b", label = "average test error")
    plt.xlabel('depth limit')
    plt.ylabel('error')
    plt.legend(loc = "best")
    plt.show()

    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    k_best = 7
    d_best = 6
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.1)
    size = np.arange(0.1,1.1,0.1)
    decision_train_result = []
    decision_test_result = []
    knn_train_result = []
    knn_test_result = []
    split_time = list(range(1,101))
    
    for i in size[0:9]:
        decision_train_score = []
        decision_test_score = []
        neighbors_train_score = []
        neighbors_test_score = []
        for j in split_time:
            
            x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size = (1-i), random_state = j)
        
            clf_decision = DecisionTreeClassifier(criterion = 'entropy', max_depth = d_best)
            clf_decision.fit(x_train, y_train)
            y_pred_d1 = clf_decision.predict(x_train)
            y_pred_d2 = clf_decision.predict(xtest)
            decision_train_error = 1 - metrics.accuracy_score(y_train, y_pred_d1, normalize=True)
            decision_test_error  = 1 - metrics.accuracy_score(ytest, y_pred_d2, normalize=True)
            decision_train_score = decision_train_score+[decision_train_error]
            decision_test_score = decision_test_score+[decision_test_error]
        
            clf_neighbors = KNeighborsClassifier(n_neighbors = k_best)
            clf_neighbors.fit(x_train, y_train)
            y_pred_n1 = clf_neighbors.predict(x_train)
            y_pred_n2 = clf_neighbors.predict(xtest)
            neighbors_train_error = 1 - metrics.accuracy_score(y_train, y_pred_n1, normalize=True)
            neighbors_test_error = 1 - metrics.accuracy_score(ytest, y_pred_n2, normalize=True)
            neighbors_train_score = neighbors_train_score+[neighbors_train_error]
            neighbors_test_score = neighbors_test_score+[neighbors_test_error]
        
        decision_train_result = decision_train_result + [np.mean(decision_train_score)]
        decision_test_result = decision_test_result + [np.mean(decision_test_score)]
        knn_train_result = knn_train_result + [np.mean(neighbors_train_score)]
        knn_test_result = knn_test_result + [np.mean(neighbors_test_score)]
    
    clf_decision = DecisionTreeClassifier(criterion = 'entropy', max_depth = d_best)
    clf_decision.fit(xtrain, ytrain)
    y_pred_d1 = clf_decision.predict(xtrain)
    y_pred_d2 = clf_decision.predict(xtest)
    decision_train_error = 1 - metrics.accuracy_score(ytrain, y_pred_d1, normalize=True)
    decision_test_error = 1 - metrics.accuracy_score(ytest, y_pred_d2, normalize=True)
    decision_train_result = decision_train_result+[decision_train_error]
    decision_test_result = decision_test_result+[decision_test_error]
     
    clf_neighbors = KNeighborsClassifier(n_neighbors = k_best)
    clf_neighbors.fit(xtrain, ytrain)
    y_pred_n1 = clf_neighbors.predict(xtrain)
    y_pred_n2 = clf_neighbors.predict(xtest)
    neighbors_train_error = 1 - metrics.accuracy_score(ytrain, y_pred_n1, normalize=True)
    neighbors_test_error = 1 - metrics.accuracy_score(ytest, y_pred_n2, normalize=True)
    knn_train_result = knn_train_result+[neighbors_train_error]
    knn_test_result = knn_test_result+[neighbors_test_error]
    
    plt.figure()
    plt.plot(size, decision_train_result,  color = "r", label = "decision tree training error")
    plt.plot(size, decision_test_result, '--', color = "r", label = "decision tree test error")
    plt.xlabel('amount of training data')
    plt.ylabel('error')
    plt.legend(loc = "best")
    plt.show()
    
    plt.figure()
    plt.plot(size, knn_train_result,  color = "b", label = "knn training error")
    plt.plot(size, knn_test_result, '--', color = "b", label = "knn test error")
    plt.xlabel('amount of training data')
    plt.ylabel('error')
    plt.legend(loc = "best")
    plt.show()
        
    

    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
