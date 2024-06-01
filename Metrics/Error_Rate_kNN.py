from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from numpy import mean


def error_rate_knn(X, labels):
    # perform cross validation using leave-one-out method
    model = KNeighborsClassifier()
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, labels, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    accuracy_mean = mean(scores)
    # get error rate
    error_rate = 1-accuracy_mean
    return 1-error_rate
