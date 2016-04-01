# -*-coding:utf-8-*-
import itertools
import numpy as np
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


"""
Caracteristicas:
 reputation,
 len(body),
 body.count('img src'),
 body.count('a href'),
 occurrences,
 neg_occurences,
 len(sentences),
 np.mean(sentences_len),
 len(words),
 np.mean(words_len)
"""

def KNNClassifier(indices, neighbors=5):
    characteristics = np.load('files/characteristics.npy')[:, indices]
    classifications = np.load('files/classification.npy')

    modelo = Pipeline([('normalizador', StandardScaler()),
                       ('knn', KNeighborsClassifier(
                        n_neighbors=neighbors))])
    
    kfold5 = KFold(characteristics.shape[0], 6, shuffle=True)
    valores = cross_val_score(
        modelo, characteristics, classifications, cv=kfold5)

    return valores
    
def SGDClassification(indices):
    characteristics = np.load('files/characteristics.npy')[:, indices]
    classifications = np.load('files/classification.npy')

    modelo = Pipeline([('normalizador', StandardScaler()),
                       ('linearSGD', SGDClassifier())])

    kfold5 = KFold(characteristics.shape[0], 6, shuffle=True)
    valores = cross_val_score(
        modelo, characteristics, classifications, cv=kfold5)

    return valores


def LogisticRegressionClassification(indices):
    characteristics = np.load('files/characteristics.npy')[:, indices]
    classifications = np.load('files/classification.npy')
    
    modelo = Pipeline([('normalizador', StandardScaler()),
                       ('logistic', LogisticRegression())])

    kfold5 = KFold(characteristics.shape[0], 6, shuffle=True)
    valores = cross_val_score(
        modelo, characteristics, classifications, cv=kfold5)
    
    return valores


if __name__ == "__main__":
    indices_range = list(range(10))
    indices = [list(itertools.combinations(indices_range, i)) for i in range(2, 10)]
    indices_flat = []
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            indices_flat.append(list(indices[i][j]))
    
    indices = indices_flat
    
    best_neighbor = (-1, 0, 0)
    best_SGD = (-1, 0)
    best_logreg = (-1, 0)
    for i in range(len(indices_flat)):
        print("{0}/{1}".format(i + 1, len(indices_flat)))
        for j in range(5, 15):
            max_value = np.max(KNNClassifier(indices[i], j))
            if max_value >= best_neighbor[0]:
                best_neighbor = (max_value, indices[i], j)
        
        max_value = np.max(SGDClassification(indices[i]))
        if max_value >= best_SGD[0]:
            best_SGD = (max_value, indices[i])
        
        max_value = np.max(LogisticRegressionClassification(indices[i]))
        if max_value >= best_logreg[0]:
            best_logreg = (max_value, indices[i])
    
    print("Best KNN: ")
    print(best_neighbor)
    
    print("Best SGD: ")
    print(best_SGD)

    print("Best Log reg: ")
    print(best_logreg)