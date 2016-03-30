# -*-coding:utf-8-*-
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

def KNNClassifier(neighbors=5):
    characteristics = np.load('files/characteristics.npy')[:, [0, 1]]
    classifications = np.load('files/classification.npy')

    modelo = Pipeline([('normalizador', StandardScaler()),
                       ('knn', KNeighborsClassifier(
                        n_neighbors=neighbors))])
    
    kfold5 = KFold(characteristics.shape[0], 6, shuffle=True)
    valores = cross_val_score(
        modelo, characteristics, classifications, cv=kfold5)

    return valores
    
def SGDClassification():
    characteristics = np.load('files/characteristics.npy')[:, [0, 1]]
    classifications = np.load('files/classification.npy')

    modelo = Pipeline([('normalizador', StandardScaler()),
                       ('linearSGD', SGDClassifier())])

    kfold5 = KFold(characteristics.shape[0], 6, shuffle=True)
    valores = cross_val_score(
        modelo, characteristics, classifications, cv=kfold5)

    return valores


def LogisticRegressionClassification():
    characteristics = np.load('files/characteristics.npy')
    classifications = np.load('files/classification.npy')
    
    normalizador = StandardScaler()
    normalizador.fit(characteristics)
    characteristics_n = normalizador.transform(characteristics)
    
    modelo = LogisticRegression()
    characteristics_f_t = modelo.fit_transform(characteristics_n, classifications)
    
    print(np.nonzero(np.in1d(characteristics_f_t[0], characteristics_n[0])))
    
    modelo = Pipeline([('normalizador', StandardScaler()),
                       ('logistic', LogisticRegression())])

    kfold5 = KFold(characteristics_f_t.shape[0], 6, shuffle=True)
    valores = cross_val_score(
        modelo, characteristics_f_t, classifications, cv=kfold5)
    
    return valores


if __name__ == "__main__":
    best = (-1, 0)
    for i in range(5, 15):
        print("{0}/15".format(i))
        valores = np.max(KNNClassifier(i))
        if valores >= best[0]:
            best = (valores, i)

    print(best)
    
    valores = SGDClassification()
    print(np.max(valores))
    
    valores = LogisticRegressionClassification()
    print(np.max(valores))
    
    
