# -*-coding:utf-8-*-
import numpy as np
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def KNNClassifier(neighbors=5):
    neigh = KNeighborsClassifier(n_neighbors=neighbors)

    characteristics = np.load('files/characteristics.npy')
    classifications = np.load('files/classification.npy')
    neigh.fit(characteristics, classifications)

    modelo = Pipeline([('normalizador', StandardScaler()),
                       ('knn', KNeighborsClassifier(
                        n_neighbors=neighbors))])

    kfold5 = KFold(characteristics.shape[0], 6, shuffle=True)
    valores = cross_val_score(
        modelo, characteristics, classifications, cv=kfold5)

    return valores


if __name__ == "__main__":
    best = (-1, 0)
    for i in range(5, 15):
        print("{0}/15".format(i), end='\r')
        valores = np.max(KNNClassifier(i))
        if valores >= best[0]:
            best = (valores, i)

    print(best)
