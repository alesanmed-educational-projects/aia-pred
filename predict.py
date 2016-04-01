# -*-coding:utf-8-*-
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def KNNClassifier(to_predict, neighbors=5):
    characteristics = np.load('files/characteristics.npy')[:, [0,2]]
    classifications = np.load('files/classification.npy')
    
    scaler = StandardScaler().fit(characteristics, classifications)
    characteristics_n = scaler.transform(characteristics)
    to_predict_n = scaler.transform(to_predict)
    
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(characteristics_n, classifications)
    
    return classifier.predict(to_predict_n)
    
def SGDClassification(to_predict):
    characteristics = np.load('files/characteristics.npy')[:, [0, 4, 8]]
    classifications = np.load('files/classification.npy')

    scaler = StandardScaler().fit(characteristics, classifications)
    characteristics_n = scaler.transform(characteristics)
    to_predict_n = scaler.transform(to_predict)
    
    classifier = SGDClassifier()
    classifier.fit(characteristics_n, classifications)
    
    return classifier.predict(to_predict_n)


def LogisticRegressionClassification(to_predict):
    characteristics = np.load('files/characteristics.npy')[:, [0, 1, 2, 3, 4, 6, 8, 9]]
    classifications = np.load('files/classification.npy')

    scaler = StandardScaler().fit(characteristics, classifications)
    characteristics_n = scaler.transform(characteristics)
    to_predict_n = scaler.transform(to_predict)
    
    classifier = LogisticRegression()
    classifier.fit(characteristics_n, classifications)
    
    return classifier.predict(to_predict_n)


if __name__ == "__main__":
    # Respuesta valorada con 13 puntos, debe dar 1 en los 3 casos
    to_predict = np.array([10162, 1831, 0, 1, 11, 1, 7, 259.85714285714283, 342, 4.0614035087719298])
    
    print("KNN 12: {0}".format(KNNClassifier(to_predict[[0, 2]].reshape(1, -1), 12)))
    print("Gradient descent: {0}".format(SGDClassification(to_predict[[0, 4, 8]].reshape(1, -1))))
    print("Logistic regression: {0}".format(LogisticRegressionClassification(to_predict[[0, 1, 2, 3, 4, 6, 8, 9]].reshape(1, -1))))
    