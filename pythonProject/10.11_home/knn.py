import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

class KNN:
    """
    k-NN классификатор

    Возвращает: предсказания k-NN
    """

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.k = None
        self.predictions = []

    @staticmethod
    def _dist(a, b):
        """
        Расстояние евклида
        Принимает на вход два вектора

        Возвращает: число с плавающей точкой
        """
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def score(y_test, predictions):
        """
        Принимает на вход два массива с данными: тестовый и предсказанный

        Возвращает: число с плавающей точкой
        """
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == predictions[i]:
                correct += 1
        return correct / len(y_test)

    def plot(self, y_test):
        colors = []
        for c in y_test:
            if c == "Iris-setosa":
                colors.append(0)
            elif c == "Iris-versicolor":
                colors.append(1)
            else:
                colors.append(2)

        plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)
        plt.title("Real Data")
        plt.show()

        colors = []
        i = 0
        for c in self.predictions:
            if c == "Iris-setosa":
                colors.append(0)
            elif c == "Iris-versicolor":
                colors.append(1)
            else:
                colors.append(2)

        plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)
        plt.title("Predicted")
        plt.show()

    def fit(self, X_train, y_train, k):
        """
        Принимает на вход два массива с данными: тренировочный Х и тренировочные лейблы
        и k - количество соседей

        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        """
        Принимает на вход двумерный массив искомых точек

        Возвращает: список предсказаний
        """

        for i in range(len(X_test)):
            distances = []
            targets = {}

            for j in range(len(X_train)):
                # пройдем по всем точкам и посчитаем расстояние до них от тестовой точки
                distances.append([self._dist(X_test[i], X_train[j]), j])

            # отсортируем расстояния
            distances = sorted(distances)

            # создадим словарь с k ближайщими значениями
            for j in range(self.k):
                index = distances[j][1]
                if targets.get(y_train[index]) != None:
                    targets[y_train[index]] += 1
                else:
                    targets[y_train[index]] = 1

            # вернем самую часто встречающаюся метку
            self.predictions.append(max(targets, key=targets.get))

        return self.predictions


def normalize(a):
    row_sums = a.sum(axis=1)
    new_matrix = a
    for i, (row, row_sum) in enumerate(zip(a, row_sums)):
        new_matrix[i, :] = row / row_sum
    return new_matrix

def normalize_data(dataset):
    num_features = len(dataset)
    for i in range(4):
        column_values = [row[i] for row in dataset]
        column_min = min(column_values)
        column_max = max(column_values)

        for row in dataset:
            row[i] = (row[i] - column_min) / (column_max - column_min)

def dist (a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


def predict(X_train, y_train, x_test, k):
    # создадим список для расстояний и словарь k-значений рядом
    distances = []
    targets = {}

    for i in range(len(X_train)):
        # пройдем по всем точкам и посчитаем расстояние до них от тестовой точки
        distances.append([dist(x_test, X_train[i]), i])

    # отсортируем расстояния
    distances = sorted(distances)

    # создадим словарь с k ближайщими значениями
    for i in range(k):
        index = distances[i][1]
        if targets.get(y_train[index]) != None:
            targets[y_train[index]] += 1
        else:
            targets[y_train[index]] = 1

    # вернем самую часто встречающаюся метку
    return max(targets, key=targets.get)

def euclidean_distance(x_train, x_test_point):
    distances = []
    # рассчитываю расстояния до каждой точки в наборе обучающих данных
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = 0

        for col in range(len(current_train_point)):
            current_distance += (current_train_point[col] - x_test_point[col]) ** 2
        current_distance = np.sqrt(current_distance)
        distances.append(current_distance)

    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances

def nearest_neighbors(distance_point, k):
    # сортировка расстояний всех точек
    knearests = distance_point.sort_values(by=['dist'], axis=0)

    # беру ближайщих k соседей
    knearests = knearests[:k]
    return knearests


def most_common(k_nearest, y_train):
    # получаю индексы видов соседей c их повторениями
    common_types = Counter(y_train[k_nearest.index])
    # определяю самый популярный вид среди соседей
    prediction = common_types.most_common()[0][0]
    return prediction


def knn_iris(x_train, y_train, x_test, k):
    prediction = []

    for x_test_point in x_test:
        # получаю все расстояния от точек до точек
        distance_point = euclidean_distance(x_train, x_test_point)
        # определяю ближайшие точки
        nearest_point = nearest_neighbors(distance_point, k)
        # ближайшие популярные точки
        pred_point = most_common(nearest_point, y_train)
        prediction.append(pred_point)

    return prediction


# функция для подсчета точности вычислений
def calculate_accuracy(y_test, y_pred):
    correct_count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct_count = correct_count + 1
    accuracy = correct_count / len(y_test)
    return accuracy



iris = datasets.load_iris()
targets = iris.target
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = targets

x = iris.get('data')
sns.pairplot(df, hue="Species", size =2.0,palette=['#006400', '#FF1493', '#DC143C'])
plt.show()

normalize_data(x)
df.data = x
sns.pairplot(df, hue="Species", size =2.0,palette=['#006400', '#FF1493', '#DC143C'])
plt.show()

y = iris.get('target')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n = X_test.shape[0]
metricas = []
for i in range(1, 10):
    knn = KNN()
    knn.fit(X_train, y_train, i)
    pred = knn.predict(X_test)
    metricas.append(knn.score(y_test,pred))
    print("k = " + str(i), ", Score: " + str(knn.score(y_test,pred)))

knn = KNN()
knn.fit(X_train,y_train,8)

pred = knn.predict(X_test)
print(pred)
print(knn.score(y_test,pred))

norm_x_train = normalize(X_train)
norm_x_test = normalize(X_test)



new_row = [[0.3	,0.3,	0.3,	0.3]]
test = pd.DataFrame(new_row)
test = np.asarray(test)
test_array = normalize(test)

di = {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}
k = 8
predictions = knn_iris(norm_x_train, y_train, test_array, k)
for i in predictions:
    print(di[i])