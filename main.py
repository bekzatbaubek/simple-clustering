import multiprocessing as mp
import random
import time
from functools import partial
from math import sqrt


def distance(l1, l2):
    return sqrt(sum([(x[0] - x[1]) ** 2 for x in list(zip(l1, l2))]))


def accuracy(labels, predictions):
    correct = sum([x[0] == x[1] for x in list(zip(labels, predictions))])
    return round(correct / len(labels), 4)


def mean(data_points):
    ans = [0] * len(data_points[0])
    for d in data_points:
        for i in range(len(d)):
            ans[i] += d[i]
    return [x / len(data_points) for x in ans]


class KMeans:
    def __init__(self, k=2, seed=None, maxiterations=10):
        self.k = k
        self.random_seed = seed
        self.maxiterations = maxiterations
        random.seed(self.random_seed)

    def train(self, train_tuples):
        self.centroids = [(point, label) for point, label in random.sample(train_tuples, self.k)]
        newcent = self.centroids
        iterations = self.maxiterations
        for i in range(iterations):
            self.centroids = newcent
            assigned_centroids = []
            for point, label in train_tuples:
                closest = []
                for centroid in self.centroids:
                    closest.append((distance(point, centroid[0]), centroid[1]))
                assigned_centroids.append(min(closest)[1])

            dct = {}
            for data, centroid in list(zip(train_tuples, assigned_centroids)):
                if centroid in dct:
                    dct[centroid].append(data)
                else:
                    dct[centroid] = [data]

            newcent = []
            for label, points in dct.items():
                newcent.append((mean([t[0] for t in points]), label))

        finalcent = []
        for centroid in self.centroids:
            lst = [b for (a, b) in dct[centroid[1]]]
            most_common = max(lst, key=lst.count)
            finalcent.append((centroid[0], most_common))
        self.centroids = finalcent

        return None

    def predict(self, data_point):
        return min([(distance(data_point, x[0]), x[1]) for x in self.centroids])[1]


def KNN(data_train, data_test, k=1):
    distances = sorted([(distance(data_test, point), label) for point, label in data_train])
    dct = {}
    for x in distances[:k]:
        if x[1] in dct:
            dct[x[1]] += 1
        else:
            dct[x[1]] = 1
    return max(dct.items(), key=lambda k: k[1])[0]


if __name__ == "__main__":
    with open("optdigits.tra", 'r') as f:
        train_lines = [[int(x) for x in line.strip().split(',')] for line in f.readlines()]

    with open("optdigits.tes", 'r') as f:
        test_lines = [[int(x) for x in line.strip().split(',')] for line in f.readlines()]

    train_tuples = [(x[:64], x[64]) for x in train_lines]
    test_labels = [x[64] for x in test_lines]
    test_data = [x[:64] for x in test_lines]

    pool = mp.Pool(mp.cpu_count())

    k_value = 1

    start = time.process_time()
    results = pool.map(partial(KNN, train_tuples, k=k_value), [x for x in test_data])
    end = time.process_time()
    time1 = end - start
    print(f'Time (s) for running KNN in parallel (k = {k_value}): {time1}')
    print(f'The accuracy: {accuracy(test_labels, results)}')

    start = time.process_time()
    results = [partial(KNN, train_tuples, k=k_value)(x) for x in test_data]
    end = time.process_time()
    time1 = end - start
    print(f'Time (s) for running KNN in series (k = {k_value}): {time1}')
    print(f'The accuracy: {accuracy(test_labels, results)}')

    kmeans = KMeans(k=10, seed=7160, maxiterations=25)

    start = time.process_time()
    kmeans.train(train_tuples)
    end = time.process_time()
    time1 = end - start
    print(f'Time for training K-Means: {time1}')

    start = time.process_time()
    results = [kmeans.predict(x) for x in test_data]
    end = time.process_time()
    time1 = end - start
    print(f'Time for predicting with K-Means: {time1}')
    print(f'The accuracy: {accuracy(test_labels, results)}')

    start = time.process_time()
    results = pool.map(kmeans.predict, test_data)
    end = time.process_time()
    time1 = end - start
    print(f'Time for predicting with K-Means in parallel: {time1}')
    print(f'The accuracy: {accuracy(test_labels, results)}')

    pool.close()
