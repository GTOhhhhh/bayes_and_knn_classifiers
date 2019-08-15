import sys
import csv
import math
import numpy as np
import heapq
from collections import Counter
from statistics import mode


def read(file):
    with open(file) as f:
        reader = csv.reader(f)
        # next(reader)  # skip header
        return [row for row in reader]


def fold_sizes(N, k):
    r = N % k
    return [math.floor((N / k) + 1) for i in range(r)] + [math.floor(N / k) for i in range(k - r)]


def gen_folds(file, k=10):
    """Return the data split into k stratified folds"""
    cnt = Counter([i[-1] for i in file])  # assumes class will be last column
    # generate tuples with the yes/no split depending on distribution e.g. (50, 27)
    sizes = zip(fold_sizes(cnt['no'], k), fold_sizes(cnt['yes'], k))
    yeses = [x for x in file if x[-1] == 'yes']
    nos = [x for x in file if x[-1] == 'no']
    folds = []
    # with open('pima-folds.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     fold_no = 1
    #     for tup in sizes:
    #         fold = []
    #         for i in range(tup[0]):
    #             fold.append(nos.pop())
    #         for i in range(tup[1]):
    #             fold.append(yeses.pop())
    #         folds.append(np.asarray(fold))
    #         writer.writerow(['fold' + str(fold_no)])
    #         for row in fold:
    #             writer.writerow(row)
    #         fold_no += 1

    return np.asarray(np.asarray(folds))


def compute_class_stats(fold):
    """Generates a double header of tuples, one header for each class
       [(mean, sd), (...), (mean, sd), no],
       [(mean, sd), (...), (mean, sd), yes]
        i.e. we have (mean, sd) for each attribute, per class per fold"""
    if isinstance(fold, list):
        fold = np.asarray(fold)

    # filter on yes/no, slice off the class, cast as float
    no = fold[fold[:, -1] == 'no'][:, :-1].astype(np.float64)
    yes = fold[fold[:, -1] == 'yes'][:, :-1].astype(np.float64)
    no_mean = np.mean(no, axis=0, dtype=np.float64)
    yes_mean = np.mean(yes, axis=0, dtype=np.float64)
    no_std = np.std(no, axis=0, ddof=1, dtype=np.float64) if no.shape[0] > 1 else 0
    yes_std = np.std(yes, axis=0, ddof=1, dtype=np.float64) if yes.shape[0] > 1 else 0

    # zip (transpose) the (mean, ...) & (sd, ...) into (mean, sd), ... then add tag
    no_row = np.array([no_mean, no_std]).T
    no_row = np.vstack((no_row, ['no', 'no']))

    yes_row = np.array([yes_mean, yes_std]).T
    yes_row = np.vstack((yes_row, ['yes', 'yes']))

    # clean up numpy array into tuples
    row1 = []
    for i in no_row:
        if i[0] == 'yes' or i[0] == 'no':
            row1.append(i[0])
        else:
            row1.append((i[0], i[1]))

    row2 = []
    for i in yes_row:
        if i[0] == 'yes' or i[0] == 'no':
            row2.append(i[0])
        else:
            row2.append((i[0], i[1]))

    # combine
    row1 = np.asarray(row1)
    row2 = np.asarray(row2)
    header = np.vstack((row1, row2))
    fold = np.vstack((header, fold))
    return fold


def calc_probability(x, mean, std):
    if std == 0:
        return 1
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp


def NB_predict(train_data, test_data):
    no_head = train_data[0]
    yes_head = train_data[1]
    if isinstance(test_data, list):
        test_data = np.asarray(test_data)

    counts = Counter(train_data[2:, -1])
    p_no = counts['no'] / (counts['yes'] + counts['no'])
    p_yes = counts['yes'] / (counts['yes'] + counts['no'])

    if test_data[0][-1] == 'yes' or test_data[0][-1] == 'no':
        test_data = test_data[:, :-1]

    results = []
    for row in test_data:
        probs = []
        row = row.astype(np.float64)
        for idx, col in enumerate(row):
            no_prob = calc_probability(col, no_head[idx][0].astype(np.float64), no_head[idx][1].astype(np.float64))
            yes_prob = calc_probability(col, yes_head[idx][0].astype(np.float64), yes_head[idx][1].astype(np.float64))
            probs.append((no_prob, yes_prob))
        no_total = 1
        yes_total = 1

        for prob in probs:
            no_total *= prob[0]
            yes_total *= prob[1]
        results.append('yes' if yes_total * p_yes >= no_total * p_no else 'no')
    return results


def euclid(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def KNN_predict(train_data, row, n):
    nearest = []
    train_data = np.asarray(train_data)
    search_row = np.asarray(row)
    for row in train_data:
        new_dist = -euclid(search_row.astype(np.float64), row[:-1].astype(np.float64))
        if len(nearest) < n:
            heapq.heappush(nearest, (new_dist, row[-1]))
        else:
            if min(nearest)[0] < new_dist:
                heapq.heapreplace(nearest, (new_dist, row[-1]))
        heapq.heapify(nearest)
    try:
        return mode([i[1] for i in nearest])
    except ValueError:
        return 'yes'


def cross_validate(folds):
    """runs the algorithm with cross validation and displays the accuracy"""
    NB_accuracies = []
    oneNN_accuracies = []
    fiveNN_accuracies = []
    for i in range(10):
        test_data_classes = folds[i][:, -1]
        test_data = folds[i][:, :-1]  # remove class
        train_data = np.concatenate([folds[:i], folds[i + 1:]])
        new_train_data = []
        for data in train_data:
            for subdata in data:
                new_train_data.append(subdata)
        new_train_data = np.asarray(new_train_data)
        NB_predictions = NB_predict(compute_class_stats(new_train_data), test_data)
        one_NN_predictions = []
        five_NN_predictions = []
        for row in test_data:
            one_NN_predictions.append(KNN_predict(new_train_data, row, 1))
            five_NN_predictions.append(KNN_predict(new_train_data, row, 5))
        predictions = [NB_predictions, one_NN_predictions, five_NN_predictions]

        for idx, predicts in enumerate(predictions):
            result = []
            for idx1, item in enumerate(predicts):
                if test_data_classes[idx] == item:
                    result.append(True)
                else:
                    result.append(False)
            counts = Counter(result)
            accuracy = counts[True] / (counts[False] + counts[True])
            if idx == 0:
                NB_accuracies.append(accuracy)
            elif idx == 1:
                oneNN_accuracies.append(accuracy)
            elif idx == 2:
                fiveNN_accuracies.append(accuracy)
    print('NB Cross-validated accuracy: ', np.around(np.mean(NB_accuracies) * 100, 4), '%')
    print('1NN Cross-validated accuracy: ', np.around(np.mean(oneNN_accuracies) * 100, 4), '%')
    print('5NN Cross-validated accuracy: ', np.around(np.mean(fiveNN_accuracies) * 100, 4), '%')


if __name__ == '__main__':
    train_data = read(sys.argv[1])
    test_data = read(sys.argv[2])
    folds = gen_folds(train_data)

    if sys.argv[3] == 'NB':
        for i in NB_predict(compute_class_stats(train_data), test_data):
            print(i)
    elif sys.argv[3] == 'acc':
        cross_validate(folds)
    else:
        # otherwise run KNN and grab n from the start
        n = int(sys.argv[3][0])
        for row in test_data:
            print(KNN_predict(train_data, row, n))
