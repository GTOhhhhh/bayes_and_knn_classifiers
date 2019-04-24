import sys
import csv
import math
import numpy as np
from collections import Counter
from functools import lru_cache
from pprint import pprint


# times_pregnant,plasma_glucose,blood_pressure,skin_thickness,insulin_level,bmi,diabetes_pedigree,age,class

@lru_cache(maxsize=None)
def read(file):
    with open(file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [row for row in reader]


def gen_folds(file, k=10):
    """Return the data split into k stratified folds"""
    cnt = Counter([i[-1] for i in file])  # assumes class will be last column
    # generate tuples with the yes/no split depending on distribution e.g. (50, 27)
    sizes = zip(fold_sizes(cnt['no'], k), fold_sizes(cnt['yes'], k))
    yeses = [x for x in file if x[-1] == 'yes']
    nos = [x for x in file if x[-1] == 'no']
    folds = []
    for tup in sizes:
        fold = []
        for i in range(tup[0]):
            fold.append(nos.pop())
        for i in range(tup[1]):
            fold.append(yeses.pop())
        folds.append(np.asarray(fold))
    return np.asarray(np.asarray(folds))


def fold_sizes(N, k):
    r = N % k
    return [math.floor((N / k) + 1) for i in range(r)] + [math.floor(N / k) for i in range(k - r)]


def compute_class_stats(fold):
    """Generates a double header of tuples, one header for each class
       [(mean, sd), (...), (mean, sd), no],
       [(mean, sd), (...), (mean, sd), yes]
        i.e. we have (mean, sd) for each attribute, per class per fold"""
    no = fold[fold[:, -1] == 'no'][:, :-1].astype(np.float64)  # filter on no, slice off the class, cast as float
    yes = fold[fold[:, -1] == 'yes'][:, :-1].astype(np.float64)
    no_mean = np.mean(no, axis=0)
    # print(no_mean.shape)
    yes_mean = np.mean(yes, axis=0)
    no_std = np.std(no, axis=0)
    yes_std = np.std(yes, axis=0)
    # zip (transpose) the (mean, ...) & (sd, ...) into (mean, sd), ... then add tag
    no_row = np.array([no_mean, no_std]).T
    no_row = np.vstack((no_row, ['no', 'no']))
    yes_row = np.array([yes_mean, yes_std]).T
    yes_row = np.vstack((yes_row, ['yes', 'yes']))
    # yes_row = np.append(np.array([yes_mean, yes_std]).T, 'yes')
    print(yes_row.shape)
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
    # header.append(row)
    row1 = np.asarray(row1)
    row2 = np.asarray(row2)
    header = np.vstack((row1, row2))
    # pprint(fold.shape)
    # fold = np.array()
    pprint(fold.shape)
    # pprint(fold)
    # fold = np.vstack((header, fold))
    # pprint(header)
    # pprint(no_row)
    return fold


def calc_probability(x, mean, std):
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp


def gen_probabilities(folds):
    prob_folds = []
    for row in folds[2:]:
        new_row = []
        for idx, val in enumerate(row[:-1]):
            if row[-1] == 'no':
                mean = folds[0][idx][0]
                sd = folds[0][idx][1]
            else:
                mean = folds[1][idx][0]
                sd = folds[1][idx][1]
            probability = calc_probability(val, mean, sd)
            new_row.append(probability)
        new_row.append(row[-1])
        prob_folds.append(new_row)
    return prob_folds


if __name__ == '__main__':
    train_file = sys.argv[1]
    algo = sys.argv[2]
    if sys.argv[2] == 'NB':
        pass
    else:
        # otherwise run KNN and grab n from the end
        n = sys.argv[2][2]
        pass

    # print([i[0] for i in read(train_file)])
    # pprint([i for i in gen_folds(read(train_file))[0]])
    folds = gen_folds(read(train_file))
    compute_class_stats(folds[0])
    # pprint(gen_probabilities(folds))
