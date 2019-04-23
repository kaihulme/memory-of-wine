#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    


def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of 
    # the function

##    n_features = train_set.shape[1]
##    fig, ax = plt.subplots(8, 10)
##    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
##
##    class_1_colour = r'#3366ff'
##    class_2_colour = r'#cc3300'
##    class_3_colour = r'#ffc34d'
##
##    class_colours = [class_1_colour, class_2_colour, class_3_colour]
##
##    x = 0
##    # write your code here
##    for i in range(13):
##        for j in range(13):
##            if i > j:
##                x += 1
##                ax[x // 10, x % 10].set_title('Features {} vs {}'.format(i + 1, j + 1))
##
##                for n in range(len(train_labels)):
##                    ax[x // 10, x % 10].scatter(train_set[n, i], train_set[n, j], color = class_colours[int(train_labels[n]) - 1], s=0.5)
##                
##    plt.show()
    
    return [12, 9]


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def knn(train_set, train_labels, test_set, k, **kwargs):
    features = feature_selection(train_set, train_labels)
    
    train_set = np.column_stack((train_set[:, features[0]], train_set[:, features[1]]))
    test_set = np.column_stack((test_set[:, features[0]], test_set[:, features[1]]))

    centroids = np.zeros((3, 2))
    for i in range(2):
        for c in range(1, 4):
            centroids[c - 1, i] = np.mean(train_set[train_labels == c, i])

    results = []

    for set in test_set:
        distances = [dist(set, p) for p in train_set]

        closest = []

        for i in range(1, k + 1):
            index = np.argmin(distances)
            closest.append(train_labels[index] - 1)
            distances[index] = float("inf")

        tied = True
        counts = []
        while tied:
            counts = np.bincount(closest)
            
            most_common = np.max(counts)
            if np.count_nonzero(counts == most_common) > 1:
                closest = closest[:-1]
            else:
                tied = False

        label = np.argmax(counts) + 1
        
        results.append(label)

    correct = 0
    
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return results

def calculate_confusion_matrix(gt_labels, pred_labels):
    gt_labels = gt_labels.astype(int)
    pred_labels = np.asarray(pred_labels)
    
    cm = np.zeros((3, 3))
    
    num_classes = max(len(set(gt_labels)), len(set(pred_labels)))
   
    for i in range(num_classes):
        gt = gt_labels[gt_labels == i + 1]
        pred = pred_labels[gt_labels == i + 1]
        
        for j in range(num_classes):
            cm[i, j] = len(pred[pred == j + 1]) / len(gt)
            
    
    return cm

def plot_matrix(matrix, ax=None):
    if ax is None:
        ax = plt.gca()
    
    handle = plt.imshow(matrix, cmap = plt.get_cmap('summer'))
    plt.colorbar(handle)
    
    for (j, i), label in np.ndenumerate(matrix):
        ax.text(i, j, "{:10.4f}".format(label), ha='center', va='center')

    plt.show()


def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    features = feature_selection(train_set, train_labels)
    
    train_set = np.column_stack((train_set[:, features[0]], train_set[:, features[1]]))
    test_set = np.column_stack((test_set[:, features[0]], test_set[:, features[1]]))


    class_data = []
    for i in range(1, 4):
        class_data.append(train_set[train_labels == i])


    means = np.zeros((3, 2))
    stdevs = np.zeros((3, 2))
    
    for i in range(len(class_data)):
        data = class_data[i]
        means[i, 0] = data[:, 0].mean()
        means[i, 1] = data[:, 1].mean()

        stdevs[i, 0] = data[:, 0].std()
        stdevs[i, 1] = data[:, 1].std()


    probabilities = np.ones((len(test_set), 3))
    for i in range(len(test_set)):
        data = test_set[i]

        for attr in range(2):
            for j in range(3):
                p = stats.norm.pdf(data[attr], loc = means[j, attr], scale = stdevs[j, attr])
                probabilities[i, j] *= p


    results = []
    for p in probabilities:
        results.append(np.argmax(p) + 1)
    
    
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return results


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def percent_correct(pred_labels, gt_labels):
    correct = 0
    
    for i in range(len(pred_labels)):
        if pred_labels[i] == gt_labels[i]:
            correct += 1

    return correct / len(pred_labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
        
        ## cm = calculate_confusion_matrix(test_labels, predictions)
        ## plot_matrix(cm)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)

        print(percent_correct(predictions, test_labels))
        cm = calculate_confusion_matrix(test_labels, predictions)
        plot_matrix(cm)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
