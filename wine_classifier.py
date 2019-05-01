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
from scipy import spatial
from utilities import load_data, print_features, print_predictions
from sklearn.decomposition import PCA

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
    
    return [6, 9]

def feature_sel_plot_3d(train_set, train_labels):
    features = feature_selection(train_set, train_labels)

    n_features = train_set.shape[1]
    fig, ax = plt.subplots(4, 6)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    class_1_colour = r'#3366ff'
    class_2_colour = r'#cc3300'
    class_3_colour = r'#ffc34d'

    class_colours = [class_1_colour, class_2_colour, class_3_colour]

    x = 0
    for i in range(13):
        if i not in features:
            for j in range(2):
                a = x % 6
                b = x // 6

                ax[b, a].set_title('Features {} vs {}'.format(i, features[j]))

                for n in range(len(train_labels)):
                    ax[b, a].scatter(train_set[n, i], train_set[n, features[j]], color = class_colours[int(train_labels[n]) - 1], s=0.5)

                x += 1

    plt.show()



def run_knn(train_set, train_labels, test_set, k):
    results = []

    for point in test_set:
        distances = [ spatial.distance.euclidean(point, neighbor) for neighbor in train_set ]

        neighbors = []
        for i in range(k):
            closest = np.argmin(distances)

            neighbors.append(train_labels[closest])
            distances[closest] = float("inf")

        counts = np.bincount(neighbors)
        most_common = np.argmax(counts)

        results.append(most_common)

    return results

def knn(train_set, train_labels, test_set, k, **kwargs):
    return run_knn(train_set, train_labels, test_set, k)

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
    return run_knn(train_set, train_labels, test_set, k)


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    pca = PCA(n_components = 2)

    pca.fit(train_set)
    train_set_r = pca.transform(train_set)
    test_set_r = pca.transform(test_set)


    # fig, ax = plt.subplots()
    # class_1_colour = r'#3366ff'
    # class_2_colour = r'#cc3300'
    # class_3_colour = r'#ffc34d'

    # class_colours = [class_1_colour, class_2_colour, class_3_colour]

    # for n in range(len(train_labels)):
    #     ax.scatter(reduced[n, 0], reduced[n, 1], color = class_colours[int(train_labels[n]) - 1], s=1)
    # plt.show()


    # write your code here and make sure you return the predictions at the end of 
    # the function
    return run_knn(train_set_r, train_labels, test_set_r, k)


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
        features = feature_selection(train_set, train_labels)
        train_set = np.column_stack((train_set[:, features[0]], train_set[:, features[1]]))
        test_set = np.column_stack((test_set[:, features[0]], test_set[:, features[1]]))

        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)

        #print(percent_correct(predictions, test_labels))
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)

        # print(percent_correct(predictions, test_labels))
        # cm = calculate_confusion_matrix(test_labels, predictions)
        # plot_matrix(cm)
    elif mode == 'knn_3d':
        features = feature_selection(train_set, train_labels)
        features.append(10)

        train_set = np.column_stack((train_set[:, features[0]], train_set[:, features[1]], train_set[:, features[2]]))
        test_set = np.column_stack((test_set[:, features[0]], test_set[:, features[1]], test_set[:, features[2]]))

        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)

        #print(percent_correct(predictions, test_labels))
    elif mode == 'knn_pca':
        predictions = knn_pca(train_set, train_labels, test_set, args.k)

        print_predictions(predictions)
        #print(percent_correct(predictions, test_labels))
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
