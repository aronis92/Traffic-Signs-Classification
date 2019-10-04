import pickle
from pandas.io.parsers import read_csv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image

#Define the function that loads the data
def load_pickled_data(file, columns):
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))

#Read the signnames from the csv file
signnames = read_csv("signnames.csv").values[:, 1]

#Define the paths for the datasets
train_dataset_file = "traffic-signs-data/train.p"
test_dataset_file = "traffic-signs-data/test.p"
valid_dataset_file = "traffic-signs-data/valid.p"

#Define the variables for X and y
X_train, y_train = load_pickled_data(train_dataset_file, ['features', 'labels'])
X_valid, y_valid = load_pickled_data(valid_dataset_file, ['features', 'labels'])
X_test, y_test = load_pickled_data(test_dataset_file, ['features', 'labels'])

#Define the variables that contain the shapes and more 
n_train = y_train.shape[0]
n_test = y_test.shape[0]
n_valid = y_valid.shape[0]
image_shape = X_train[0].shape
image_size = image_shape[0]
sign_classes, class_indices, class_counts = np.unique(y_train, return_index = True, return_counts = True)
n_classes = class_counts.shape[0]

##Print quick data summary
#print("Number of training examples =", n_train)
#print("Number of testing examples =", n_test)
#print("Number of validation examples: ", n_valid)
#print("Image data shape =", image_shape)
#print("Number of classes =", n_classes)
#
##Define the function that plots the number of instances per class
#def plot_instance_counts(dataset, name="dataset"):
#    from collections import Counter
#    counts = Counter(dataset)
#    labels, values = zip(*counts.items())
#    indexes = np.arange(len(labels))
#    width = 0.5
#    with plt.style.context(('seaborn-muted')):
#        figure = plt.figure(figsize=(12, 3))
#        plt.bar(indexes, values, width)
#        plt.xticks(indexes + width * 0.5, labels)
#        plt.xlabel('Class Label')
#        plt.title('{} : Number of instance per class'.format(name))
#        plt.show()
#
#plot_instance_counts(y_train, 'Training Data')
#
##Define the function that returns the indices of images that belong to a given class
#def indices_for_class(class_label, labels=y_train):
#    return np.where(labels == [class_label])[0]
#
## Plot classes
#try:
#    os.mkdir('./figs')
#except:
#    print('Folder exists')
#with open(train_dataset_file, mode='rb') as f:
#    images = pickle.load(f)['features']
#    n = 5
#    img = list([])
#    plt.rcParams["figure.figsize"] = [8,2]
#    fig, axs = plt.subplots(1, 5, constrained_layout=True)
#
#    for cl in range(n_classes):
#        indices = np.random.choice(indices_for_class(cl), n)
#        for i, index in enumerate(indices):
#            axs[i].imshow(images[index], interpolation='nearest', cmap='jet')
#            axs[i].axis('off')
#        plt.suptitle('Images for class: {}'.format(signnames[cl]))
#        plt.savefig('./figs/class'+str(cl)+'.png')
#    plt.close()
#
#image_list = []
#url = './figs/*.png'
#for filename in glob.glob(url):
#    image_list.append(Image.open(filename))
#
#w, h = image_list[0].size
#
#for j in range(8):
#    offset = 0
#    new_im = Image.new('RGB', (w, h*5))
#    for i in range(5):
#      new_im.paste(image_list[i + j*5], (0,offset))
#      offset += image_list[i].size[1]
#    new_im.save('./classes'+str(i + j*5 - 4)+'-'+str(i + j*5)+'.png')
#offset = 0
#new_im = Image.new('RGB', (w, h*3))
#for i in range(3):
#  new_im.paste(image_list[40 + i], (0,offset))
#  offset += image_list[i].size[1]
#new_im.save('./classes40-42.png')
#
##for i in range(43):
##    os.remove('./class'+str(i)+'.png')
#
##Define the function that plots 5 sample images for each classs
#def plot_n_images_for_class(class_label, n=5, dataset_source=train_dataset_file, labels=y_train, cmap='jet'):
#    with open(dataset_source, mode='rb') as f:
#        images = pickle.load(f)['features']
#        indices = np.random.choice(indices_for_class(class_label), n)
#        figure = plt.figure(figsize = (6,6))
#
#        for i, index in enumerate(indices):
#            a = figure.add_subplot(n,n, i+1)
#            img = plt.imshow(images[index], interpolation='nearest', cmap=cmap)
#            plt.axis('off')
#        plt.suptitle('images for class {}'.format(class_label))
#        plt.show()
#
#for i in np.arange(43):
#    plot_n_images_for_class(i)
#
##Define the function that plots the number of instances per class for a given subset
#def histogram_plot(dataset, label):
#    hist, bins = np.histogram(dataset, bins=n_classes)
#    width = 0.7 * (bins[1] - bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    plt.bar(center, hist, align='center', width=width)
#    plt.xlabel(label)
#    plt.ylabel("Image count")
#    plt.show()
#    
##Plotting instance counts for each subset
#histogram_plot(y_train, "Training examples")
#histogram_plot(y_test, "Testing examples")
#histogram_plot(y_valid, "Validation examples")