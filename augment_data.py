from pandas.io.parsers import read_csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_pickled_data(file, columns):
    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))

def histogram_plot(dataset, label):
    hist, bins = np.histogram(dataset, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()

def apply_bilateral_filtering(data, label):
    new_data = []
    new_labels = []
    for i in range(data.shape[0]):
        img = cv2.bilateralFilter(data[i],6,50,50)
        new_data.append(img)
        new_labels.append(label)
    return np.asarray(new_data), np.asarray(new_labels)

def apply_perspective_transformation(data, label):
    new_data = []
    new_labels = []
    pts1 = np.float32([[4,4],[28,4],[4,28],[28,28]])
    pts2 = np.float32([[0,0],[32,0],[0,32],[32,32]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    for i in range(data.shape[0]):
        img = cv2.warpPerspective(data[i],M,(32,32))
        new_data.append(img)
        new_labels.append(label)
    return np.asarray(new_data), np.asarray(new_labels)

def apply_rotation(angle, data, label):
    new_data = []
    new_labels = []
    image_shape = data[0].shape[1::-1]
    image_center = tuple(np.array(image_shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    for i in range(data.shape[0]):
        img = cv2.warpAffine(data[i], rot_mat, data[i].shape[1::-1], flags=cv2.INTER_LINEAR)
        new_data.append(img)
        new_labels.append(label)
    return np.asarray(new_data), np.asarray(new_labels)

def apply_translation(data, label):
    new_data = []
    new_labels = []
    M = np.float32([[1,0,3],[0,1,3]])
    for i in range(data.shape[0]):
        img = cv2.warpAffine(data[i],M,(32,32))
        new_data.append(img)
        new_labels.append(label)
    return np.asarray(new_data), np.asarray(new_labels)

def apply_gaussian_filtering(data, label):
    new_data = []
    new_labels = []
    for i in range(data.shape[0]):
        img = cv2.GaussianBlur(data[i],(3,3),0)
        new_data.append(img)
        new_labels.append(label)
    return np.asarray(new_data), np.asarray(new_labels)

def augment_dataset(n_classes, X_train, y_train):
    X_augmented = X_train
    y_augmented = y_train
    
    for i in range(n_classes):
        class_patterns = sum(y_train==i)
        if (class_patterns<=250):
            k = 500 - 2*class_patterns
    
            new_data, new_labels = apply_bilateral_filtering(X_augmented[y_augmented==i], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
            
            data_for_gaussian = X_train[y_train==i][:k]
            new_data, new_labels = apply_gaussian_filtering(data_for_gaussian, i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
            
            new_data, new_labels = apply_translation(X_augmented[y_augmented==i], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
            
            new_data, new_labels = apply_perspective_transformation(X_augmented[y_augmented==i], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
        
            new_data, new_labels = apply_rotation(10, X_augmented[y_augmented==i], i)
            new_data2, new_labels2 = apply_rotation(-10, X_augmented[y_augmented==i], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
            X_augmented = np.append(X_augmented, new_data2, axis=0)
            y_augmented = np.append(y_augmented, new_labels2, axis=0)
            
        elif (class_patterns>=500):
            # Calculate the size of the sample that we need to take from the class
            k = int(1 + np.round((6000 - class_patterns)/12))
            # Create k new image by applying Bilateral filtering
            augmented_data, augmented_labels = apply_bilateral_filtering(X_augmented[y_augmented==i][:k], i)
            # Create another k images by applying perspective transformation (zoom) on the first k images
            new_data, new_labels = apply_perspective_transformation(augmented_data, i)
            augmented_data = np.append(augmented_data, new_data, axis=0)
            augmented_labels = np.append(augmented_labels, new_labels, axis=0)
            # Create another 2k images by applying translation to existing 2k augmented images
            new_data, new_labels = apply_translation(augmented_data, i)
            augmented_data = np.append(augmented_data, new_data, axis=0)
            augmented_labels = np.append(augmented_labels, new_labels, axis=0)
            # Create another 4k + 4k images by applying rotation to existing 4k augmented images
            new_data, new_labels = apply_rotation(10, augmented_data, i)
            new_data2, new_labels2 = apply_rotation(-10, augmented_data, i)
            augmented_data = np.append(augmented_data, new_data, axis=0)
            augmented_labels = np.append(augmented_labels, new_labels, axis=0)
            augmented_data = np.append(augmented_data, new_data2, axis=0)
            augmented_labels = np.append(augmented_labels, new_labels2, axis=0)
            # Append augmented data to the dataset
            X_augmented = np.append(X_augmented, augmented_data, axis=0)
            y_augmented = np.append(y_augmented, augmented_labels, axis=0)
            augmented_data = []
            augmented_labels = []
        else:
            # Calculate the size of the sample that we need to take from the class to apply translation
            k = 6000 - 12*class_patterns
            # Create new images by applying Bilateral filtering on the data of the whole class
            new_data, new_labels = apply_bilateral_filtering(X_augmented[y_augmented==i], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
            # Create new images by applying perspective transformation on the new and old images of the class
            new_data, new_labels = apply_perspective_transformation(X_augmented[y_augmented==i], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
            # Create new images by applying rotation on the new and old images of the class
            new_data, new_labels = apply_rotation(10, X_augmented[y_augmented==i], i)
            new_data2, new_labels2 = apply_rotation(-10, X_augmented[y_augmented==i], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
            X_augmented = np.append(X_augmented, new_data2, axis=0)
            y_augmented = np.append(y_augmented, new_labels2, axis=0)
            # Fill the free spots with k images created with translation.
            new_data, new_labels = apply_translation(X_augmented[y_augmented==i][:k], i)
            X_augmented = np.append(X_augmented, new_data, axis=0)
            y_augmented = np.append(y_augmented, new_labels, axis=0)
    
    histogram_plot(y_augmented, "Training examples after Preprocessing")
    return X_augmented, y_augmented

signnames = read_csv("signnames.csv").values[:, 1]
train_dataset_file = "traffic-signs-data/train.p"
X_train, y_train = load_pickled_data(train_dataset_file, ['features', 'labels'])
n_train = y_train.shape[0]
image_shape = X_train[0].shape
image_size = image_shape[0]
sign_classes, class_indices, class_counts = np.unique(y_train, return_index = True, return_counts = True)
n_classes = class_counts.shape[0]
print("Number of training examples =", n_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
histogram_plot(y_train, "Training examples before preprocessing")

X_train, y_train = augment_dataset(n_classes, X_train, y_train)