import os
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import myextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

def class_vector():
    Tk().withdraw()
    pore_folder = askdirectory(title="Select the pore image folder")

    Tk().withdraw()
    non_pore_folder = askdirectory(title="Select the non-pore image folder")

    pore_images = []
    pore_labels = []
    for filename in os.listdir(pore_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(pore_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = myextract.extract_features(image)
            pore_images.append(features)
            pore_labels.append(1)

    non_pore_images = []
    non_pore_labels = []
    for filename in os.listdir(non_pore_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(non_pore_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = myextract.extract_features(image)
            non_pore_images.append(features)
            non_pore_labels.append(0)

    feature_vectors = np.concatenate((pore_images, non_pore_images), axis=0)
    class_labels = np.concatenate((pore_labels, non_pore_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, class_labels, test_size=0.2, random_state=42)

    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    import matplotlib.pyplot as plt

    # Assuming you have feature_vectors as a NumPy array with shape (num_samples, 6)
    # and class_labels as a NumPy array with shape (num_samples,)
    X = feature_vectors[:, :2]  # Selecting the first two features for the first diagram
    y = class_labels

    # Create a scatter plot with two different colors for each class
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Non-Pore', marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Pore', marker='x')

    # Add labels and legend
    plt.xlabel('Minimum Intensity')
    plt.ylabel('Maximum Intensity')
    plt.legend()
    plt.title('SVM: Minimum Intensity vs Maximum Intensity')
    plt.show()

    # Repeat the process for the next two features
    X = feature_vectors[:, 2:4]  # Selecting the next two features for the second diagram

    # Create a scatter plot with two different colors for each class
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Non-Pore', marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Pore', marker='x')

    # Add labels and legend
    plt.xlabel('Mean Intensity')
    plt.ylabel('Average Intensity')
    plt.legend()
    plt.title('SVM: Mean Intensity vs Average Intensity')
    plt.show()

    # Repeat the process for the last two features
    X = feature_vectors[:, 4:]  # Selecting the last two features for the third diagram

    # Create a scatter plot with two different colors for each class
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Non-Pore', marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Pore', marker='x')

    # Add labels and legend
    plt.xlabel('Standard Deviation')
    plt.ylabel('Spectral Entropy')
    plt.legend()
    plt.title('SVM: Standard Deviation vs Spectral Entropy')
    plt.show()

def class_random():
    Tk().withdraw()
    pore_folder = askdirectory(title="Select the pore image folder")

    Tk().withdraw()
    non_pore_folder = askdirectory(title="Select the non-pore image folder")

    pore_images = []
    pore_labels = []
    for filename in os.listdir(pore_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(pore_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = myextract.extract_features(image)
            pore_images.append(features)
            pore_labels.append(1)

    non_pore_images = []
    non_pore_labels = []
    for filename in os.listdir(non_pore_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(non_pore_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = myextract.extract_features(image)
            non_pore_images.append(features)
            non_pore_labels.append(0)

    feature_vectors = np.concatenate((pore_images, non_pore_images), axis=0)
    class_labels = np.concatenate((pore_labels, non_pore_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, class_labels, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def class_decision():
    Tk().withdraw()
    pore_folder = askdirectory(title="Select the pore image folder")

    Tk().withdraw()
    non_pore_folder = askdirectory(title="Select the non-pore image folder")

    pore_images = []
    pore_labels = []
    for filename in os.listdir(pore_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(pore_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = myextract.extract_features(image)
            pore_images.append(features)
            pore_labels.append(1)

    non_pore_images = []
    non_pore_labels = []
    for filename in os.listdir(non_pore_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(non_pore_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            features = myextract.extract_features(image)
            non_pore_images.append(features)
            non_pore_labels.append(0)

    feature_vectors = np.concatenate((pore_images, non_pore_images), axis=0)
    class_labels = np.concatenate((pore_labels, non_pore_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, class_labels, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def class_cnn():
    Tk().withdraw()
    pore_folder = askdirectory(title="Select the pore image folder")

    Tk().withdraw()
    non_pore_folder = askdirectory(title="Select the non-pore image folder")

    pore_images = []
    pore_labels = []

    for filename in os.listdir(pore_folder):
        img = cv2.imread(os.path.join(pore_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        pore_images.append(img)
        pore_labels.append(1)

    nonpore_images = []
    nonpore_labels = []

    for filename in os.listdir(non_pore_folder):
        img = cv2.imread(os.path.join(non_pore_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        nonpore_images.append(img)
        nonpore_labels.append(0)

    images = pore_images + nonpore_images
    labels = pore_labels + nonpore_labels

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    plot_model(model, to_file='cnn_architecture.png', show_shapes=True)
