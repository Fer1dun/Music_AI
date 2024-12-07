
import streamlit as st
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import seaborn as sns
import librosa.display
import librosa
from collections import defaultdict
from tempfile import TemporaryFile
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import os
import math
import pickle
import random
import operator
import base64
import requests


import math
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Additional libraries for data preprocessing and visualization
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV


st.set_page_config(page_title='MusicMapping', layout="wide")
st.title('Music Genere Classification')

st.sidebar.image('https://editor.analyticsvidhya.com/uploads/94476Music%20Genre%20Classification%20Project.png')
st.sidebar.write('Developed by ')
st.sidebar.write('V K Deeksha - 21PD37')
st.sidebar.write('M Aiswarya - 21PD20')


# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview and Data","📈 Visualize","KNN-Model","RandomForest_Model","Prediction"])

Root = "/content/drive/MyDrive/Data"
os.chdir(Root)
audio_data_path = "/content/drive/MyDrive/Data/genres_original"

with tab1:
    st.header("Overview")
    st.markdown("Certainly! A music genre classification project involves building a system that can automatically categorize music tracks into predefined genres based on their audio features. This type of project is often approached using machine learning techniques, where the model learns patterns and characteristics of different genres from a labeled dataset.")
    st.markdown('---')
    st.markdown('GTZAN MUSIC dataset is a vast collection of audio music files.')
    music_data = pd.read_csv("/content/drive/MyDrive/Data/features_30_sec.csv")
    st.write("GTZAN Dataset:")
    st.write(music_data)


with tab2:
    st.header("Visualization")

    st.subheader("Waveform of every Genere")
    from importlib import reload
    plt=reload(plt)
    base_path = '/content/drive/MyDrive/Data/genres_original/'
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    plt.figure(figsize=(18, 10))
    for i, genre in enumerate(genres):
        # Get the list of audio files in the genre folder
        genre_folder = os.path.join(base_path, genre)
        audio_files = os.listdir(genre_folder)

        # Plot the first audio file in the genre
        path = os.path.join(genre_folder, audio_files[0])

        plt.subplot(5, 2, i + 1)
        x, sr = librosa.load(path)
        librosa.display.waveshow(x, sr=sr)

        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Waveform of the Genre {genre}')

    plt.tight_layout()
    st.pyplot(plt)

    st.subheader("Heatmap")
    spike_cols = [col for col in music_data.columns if 'mean' in col]
    fig, ax = plt.subplots(figsize=(16, 11))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(music_data[spike_cols].corr(), cmap='YlGn', ax=ax)
    plt.title('Heatmap for MEAN variables', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(fig)

    st.subheader("Genre Spectrogram Viewer")
    genres_path = '/content/drive/MyDrive/Data/genres_original/'
    genres = os.listdir(genres_path)
    selected_genre = st.selectbox('Select Genre', genres)
    audio_path = os.path.join(genres_path, selected_genre, f'{selected_genre}.00000.wav')
    data, sr = librosa.load(audio_path)
    stft = librosa.stft(data)
    stft_db = librosa.amplitude_to_db(abs(stft))
    # Display the spectrogram using Matplotlib
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
    plt.ylim(0, 200)
    plt.title(f'Spectrogram for Genre {selected_genre.capitalize()}')
    plt.colorbar()
    st.pyplot(plt)

    st.subheader("MFCC Visualization")
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    # Display the MFCCs using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, format='%+2.0f dB')
    # Display the plot using Streamlit
    st.pyplot(fig)

with tab3:
    st.header("KNN Model and Lime Explanations")
    def getNeighbors(trainingSet, instance, k):
        distances = []
        for x in range (len(trainingSet)):
            dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
            distances.append((trainingSet[x][2], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def nearestClass(neighbors):
        classVote = {}

        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVote:
                classVote[response]+=1
            else:
                classVote[response]=1

        sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
        return sorter[0][0]

    def getAccuracy(testSet, predictions):
        correct = 0
        for x in range (len(testSet)):
            if testSet[x][-1]==predictions[x]:
                correct+=1
        return 1.0*correct/len(testSet)

    directory = audio_data_path
    f = open("mydataset.dat", "wb")
    i = 0
    for folder in os.listdir(directory):
        #print(folder)
        i += 1
        if i == 11:
            break
        for file in os.listdir(directory+"/"+folder):
            #print(file)
            try:
                (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
                mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, f)
            except Exception as e:
                print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
    f.close()

    dataset = []

    def loadDataset(filename, split, trset, teset):
        with open('mydataset.dat','rb') as f:
            while True:
                try:
                    dataset.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break
        for x in range(len(dataset)):
            if random.random() < split:
                trset.append(dataset[x])
            else:
                teset.append(dataset[x])

    trainingSet = []
    testSet = []
    loadDataset(music_data, 0.68, trainingSet, testSet)

    def distance(instance1, instance2, k):
        distance = 0
        mm1 = instance1[0]
        cm1 = instance1[1]
        mm2 = instance2[0]
        cm2 = instance2[1]
        distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
        distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
        distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
        distance -= k
        return distance

    # Make the prediction using KNN(K nearest Neighbors)
    length = len(testSet)
    predictions = []
    for x in range(length):
        predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))
    accuracy = getAccuracy(testSet, predictions)
    st.write(f"Accuracy of the KNN Model: {accuracy:.2%}")

    # Lime Explanations
    st.subheader("Lime Explanations")
    data = pd.read_csv('/content/drive/MyDrive/Data/features_30_sec.csv')
    data = data.iloc[0:, 1:]
    y = data['label']
    X = data.loc[:, data.columns != 'label']
    cols = X.columns

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    st.write("Lime Explanation:")
    knn_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=y.unique())
    knn_exp = knn_explainer.explain_instance(X_test.iloc[0].values, knn_model.predict_proba)
    st.write(knn_exp)

with tab4:
    st.header("RandomForest Model")
    data = pd.read_csv('/content/drive/MyDrive/Data/features_30_sec.csv')
    data = data.iloc[0:, 1:]
    y = data['label']
    X = data.loc[:, data.columns != 'label']
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    scaled_df = pd.DataFrame(np_scaled, columns = cols)
    st.write("Scaled Data:")
    st.write(scaled_df.head())
    X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    st.write("Classification Report:")
    st.write(class_report)
    st.subheader("After hyperparameter Tuning:")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_rf_model = RandomForestClassifier(**best_params, random_state=42)
    best_rf_model.fit(X_train, y_train)
    st.write("Best Random Forest Model Parameters:")
    st.write(best_params)
    if hasattr(best_rf_model, 'feature_importances_'):
        feature_importances = dict(zip(X_train.columns, best_rf_model.feature_importances_))
        st.write("Feature Importances:")
        st.write(pd.Series(feature_importances).sort_values(ascending=False))
    train_predictions = best_rf_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    st.write(f"Training Accuracy After predicting using the best model:{train_accuracy}")
    test_predictions = best_rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    st.write(f"Test Accuracy After predicting using the best model:{test_accuracy}")
    # Create a LIME explainer
    # LIME for Random Forest
    rf_explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=y.unique())
    rf_exp = rf_explainer.explain_instance(X_test.iloc[0].values, best_rf_model.predict_proba)
    st.write("Lime Explanation:")
    st.write(rf_exp)

with tab5:
    st.header('Audio Genre Prediction')
    def getNeighbors(trainingSet, instance, k):
        distances = []
        for x in range (len(trainingSet)):
            dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
            distances.append((trainingSet[x][2], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def nearestClass(neighbors):
        classVote = {}

        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVote:
                classVote[response]+=1
            else:
                classVote[response]=1

        sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
        return sorter[0][0]

    def getAccuracy(testSet, predictions):
        correct = 0
        for x in range (len(testSet)):
            if testSet[x][-1]==predictions[x]:
                correct+=1
        return 1.0*correct/len(testSet)

    directory = audio_data_path
    f = open("mydataset.dat", "wb")
    i = 0
    for folder in os.listdir(directory):
        #print(folder)
        i += 1
        if i == 11:
            break
        for file in os.listdir(directory+"/"+folder):
            #print(file)
            try:
                (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
                mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, f)
            except Exception as e:
                print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
    f.close()

    dataset = []

    def loadDataset(filename, split, trset, teset):
        with open('mydataset.dat','rb') as f:
            while True:
                try:
                    dataset.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break
        for x in range(len(dataset)):
            if random.random() < split:
                trset.append(dataset[x])
            else:
                teset.append(dataset[x])

    trainingSet = []
    testSet = []
    loadDataset(music_data, 0.68, trainingSet, testSet)

    def distance(instance1, instance2, k):
        distance = 0
        mm1 = instance1[0]
        cm1 = instance1[1]
        mm2 = instance2[0]
        cm2 = instance2[1]
        distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
        distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
        distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
        distance -= k
        return distance

    # Make the prediction using KNN(K nearest Neighbors)
    length = len(testSet)
    predictions = []
    for x in range(length):
        predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

    def predict_genre(audio_file):
      results = defaultdict(int)
      directory = audio_data_path
      i = 1
      for folder in os.listdir(directory):
          results[i] = folder
          i += 1
      pred = nearestClass(getNeighbors(dataset, feature, 5))
      return (results[pred])

    uploaded_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav'])
    if uploaded_file:
        st.audio(uploaded_file)
        prediction = predict_genre(uploaded_file)
        st.write(f"Predicted Genre: {prediction}")
