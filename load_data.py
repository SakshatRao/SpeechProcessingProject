import numpy as np
import os
from tqdm.notebook import tqdm

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Sound Processing
import librosa

# Training Data Preparation
from sklearn.model_selection import train_test_split

NUM_AUDIO_PER_ACTOR = 60
NUM_ACTORS = 24
FREQ_RANGE = 128
MAX_TIMESTEP = 228
NUM_EMOTIONS = 8
N_FFT = 2048
HOP_LENGTH = 512
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# # For creating data_X & data_y numpy arrays
# data_X = np.zeros((NUM_AUDIO_PER_ACTOR * NUM_ACTORS, MAX_TIMESTEP, FREQ_RANGE), dtype = np.float64)
# data_y = np.zeros((NUM_AUDIO_PER_ACTOR * NUM_ACTORS, NUM_EMOTIONS), dtype = np.uint8)
# actors_list = os.listdir('./Datasets/RAVDESS/')
# idx = 0
# for actor in tqdm(actors_list, total = NUM_ACTORS):
#     audio_list = os.listdir(f'./Datasets/RAVDESS/{actor}/')
#     emotion = [int(x.split('-')[2]) - 1 for x in audio_list]
#     for audio in audio_list:
#         audio_filename = f'./Datasets/RAVDESS/{actor}/{audio}'
#         y, sr = librosa.load(audio_filename)
#         spect = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = N_FFT, hop_length = HOP_LENGTH)
#         spect = librosa.power_to_db(spect, ref = np.max)
#         data_X[idx, :spect.shape[1], :] = spect.transpose()
#         data_y[idx, :] = np.identity(NUM_EMOTIONS)[emotion[idx % NUM_AUDIO_PER_ACTOR]]
#         idx += 1
# with open('./Processed_Data/data_X.npy', 'wb') as save_file:
#     np.save(save_file, data_X)
# with open('./Processed_Data/data_y.npy', 'wb') as save_file:
#     np.save(save_file, data_y)

def get_train_val_split(X_filepath, y_filepath, train_perc = 75, val_perc = 10):
    
    with open(X_filepath, 'rb') as load_file:
        data_X = np.load(load_file)
    with open(y_filepath, 'rb') as load_file:
        data_y = np.load(load_file)
    
    def standardize(X):
        return (X - X.mean()) / (X.std() + 0.000001)
    data_X = standardize(data_X)
    
    test_perc = 100 - (train_perc + val_perc)
    train_val_X, test_X, train_val_y, test_y = train_test_split(data_X, data_y, test_size = test_perc / 100, stratify = np.argmax(data_y, axis = 1), random_state = 42)
    train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size = val_perc / (train_perc + val_perc), stratify = np.argmax(train_val_y, axis = 1), random_state = 42)
    return train_X, train_y, val_X, val_y, test_X, test_y

# # Sample Usage
# train_X, train_y, val_X, val_y, test_X, test_y = get_train_val_split(X_filepath = './Processed_Data/data_X.npy', y_filepath = './Processed_Data/data_y.npy')
# print(f"Shape of Train_X: {train_X.shape}")
# print(f"Shape of Train_X: {train_y.shape}")
# print(f"Shape of Train_X: {val_X.shape}")
# print(f"Shape of Train_X: {val_y.shape}")
# print(f"Shape of Train_X: {test_X.shape}")
# print(f"Shape of Train_X: {test_y.shape}")