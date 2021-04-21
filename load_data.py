import numpy as np
import os
from tqdm.notebook import tqdm

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Sound Processing
import librosa
from Signal_Analysis.features.signal import get_F_0, get_HNR

# Training Data Preparation
from sklearn.model_selection import train_test_split

NUM_AUDIO_PER_ACTOR = 60
NUM_ACTORS = 24
RNN_FEATS = 150
DENSE_FEATS = 43
MAX_TIMESTEP = 224
NUM_EMOTIONS = 8
N_FFT = 2048
HOP_LENGTH = 512
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

def extract_HSF(lld):
    mean_val = lld.mean()
    min_val = lld.min()
    max_val = lld.max()
    var_val = lld.var()
    range_val = np.subtract(max_val, min_val)
    q25_val = np.quantile(lld, 0.25)
    q50_val = np.quantile(lld, 0.5)
    q75_val = np.quantile(lld, 0.75)
    return np.asarray([
        mean_val,
        min_val,
        max_val,
        var_val,
        range_val,
        q25_val,
        q50_val,
        q75_val,
    ])

def extract_LLD_from_subaudio(subaudio, fs):
    # Frame-wise energy
    energy_val = np.sum(np.square(subaudio)) / (subaudio.shape[0] / fs + 0.00000000000001)
    
    # Frame-wise Zero Crossing Rate
    zcr_val = np.sum((subaudio[:-1] * subaudio[1:]) < 0)
    
    return np.asarray([
        energy_val,
        zcr_val,
    ])

def extract_LLD_from_audio(audio, fs):
    # MFCC
    mfcc = librosa.feature.mfcc(audio, fs, n_fft = N_FFT, hop_length = HOP_LENGTH, center = False).transpose()
    mfcc_hsf = extract_HSF(mfcc)
    
    # LPC
    lpc = librosa.lpc(audio, 16)
    
    # Mel-Spectrogram
    spect = librosa.feature.melspectrogram(y = audio, sr = fs, n_fft = N_FFT, hop_length = HOP_LENGTH, center = False)
    spect = librosa.power_to_db(spect, ref = np.max).transpose()
    spect_hsf = extract_HSF(spect)
    
    # Other features
    f0 = get_F_0(audio, fs)[0]
    hnr = get_HNR(audio, fs)
    
    return np.asarray(mfcc), np.asarray(mfcc_hsf), np.asarray(lpc), np.asarray(spect), np.asarray(spect_hsf), np.asarray([f0, hnr])

def extract_LLD(audio, fs):
    num_windows = int((audio.shape[0] - N_FFT) // HOP_LENGTH) + 1
    framewise_lld = np.zeros((num_windows, 2))
    for idx in range(num_windows):
       subaudio = audio[int(idx * HOP_LENGTH): int(idx * HOP_LENGTH + N_FFT)]
       framewise_lld[idx, :] = extract_LLD_from_subaudio(subaudio, fs)
    framewise_lld_hsf = extract_HSF(framewise_lld)
    
    mfcc, mfcc_hsf, lpc, spect, spect_hsf, others = extract_LLD_from_audio(audio, fs)
    
    assert(framewise_lld.shape[0] == mfcc.shape[0])
    assert(mfcc.shape[0] == spect.shape[0])

    rnn_feats = np.concatenate((framewise_lld, mfcc, spect), axis = 1)
    dense_feats = np.concatenate((framewise_lld_hsf, mfcc_hsf, lpc, spect_hsf, others))
    return rnn_feats, dense_feats

# # For creating data_X_rnn, data_X_dense & data_y
# data_X_rnn = np.zeros((NUM_AUDIO_PER_ACTOR * NUM_ACTORS, MAX_TIMESTEP, RNN_FEATS), dtype = np.float64)
# data_X_dense = np.zeros((NUM_AUDIO_PER_ACTOR * NUM_ACTORS, DENSE_FEATS), dtype = np.float64)
# data_y = np.zeros((NUM_AUDIO_PER_ACTOR * NUM_ACTORS, NUM_EMOTIONS), dtype = np.uint8)
# actors_list = os.listdir('./Datasets/RAVDESS/')
# idx = 0
# for actor in tqdm(actors_list, total = NUM_ACTORS):
#     audio_list = os.listdir(f'./Datasets/RAVDESS/{actor}/')
#     emotion = [int(x.split('-')[2]) - 1 for x in audio_list]
#     for audio in audio_list:
#         audio_filename = f'./Datasets/RAVDESS/{actor}/{audio}'
#         y, sr = librosa.load(audio_filename)
#         rnn_feats, dense_feats = extract_LLD(y, sr)
#         data_X_rnn[idx, :rnn_feats.shape[0], :] = rnn_feats
#         data_X_dense[idx, :] = dense_feats
#         data_y[idx, :] = np.identity(NUM_EMOTIONS)[emotion[idx % NUM_AUDIO_PER_ACTOR]]
#         idx += 1

# with open('./Processed_Data/data_X_rnn.npy', 'wb') as save_file:
#     np.save(save_file, data_X_rnn)
# with open('./Processed_Data/data_X_dense.npy', 'wb') as save_file:
#     np.save(save_file, data_X_dense)
# with open('./Processed_Data/data_y.npy', 'wb') as save_file:
#     np.save(save_file, data_y)

def get_train_val_split(X_rnn_filepath, X_dense_filepath, y_filepath, train_perc = 75, val_perc = 10):
    
    with open(X_rnn_filepath, 'rb') as load_file:
        data_X_rnn = np.load(load_file)
    with open(X_dense_filepath, 'rb') as load_file:
        data_X_dense = np.load(load_file)
    with open(y_filepath, 'rb') as load_file:
        data_y = np.load(load_file)
    
    def standardize_rnn_feats(X):
        for col_idx in range(X.shape[2]):
            X[:, :, col_idx] = (X[:, :, col_idx] - X[:, :, col_idx].mean()) / (X[:, :, col_idx].std() + 0.00000000001)
        return X
    data_X_rnn = standardize_rnn_feats(data_X_rnn)
    
    def standardize_dense_feats(X):
        for col_idx in range(X.shape[1]):
            X[:, col_idx] = (X[:, col_idx] - X[:, col_idx].mean()) / (X[:, col_idx].std() + 0.00000000001)
        return X
    data_X_dense = standardize_dense_feats(data_X_dense)

    def zip_rnn_dense(X_rnn, X_dense):
        return list(zip(X_rnn, X_dense))
    
    def unzip_rnn_dense(X):
        X_rnn = np.asarray([x[0] for x in X])
        X_dense = np.asarray([x[1] for x in X])
        return X_rnn, X_dense
    
    data_X = zip_rnn_dense(data_X_rnn, data_X_dense)

    test_perc = 100 - (train_perc + val_perc)
    train_val_X, test_X, train_val_y, test_y = train_test_split(data_X, data_y, test_size = test_perc / 100, stratify = np.argmax(data_y, axis = 1), random_state = 42)
    train_X, val_X, train_y, val_y = train_test_split(train_val_X, train_val_y, test_size = val_perc / (train_perc + val_perc), stratify = np.argmax(train_val_y, axis = 1), random_state = 42)
    
    train_X_rnn, train_X_dense = unzip_rnn_dense(train_X)
    val_X_rnn, val_X_dense = unzip_rnn_dense(val_X)
    test_X_rnn, test_X_dense = unzip_rnn_dense(test_X)

    return train_X_rnn, train_X_dense, train_y, val_X_rnn, val_X_dense, val_y, test_X_rnn, test_X_dense, test_y

# # Sample Usage
# train_X_rnn, train_X_dense, train_y, val_X_rnn, val_X_dense, val_y, test_X_rnn, test_X_dense, test_y = get_train_val_split(X_rnn_filepath = './Processed_Data/data_X_rnn.npy', X_dense_filepath = './Processed_Data/data_X_dense.npy', y_filepath = './Processed_Data/data_y.npy')

# print(f"Shape of Train_X_rnn: {train_X_rnn.shape}")
# print(f"Shape of Train_X_dense: {train_X_dense.shape}")
# print(f"Shape of Train_y: {train_y.shape}")
# print()
# print(f"Shape of Val_X_rnn: {val_X_rnn.shape}")
# print(f"Shape of Val_X_dense: {val_X_dense.shape}")
# print(f"Shape of Val_y: {val_y.shape}")
# print()
# print(f"Shape of Test_X_rnn: {test_X_rnn.shape}")
# print(f"Shape of Test_X_dense: {test_X_dense.shape}")
# print(f"Shape of Test_y: {test_y.shape}")