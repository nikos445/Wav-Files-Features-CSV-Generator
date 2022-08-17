import librosa
import librosa.display  
import numpy as np
import os, csv

from matplotlib import pyplot as plt
 
features_no = 1000

folder = os.listdir('./input/')

files_and_features = []

for index, filename  in enumerate(folder):
    
    audio, sample_rate = librosa.load('./input/'+filename,res_type='kaiser_fast') 
    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=features_no)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    print(mfccs_scaled_features)

    files_and_features.append([index, mfccs_scaled_features, filename])

with open('./output/features.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(files_and_features)