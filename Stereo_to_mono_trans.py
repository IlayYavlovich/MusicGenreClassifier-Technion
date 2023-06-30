import torch
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from tqdm import tqdm
from os import listdir
from pydub import AudioSegment
import os
#Convert Stereo music files to mono music files
time_split = 15
input_path = '/home/ilayy/Desktop/10_genre_split/' + str(time_split) + '_sec'
for genre in tqdm(os.listdir(input_path)):
    song_path = input_path + '/' + genre
    for song in os.listdir(song_path):
        a = read(song_path + '/' + song)
        b = np.array(a[1], dtype=float)

        if b.shape[0] == b.size:
            continue
        else:
            sound = AudioSegment.from_wav(song_path + '/' + song)
            sound = sound.set_channels(1)
            sound.export(song_path + '/' + song, format="wav")

