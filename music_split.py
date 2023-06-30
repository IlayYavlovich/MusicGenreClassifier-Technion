import pathlib
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pydub import AudioSegment
from tqdm import tqdm

# create a copy of each song in each list in the new format
path_dir = os.path.join(os.getcwd(), 'changing_format_channels_input')
path_location = pathlib.Path(path_dir)
NAMES = ['pop','classical','electronic'] #Genres to exclude
for file in tqdm(path_location.iterdir()):
    new_file_name = ((str(file).split('/')[-1]).split('_')[-1]).split('.')[0]
    if not new_file_name in NAMES:
        continue
    path_dir = os.path.join('/home/ilayy/Desktop/10_genre_dataset', new_file_name)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    file_1 = open(file, 'rt')
    Lines = file_1.readlines()
    for song in Lines:
        song = song.split('\n')[0]
        path_all_songs = os.path.join('/home/ilayy/Desktop/', 'all_songs/{}.mp3'.format(song))
        shutil.copy(path_all_songs, path_dir)
        original_name_dir = path_dir + '/{}.mp3'.format(song)
        new_name_dir = path_dir + '/' + new_file_name + '.{}.wav'.format(song)
        os.rename(original_name_dir, new_name_dir)

#Slice each song to time_split intervals
time_split = 15
NAMES = ['pop','classical','electronic'] #Genres to exclude
old_dataset = '/home/ilayy/Desktop/10_genre_dataset'
#os.mkdir('/home/ilayy/PycharmProjects/pythonProject/new_dateset_' + str(time_split))
output_path = '/home/ilayy/Desktop/10_genre_split/' + str(time_split) + '_sec'
if not os.path.exists(output_path):
    os.mkdir(output_path)
for genre in tqdm(os.listdir(old_dataset)):
    if not genre in NAMES:
        continue
    genre_file_name = old_dataset + '/' + genre
    output_genre_file_name = output_path + '/' + genre
    if not os.path.exists(output_genre_file_name):
        os.mkdir(output_genre_file_name)
        for filename in os.listdir(genre_file_name):
            save_file_name = filename[:-4]
            myaudio = AudioSegment.from_file(genre_file_name + '/' + filename)
            #print(myaudio.duration_seconds)
            for chunk_number in range(int(np.floor(myaudio.duration_seconds / float(time_split)))):
                save_file_name_chunk = save_file_name + '_' + str(chunk_number) + '.wav'
                #print(save_file_name_chunk)
                start_time = chunk_number * time_split * 1000
                stop_time = (chunk_number + 1) * time_split * 1000
                #print(start_time, stop_time)
                audio_seg = myaudio[start_time:stop_time]
                #print(audio_seg.duration_seconds)
                f = audio_seg.export(output_genre_file_name + '/' + save_file_name_chunk , format="wav")
                f.close()
