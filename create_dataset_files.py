import os
import pathlib
import random

GENRES = []
#creating Genres List
save_orgin_home = '/home/ilayy/HD-25/docker_test/PycharmProjects/pythonProject1'
path_dir = os.path.join(save_orgin_home,'Data/genres_original')
path_location = pathlib.Path(path_dir)
#choose size of test/train/validation
split_rate = (('Train', 0.6),('Validation', 0.2),('Test', 0.2))
print(split_rate)
for item in path_location.iterdir():
    GENRES.append(str(item).split('/')[-1])
#print(GENRES)
all_songs = {}
for item in GENRES:
    curr_dir = os.path.join(path_dir,f'{item}')
    curr_dir_location = pathlib.Path(curr_dir)
    for song in curr_dir_location.iterdir():
        #print(song)
        song_name = str(song).split('/')[-1]
        song_number = song_name.split('_')[0].split('.')[-1]
        if song_number in  all_songs:
            all_songs[song_number] = all_songs[song_number] + [item + '/' + song_name]
        else:
            all_songs[song_number] = [item + '/' + song_name]
        #all_songs.append(f'{item}{tmp}{song_name}')
#print(all_songs)
#random.shuffle(all_songs)

all_songs_list = list(all_songs.items())

# Shuffle the list
random.shuffle(all_songs_list)

# Convert the shuffled list back to a dictionary
shuffled_all_songs_list= dict(all_songs_list)

def dict_placment(my_dict , start, stop):
    out_dict = {}
    for i, song in enumerate(my_dict):
        if i > start and i < stop:
            out_dict[song] = my_dict[song]
    return out_dict

split_point = (int(split_rate[0][1] * len(all_songs)),int(split_rate[0][1] * len(all_songs))+int(split_rate[1][1] * len(all_songs)))
Train_Song_List = dict_placment(shuffled_all_songs_list, 0,split_point[0])
Valid_Song_List = dict_placment(shuffled_all_songs_list, split_point[0] + 1 ,split_point[1])
Test_Song_List = dict_placment(shuffled_all_songs_list, split_point[1] + 1 ,len(shuffled_all_songs_list))

#fp= open(r'test_filtered1.txt', 'w')
with open(r'train_filtered.txt', 'w') as fp:
    for item in Train_Song_List:
        for song in Train_Song_List[item]:
        # write each item on a new line
            fp.write("%s\n" % song)
with open(r'test_filtered.txt', 'w') as fp:
    for item in Test_Song_List:
        for song in Test_Song_List[item]:
        # write each item on a new line
            fp.write("%s\n" % song)
with open(r'valid_filtered.txt', 'w') as fp:
    for item in Valid_Song_List:
        # write each item on a new line
        for song in Valid_Song_List[item]:
            # write each item on a new line
            fp.write("%s\n" % song)