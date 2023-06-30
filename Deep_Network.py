import os
import random
import torch
import numpy as np
import torchviz

import soundfile as sf
import torchaudio
from torch.utils import data
from tqdm import tqdm
from torch import nn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)

GTZAN_GENRES = ['electronic', 'classical', 'pop', 'ambient', 'hiphop', 'jazz', 'metal', 'reggae', 'rock', 'soundtrack']
#GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
Loaded_Best = 0
Weight_dataset = 0

class JamendoDataset(data.Dataset):
    def __init__(self, data_path, split, num_samples, num_chunks, is_augmentation):
        self.data_path =  data_path if data_path else ''
        self.split = split
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation
        self.genres = GTZAN_GENRES
        self._get_song_list()
        if is_augmentation:
            self._get_augmentations()

    def _get_song_list(self):
        list_filename = os.path.join(self.data_path, '%s_filtered.txt' % self.split)
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]

    def _get_augmentations(self):
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=22050)], p=0.8),
            RandomApply([Delay(sample_rate=22050)], p=0.5),
            RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=22050)], p=0.4),
            RandomApply([Reverb(sample_rate=22050)], p=0.3),
        ]
        self.augmentation = Compose(transforms=transforms)

    def _adjust_audio_length(self, wav):
        if self.split == 'train':
            random_index = random.randint(0, len(wav) - self.num_samples - 1)
            wav = wav[random_index : random_index + self.num_samples]
        else:
            hop = (len(wav) - self.num_samples) // self.num_chunks
            wav = np.array([wav[i * hop : i * hop + self.num_samples] for i in range(self.num_chunks)])
        return wav

    def __getitem__(self, index):
        line = self.song_list[index]

        # get genre
        genre_name = line.split('/')[0]
        genre_index = self.genres.index(genre_name)

        # get audio
        audio_filename = os.path.join(self.data_path, 'Data/genres_original', line)
        #print(audio_filename)
        wav, fs = sf.read(audio_filename)

        # adjust audio length
        wav = self._adjust_audio_length(wav).astype('float32')

        # data augmentation
        if self.is_augmentation:
            wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()

        return wav, genre_index

    def __len__(self):
        return len(self.song_list)
#num_samples=22050 * 29,
song_len = 30
def get_dataloader(data_path=None, 
                   split='train', 
                   num_samples=22050 * (song_len-1),
                   num_chunks=1, 
                   batch_size=16,
                   num_workers=0, 
                   is_augmentation=False):

    is_shuffle = True if (split == 'train') else False
    batch_size = batch_size if (split == 'train') else (batch_size // num_chunks)
    data_loader = data.DataLoader(dataset=JamendoDataset(data_path, 
                                                       split, 
                                                       num_samples, 
                                                       num_chunks, 
                                                       is_augmentation),
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader


train_loader = get_dataloader(split='train', is_augmentation=True)
iter_train_loader = iter(train_loader)
train_wav, train_genre = next(iter_train_loader)

valid_loader = get_dataloader(split='valid')
test_loader = get_dataloader(split='test')

iter_test_loader = iter(test_loader)
test_wav, test_genre = next(iter_test_loader)

print('training data shape: %s' % str(train_wav.shape))
print('validation/test data shape: %s' % str(test_wav.shape))
"""
###########
def find_weights(song_list , genre_list):
    weights = np.zeros(len(genre_list))
    for song in song_list:
        song_genre = song.split('/')[0]
        for i,genre in enumerate(genre_list):
            if song_genre == genre:
                weights[i] += 1
                break
    tot_sum = 0
    for i in range(0,len(weights)):
        tot_sum = tot_sum + weights[i]
    for i in range(0,len(weights)):
        weights[i] = weights[i]/tot_sum
    #weights = 1/weights
    return torch.from_numpy(weights)

train_size = train_loader.dataset.__len__()
valid_size = valid_loader.dataset.__len__()
test_size = test_loader.dataset.__len__()
weights = find_weights(train_loader.dataset.song_list , train_loader.dataset.genres)
weight_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights.type('torch.DoubleTensor'), len(weights))
"""
###########

#train_loader = get_dataloader(split='train', is_augmentation=True,add_sampler= weight_sampler if (Weight_dataset) else None)
#iter_train_loader = iter(train_loader)
#train_wav, train_genre = next(iter_train_loader)

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#num_classes=10
class CNN(nn.Module):
    def __init__(self, num_channels=16,
                 sample_rate=22050,
                 n_fft=1024,
                 f_min=0.0,
                 f_max=11025.0,
                 num_mels=128,
                 num_classes=10):
        super(CNN, self).__init__()

        # mel spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(1)

        # convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))
        self.layer2 = Conv_2d(num_channels, num_channels, pooling=(3, 4))
        self.layer3 = Conv_2d(num_channels, num_channels * 2, pooling=(2, 5))
        self.layer4 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(3, 3))
        self.layer5 = Conv_2d(num_channels * 2, num_channels * 4, pooling=(3, 4))

        # dense layers
        self.dense1 = nn.Linear(num_channels * 4, num_channels * 4)
        self.dense_bn = nn.BatchNorm1d(num_channels * 4)
        self.dense2 = nn.Linear(num_channels * 4, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, wav):
        # input Preprocessing
        out = self.melspec(wav)
        out = self.amplitude_to_db(out)

        # input batch normalization
        out = out.unsqueeze(1)
        out = self.input_bn(out)

        # convolutional layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = out.reshape(len(out), -1)

        # dense layers
        out = self.dense1(out)
        out = self.dense_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)

        return out



from sklearn.metrics import accuracy_score, confusion_matrix


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)
cnn = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
valid_losses = []
num_epochs = 30
if not Loaded_Best:
    for epoch in tqdm(range(num_epochs)):
        losses = []
        print (epoch, ' / ' ,num_epochs )
        # Train
        cnn.train()
        for (wav, genre_index) in train_loader:
            #print((wav, genre_index), ' / ', train_loader)
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # Forward
            out = cnn(wav)
            loss = loss_function(out, genre_index)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Epoch: [%d/%d], Train loss: %.4f' % (epoch+1, num_epochs, np.mean(losses)))

        # Validation
        cnn.eval()
        y_true = []
        y_pred = []
        losses = []
        for wav, genre_index in valid_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # reshape and aggregate chunk-level predictions
            b, c, t = wav.size()
            logits = cnn(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            loss = loss_function(logits, genre_index)
            losses.append(loss.item())
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())
        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(losses)
        print('Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f' % (epoch+1, num_epochs, valid_loss, accuracy))

        # Save model
        valid_losses.append(valid_loss.item())
        if np.argmin(valid_losses) == epoch:
            print('Saving the best model at %d epochs!' % epoch)
            torch.save(cnn.state_dict(), 'best_model.ckpt')

# Load the best model
S = torch.load('best_model.ckpt')
cnn.load_state_dict(S)
print('loaded!')
# Run evaluation
cnn.eval()
y_true = []
y_pred = []


with torch.no_grad():
    for wav, genre_index in test_loader:
        wav = wav.to(device)
        genre_index = genre_index.to(device)

        # reshape and aggregate chunk-level predictions
        b, c, t = wav.size()
        logits = cnn(wav.view(-1, t))
        logits = logits.view(b, c, -1).mean(dim=1)
        _, pred = torch.max(logits.data, 1)

        # append labels and predictions
        y_true.extend(genre_index.tolist())
        y_pred.extend(pred.tolist())

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

GTZAN_GENRES_Predict = ['electronic_Predict', 'classical_Predict', 'pop_Predict', 'ambient_Predict', 'hiphop_Predict', 'jazz_Predict', 'metal_Predict', 'reggae_Predict', 'rock_Predict', 'soundtrack_Predict']
#GTZAN_GENRES_Predict = ['blues_Predict', 'classical_Predict', 'country_Predict', 'disco_Predict', 'hiphop_Predict', 'jazz_Predict', 'metal_Predict', 'pop_Predict', 'reggae_Predict', 'rock_Predict']
#sns.heatmap(cm, annot=True, xticklabels=GTZAN_GENRES, yticklabels=GTZAN_GENRES_Predict, cmap='YlGnBu')
print('Accuracy: %.4f' % accuracy)
#plt.show()

#from here we added code

#sort numer of splits for each song

def count_lines_same_up_to_delimiter(filename, delimiter="_"):
    line_counts = []
    current_line = None
    count = 0
    prev_line = ''
    with open(filename, 'r') as file:
        for line in file:
            song = line.split(delimiter)[0]
            if prev_line == '':
                prev_line = song
                count = 1
                continue
            elif prev_line == song:
                count+=1
            else:
                prev_line = song
                line_counts.append(count)
                count = 1
    return line_counts


result_loader = get_dataloader(split='result')
file_test = "/home/ilayy/HD-25/docker_test/PycharmProjects/pythonProject1/result_filtered.txt"
line_counts = count_lines_same_up_to_delimiter(file_test)
print(line_counts)
cnn.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for wav, genre_index in tqdm(result_loader):
        wav = wav.to(device)
        genre_index = genre_index.to(device)

        # reshape and aggregate chunk-level predictions
        b, c, t = wav.size()
        logits = cnn(wav.view(-1, t))
        logits = logits.view(b, c, -1).mean(dim=1)
        _, pred = torch.max(logits.data, 1)

        # append labels and predictions
        y_true.extend(genre_index.tolist())
        y_pred.extend(pred.tolist())


def majority_rule(X_list, Y_list):
    Z_list = []
    index = 0
    for x in X_list:
        sublist = Y_list[index:index + x]
        if sublist:
            counter = Counter(sublist)
            most_common_element = counter.most_common(1)[0][0]
            Z_list.append(most_common_element)
        index += x
    return Z_list

y_true_mr = majority_rule(line_counts, y_true)
y_pred_mr = majority_rule(line_counts, y_pred)


accuracy = accuracy_score(y_true_mr, y_pred_mr)
cm = confusion_matrix(y_true_mr, y_pred_mr)
GTZAN_GENRES_Predict = ['electronic_Predict', 'classical_Predict', 'pop_Predict', 'ambient_Predict', 'hiphop_Predict', 'jazz_Predict', 'metal_Predict', 'reggae_Predict', 'rock_Predict', 'soundtrack_Predict']
#GTZAN_GENRES_Predict = ['blues_Predict', 'classical_Predict', 'country_Predict', 'disco_Predict', 'hiphop_Predict', 'jazz_Predict', 'metal_Predict', 'pop_Predict', 'reggae_Predict', 'rock_Predict']
sns.heatmap(cm, annot=True, xticklabels=GTZAN_GENRES, yticklabels=GTZAN_GENRES_Predict, cmap='YlGnBu')
print('Accuracy: %.4f' % accuracy)
plt.show()
"""
###################################################################################
x = torch.randn(1, 22050 * 29).to(device)
graph = torchviz.make_dot(cnn(x), params=dict(cnn.named_parameters()))
graph.render(filename='com_graph', format='png')

"""
print(1)
