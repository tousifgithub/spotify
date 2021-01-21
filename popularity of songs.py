# -*- 

#popularity of songs


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

tracks_data_df = pd.read_csv('I:\Downloads(IDM) & Browser\spotify_data/data.csv')
tracks_data_df.head()

tracks_data_df.tail()


tracks_data_df.describe()
#Visualizing Data

tracks_data_df.hist(figsize=(15, 15), color='black')
plt.show()

plt.figure(figsize=(20, 10))
sns.heatmap(tracks_data_df.corr(), annot=True)


#As you can see above, year, danceability, energy, loudness and tempo are important features for predicting popularity.

#Let's take a look on the correlations between these features and popularity:
    
    
sns.scatterplot(x="danceability", y="popularity", data=tracks_data_df, alpha=0.03, color='blue')


sns.scatterplot(x="energy", y="popularity", data=tracks_data_df, alpha=0.03, color='blue')    


sns.scatterplot(x="loudness", y="popularity", data=tracks_data_df, alpha=0.03, color='blue')


sns.scatterplot(x="tempo", y="popularity", data=tracks_data_df, alpha=0.03, color='blue')    


 
 

features = ['year', 'danceability', 'energy', 'loudness', 'tempo']
tracks_data = tracks_data_df.copy()
features_tracks_data = tracks_data_df[features]


class Artist: 
    def __init__(self, name, popularity): 
        self.name = name
        self.popularity = popularity
        
        
class Track: 
    def __init__(self, name, artists, popularity): 
        self.name = name
        self.artists = artists
        self.popularity = popularity   
        
        

tracks = []

names = tracks_data.name.values
artists_names = tracks_data.artists.values
popularity = tracks_data.popularity.values

for index in range(len(names)): 
    track = Track(names[index], artists_names[index], popularity[index])
    tracks.append(track)
    
artists = []
artists_names_done = []
artists_popularities = []

for artists_str in tqdm(artists_names): 
    artists_sub_list = artists_str[1:-1].split(', ')
    
    track_pop = 0
    for artist in artists_sub_list: 
        
        if artist in artists_names_done: 
            a = [x for x in artists if x.name == artist][0]
            artist_pop = a.popularity
            
        else: 
            songs_pop = [x.popularity for x in tracks if artist in x.artists]
            artist_pop = sum(songs_pop) / len(songs_pop)
            artists_names_done.append(artist)
            a = Artist(artist, artist_pop)
            artists.append(a)
            
            
            
            track_pop += artist_pop
    track_pop /= len(artists_sub_list)
    artists_popularities.append(track_pop)
    
artists_popularities = np.asarray(artists_popularities)

print(artists_popularities.max())



scaler = StandardScaler()
scaler.fit(features_tracks_data)
features_tracks_data = scaler.transform(features_tracks_data)

print(features_tracks_data.shape) 
features_tracks_data = np.column_stack((artists_popularities / 100, features_tracks_data))
print(features_tracks_data.shape)

y_tracks_data = tracks_data.popularity.values / 100

X_train, X_test, y_train, y_test = train_test_split(features_tracks_data, y_tracks_data, test_size=0.2, random_state=42)


for column in range(X_train.shape[1]): 
    print(X_train[:, column].min(), X_train[:, column].max())


clf = RandomForestRegressor()
clf.fit(X_train, y_train)



preds = clf.predict(X_test)

accuracy = clf.score(X_test, y_test)
print("Test Accuracy: {:.4f}".format(accuracy*100))

average_error = (abs(y_test - preds)).mean()
print("{:.4f} average error".format(average_error))


for index in range(len(preds[:100])): 
    
    pred = preds[index]
    actual = y_test[index]
    
    print("Actual / Predicted: {:.4f} / {:.4f}".format(actual, pred))