import requests

url = 'http://localhost:9696/predict'


instance_id = 77879

music = {
  "artist_name": "Big Sean",
  "popularity": 70,
  "acousticness": 0.10,
  "danceability": 0.10,
  "energy": 0.01,
  "instrumentalness": 0.12,
  "liveness": 0.12, 
  "loudness": 1.00,
  "mode": 0,
  "speechiness": 0.05,
  "tempo": 34.0,
  "valence": 0.221,
  "key_A": 0,
  "key_A#": 0,
  "key_B": 0,
  "key_C": 0,
  "key_C#": 0,
  "key_D": 0,
  "key_D#": 1,
  "key_E": 0,
  "key_F": 0,
  "key_F#": 0,
  "key_G": 0,
  "key_G#": 0,
  

}


response = requests.post(url, json=music).json()
print(response)

# if response['music_genre'] == True:
#     print(f'Patient {instance_id} has a high chance of having a cardiovascular disease')
# else:
#     print(f'Patient {instance_id} is not under threat')