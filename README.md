# Music Genre Prediction 

This repository contains EDA and prediction of musical genres such as  'Electronic', 'Anime', 'Jazz', 'Alternative', 'Country', 'Rap', 'Blues', 'Rock', 'Classical', 'Hip-Hop'.Also, Docker Containerization was achieved in this project.


## Why this Project?

Access to internet and technology disruption has made it easier to stream music all over the world which has now become a lifestyle. A lifestyle of the people in the globe is dependent on music and building a classification of music genre to enhance users is key. This project aims at developing a machine model that predicts the type of a musical genre based on musical details and audio features of what is provided.

## Data
The data (with unique 50,000 records) is sourced from Kaggle. Check the dataset [here](https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre/data).

### Data Features

**instance_id:** A unique identifier for each instance in the dataset, represented as a floating-point number. There are 50,000 instances in total

**artist_name:** The name of the artist who created the track, represented as a string (object).

**track_name:** The name of the track, represented as a string (object)

**popularity:** A measure of the popularity of the track, represented as a floating-point number between 0 and 100. Higher values indicate greater popularity.

**acousticness:** A measure of the acousticness of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater acousticness.

**danceability:** A measure of the danceability of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater danceability.

**duration_ms:** The duration of the track in milliseconds, represented as a floating-point number.

**energy:** A measure of the energy of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater energy.

**instrumentalness:** A measure of the instrumentalness of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater instrumentalness.

**key:** The key in which the track is performed, represented as a string (object). There are 12 possible values: 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', and 'B'.

**liveness:** A measure of the liveness of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater liveness.

**loudness:** A measure of the loudness of the track, represented as a floating-point number in decibels (dB). Higher values indicate greater loudness.

**mode:** The mode in which the track is performed, represented as a string (object). There are two possible values: 'major' and 'minor'.

**speechiness:** A measure of the speechiness of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater speechiness.

**tempo:** The tempo of the track in beats per minute (BPM), represented as a string (object).

**obtained_date:** The date on which the track information was obtained, represented as a string (object).

**valence:** A measure of the valence (positivity) of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater positivity.

**music_genre:** The genre of the track, represented as a string (object). There are multiple possible genres in the dataset.
 


### Exploratory Data Analysis (EDA)
An extensive EDA was carried out on the dataset. Data had 5 null values, duplicates removed and 50,000 unique instance were created. This also provided some basic answers to providing a best fit for prediction. Upon cleaning and feature engineering, **42,491** records were analyzed.


### Model Training.
Multiple models were trained and compared for the dataset (60% training/ 20% validation / 20% testing). Models such as Logistic Regression, RandomForestClassifier, DecisionTreeClassifier, and RidgeClassifier with optimized parameters were selected to get the best metrics based on ##accuracy, ##confusion matrix, ##precision, ##recall, and ##f1_score. 

### Model Predictions on Validation set

| Model                              | Without Hyperparameters (Accuracy)   |With Hyperparameters (Accuracy)   | 
| :-------------                     |:-------------:                       |-------------:| 
| Logistic Regression                | 35%                                  |45%   | 
| DecisionTree Classifier            | 48%                                  |52%   | 
| Random Forest Classifier           | 61%                                  |53%   |
| DRidge Classifier                  | 77%                                  |78%   |  



### Cross-Validation
The Ridge Classifier performed well across 10-folds with mean accuracy of 78% on validation set

### Conclusion
Upon hypermetric tuning, the model `RidgeClassifier(alpha=1, random_state=42)` has an accuracy of **80%** on full data. Convert to py file using `jupyter nbconvert Project_prediction.ipynb --to python`. Change file name to `train.py` and run file.

### Model and Deployment to Flask
* The best model is saved into `modelrcl.pkl` with the `dv` and `modelrcl` features.
* waitress-serve --listen=0.0.0.0:9696 predict:app
* Create a virtual environment using: python -m venv env
* In the project directory run `env\Scripts\activate`
* Install all packages `pip install [packages]`
* `pip freeze > requirements.txt` to create `requirement.txt` file
* run `python predict_test.py`

### Containerization
* Build the Docker file: docker build -t music-genre .
* To run it: docker run -it -p 9696:9696 music-genre:latest


### Future Scope
*  Deployment to Cloud

