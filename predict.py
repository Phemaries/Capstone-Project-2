from flask import Flask
from flask import request
from flask import jsonify
import sklearn
import pickle
import pandas as pd

app = Flask("music_genre")
dv, modelrcl = pickle.load(open("modelrcl.pkl", "rb"))


@app.route('/predict', methods=['POST'])
def predict():
    music = request.get_json()

    X = dv.transform([music])
    y_pred = modelrcl.predict(X)
    # music_genre = y_pred >= 0.5

    result = {
        'music_status': str(y_pred),
        # 'music_genre': str(music_genre)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
