import pickle
import numpy as np


def predict_car_price(data):

    with open("notebooks/car_price_prediction/model12.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    arr = np.array(data, dtype=np.float64)
        
    result = model.predict([arr])[0]

    return result
