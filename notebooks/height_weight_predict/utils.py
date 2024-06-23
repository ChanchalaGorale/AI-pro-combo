import pickle
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()



def predict_height(weight):
    print(weight)
    with open("notebooks/height_weight_predict/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    
    prediction = model.predict(scaler.fit_transform([[int(weight)]]))

    print(prediction)

    return prediction[0]