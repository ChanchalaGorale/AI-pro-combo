from flask import Flask , jsonify
from flask_cors import CORS

app=Flask(__name__)

CORS(app, resources={r"/api/*": {"origins":"https://voluble-taiyaki-30e7ef.netlify.app/"}})

@app.route("/api/data", methods=["GET", "POST"])
def get_data():
    data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "123-456-7890"
    }

    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', 'https://voluble-taiyaki-30e7ef.netlify.app')  
    return response




if __name__=="__main__":
    app.run(debug=True)
