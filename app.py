from flask import Flask

app=Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def get_data():
    data = {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "123-456-7890"
    }
    return jsonify(data)




if __name__=="__main__":
    app.run(debug=True)
