from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)

# Load trained ML pipeline
model = pickle.load(open("house_price_pipeline.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # =============================
    # 1. Get form data
    # =============================

    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])

    parking = int(request.form['parking'])

    # Convert Yes/No to 1/0
    mainroad = 1 if request.form['mainroad'] == "yes" else 0
    guestroom = 1 if request.form['guestroom'] == "yes" else 0
    basement = 1 if request.form['basement'] == "yes" else 0
    hotwaterheating = 1 if request.form['hotwaterheating'] == "yes" else 0
    airconditioning = 1 if request.form['airconditioning'] == "yes" else 0
    prefarea = 1 if request.form['prefarea'] == "yes" else 0

    # Furnishing encoding
    furnishing_map = {
        "furnished": 2,
        "semi-furnished": 1,
        "unfurnished": 0
    }

    furnishingstatus = furnishing_map[request.form['furnishingstatus']]

    # =============================
    # 2. Create input array
    # =============================

    input_data = [[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]]

    # =============================
    # 3. Predict price
    # =============================

    prediction = model.predict(input_data)

    predicted_price = round(prediction[0], 2)

    return f"Predicted House Price: ₹{predicted_price}"


if __name__ == "__main__":
    app.run(debug=True)