from flask import Flask, render_template, request
import joblib

model = joblib.load('models/modelo_regresion.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    SepaloLargo = float(request.form['LargoSepalo'])
    SepaloAncho = float(request.form['AnchoSepalo'])
    PetaloLargo = float(request.form['LargoPetalo'])
    PetaloAncho = float(request.form['AnchoPetalo'])
    
    pred_probabilities = model.predict_proba([[SepaloLargo, SepaloAncho, PetaloLargo, PetaloAncho]])
    
    flores = model.classes_

    mensaje = ""
    for i, flor in enumerate(flores):
        prob = pred_probabilities[0, i] * 100
        mensaje += f"Probabilidad de {flor}: {round(prob, 2)}% <br/>"

    return render_template('result.html', predi=mensaje)

if __name__ == '__main__':
    app.run()