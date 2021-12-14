from flask import Flask, render_template, request
import pandas as pd



import pickle

model = None
with open('model_lr.pkl','rb') as pickle_model_file:
    model = pickle.load(pickle_model_file)

scaler = None
with open('full_scaler.pkl','rb') as pickle_scaler_file:
    scaler = pickle.load(pickle_scaler_file)

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/')
def main_html():
    return render_template('main.html')

@app.route('/diagnosis')
def diag_html():
    return render_template('diagnosis.html')

def feature_engneering(bmi):
    
    if bmi >= 25 :
        high_bmi = 1
    else :
        high_bmi = 0
    
    
    df['no_of_risk_factors'] = df['hypertension']+df['heart_disease']+df['smokes']+df['high_bmi']
    df['attrib1'] = df['avg_glucose_level'] * (1 + df['no_of_risk_factors'])


@app.route('/diag_input')
def diag_input():
    gender          = request.args.get("gender")
    age             = float(request.args.get("age"))
    hypertension    = int(request.args.get("hypertension"))
    heart_disease   = int(request.args.get("heart_disease"))
    avg_glucose_level = float(request.args.get("avg_glucose_level"))
    bmi             = float(request.args.get("bmi"))
    smokes          = float(request.args.get("smokes"))

    if float(bmi) >= 25 :
        high_bmi = 1
    else :
        high_bmi = 0

    no_of_risk_factors = hypertension + heart_disease + smokes + high_bmi
    attrib1 = avg_glucose_level + (1 + no_of_risk_factors)


    x = [[gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smokes, high_bmi, float(no_of_risk_factors), float(attrib1)]]

    column_name = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smokes', 'high_bmi', 'no_of_risk_factors', 'attrib1']
    x = pd.DataFrame(x, columns=column_name)

    print(x)
    x_prepared = scaler.transform(x)


    y_pred = model.predict(x_prepared)

    if y_pred == "0":
        datas = "당신은 뇌졸중의 위험이 낮습니다."
    else:
        datas = "당신은 뇌졸중의 위험이 높습니다. \n 지속적인 건강관리가 필요합니다. "

        
    return render_template('result.html', data=datas)
