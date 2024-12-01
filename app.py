from flask import Flask, render_template, request, jsonify
from src.utils.utils import load_object
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

import os

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def predict_datapoint():
    if request.method =="GET":
        return render_template("form.html")
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = request.form.get('cut'),
            color = request.form.get('color'),
            clarity = request.form.get('clarity')
        )
        final_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_data)

        result = round(pred[0],2)

        return render_template("result.html", final_result = result)

@app.route('/reports', methods=["GET"])
def get_reports():
    report_path:str=os.path.join("artifacts", "report.pkl")
    print(report_path)
    data = load_object(report_path)
    print(data)
    return render_template("reports.html", data = data)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)