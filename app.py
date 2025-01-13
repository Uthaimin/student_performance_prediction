from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Setup logging
logging.info("Handling /predictdata request")


# Route for home page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            logging.info("Handling /predictdata request")
            logging.info("Form data received: %s", request.form.to_dict())
            
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            pred_df = data.get_data_as_data_frame()
            logging.info("DataFrame Created: %s", pred_df)
            logging.info("Before Prediction")

            predict_pipeline = PredictPipeline()
            logging.info("PredictPipeline instantiated")
            results = predict_pipeline.predict(pred_df)
            logging.info("Prediction Results: %s", results)
            logging.info("After Prediction")
            
            return render_template('home.html', results=results[0])
        except Exception as e:
            logging.error("Error occurred: %s", e)
            return render_template('home.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
