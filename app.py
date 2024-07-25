from flask import Flask, render_template, request
import os
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/prediction",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('prediction.html')
    else:
        data=CustomData(
            year=int(request.form.get('year')),
            kilometers=int(request.form.get('kilometers')),
            fuelType=request.form.get('fuelType'),
            transmission=request.form.get('transmission'),
            Owner_Type=request.form.get('ownerType'),
            Mileage=request.form.get('mileage'),
            Engine=request.form.get('engine'),
            Power=request.form.get('power'),
            Seats=request.form.get('seats'),
            Brand=request.form.get('brand'),
            Model=request.form.get('model'),
            region=request.form.get('region'),
        )
        pred_df=data.get_data_as_data_frame()
        # print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        re=results[0]
        round_re=round(re,3)-4
        return render_template('prediction.html',results=round_re)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
    
