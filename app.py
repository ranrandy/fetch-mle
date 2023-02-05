from flask import Flask, request, render_template
import os
import time

app = Flask(__name__)

EVAL_FOLDER = os.path.join('static', 'save_eval')
app.config['UPLOAD_FOLDER'] = EVAL_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        select = request.form.get('model_options')
        if select == 'Linear':
            os.system("python -m eval.eval --arch linear --date True --norm True --model_path 'save/linear_bs64_99.pth'")
            linear_forecasting_2022 = os.path.join(app.config['UPLOAD_FOLDER'], 'linear_forecasting_2022.mp4')
            linear_forecasting_2022_monthly = os.path.join(app.config['UPLOAD_FOLDER'], 'linear_forecasting_2022_monthly.mp4')
            return render_template(
                "forecast.html", 
                receipt_count_mp4=linear_forecasting_2022, 
                receipt_count_monthly_mp4=linear_forecasting_2022_monthly)
        elif select == 'MLP':
            os.system("python -m eval.eval --arch mlp --date True --norm True --hidden_dim 10 --model_path 'save/mlp_hid10_bs64_70.pth'")
            mlp_forecasting_2022 = os.path.join(app.config['UPLOAD_FOLDER'], 'mlp_forecasting_2022.mp4')
            mlp_forecasting_2022_monthly = os.path.join(app.config['UPLOAD_FOLDER'], 'mlp_forecasting_2022_monthly.mp4')
            return render_template(
                "forecast.html", 
                receipt_count_mp4=mlp_forecasting_2022, 
                receipt_count_monthly_mp4=mlp_forecasting_2022_monthly)
        elif select == 'LSTM':
            os.system("python keras_lstm_inference_only.py --model_path 'save/lstm_in20_out1_hid40_100.h5'")
            lstm_forecasting_2022 = os.path.join(app.config['UPLOAD_FOLDER'], 'lstm_forecasting_2022.mp4')
            lstm_forecasting_2022_monthly = os.path.join(app.config['UPLOAD_FOLDER'], 'lstm_forecasting_2022_monthly.mp4')
            return render_template(
                "forecast.html", 
                receipt_count_mp4=lstm_forecasting_2022, 
                receipt_count_monthly_mp4=lstm_forecasting_2022_monthly)
    
    # Load initial page with 2021 data
    receipt_count_2021 = os.path.join(app.config['UPLOAD_FOLDER'], '2021.png')
    return render_template("index.html", receipt_count=receipt_count_2021)


if __name__ == "__main__":
    app.run(debug=True)
