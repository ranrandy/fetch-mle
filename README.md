# MLE Take Home Exercise - Fetch
Runfeng Li. 

Please contact me by runfeng4[at]illinois[dot]edu if you have any question.

## 1. Run the App
Turn on Docker on your computer, and run the following commands to build the docker image and run. You may need to wait 5-10 minutes until this image finishes setting up.
```
git clone https://github.com/ranrandy/fetch-mle.git
cd fetch-mle
docker build -t forecast_app .
docker run -it -p 5001:5000 forecast_app
```
Next, go to [http://localhost:5001/](http://localhost:5001/) to visit the app. If you encounter any error related to your port has been used, please change 5001 to any other numbers. For example, run `docker run -it -p 5002:5000 forecast_app`, and visit 
 [http://localhost:5002/](http://localhost:5002/).

Then you will see an interface like below

<img src="static/web_interface.png" width=800>

Click "choose the model", choose from "Linear Regression", "MLP", or "LSTM", and click "Forecast 2022" to get the animated predicted receipt count values (daily and monthly) in 2022, like below:

<i> Note: After clicking "forecast 2022", the inference procedure will start running, during which a predicted result video will be saved. After that, the app would show that video. Because it may take a while to inference, you would see the webpage "loading" for a little while.</i>

<i>If you don't want to wait that few seconds, you can go to the `app.py` file, and comment out those 3 `os.system()` lines. Then rebuild (well, this would take much longer time) or retype and run `flask run` at your local machine.</i>

<img src="static/web_interface_forecast.png" width=800>

You can see the forecasted monthly sum of receipt count from the numbers in the bottom video.


---


## 2. Run the code by yourself

### 2.1 Requirements

- [Python >= 3.9.4](https://www.python.org/downloads/)

You also need a few packages, please install by running
```
pip install -r requirements.txt
```

### 2.2 Train (and Test) the models

#### 2.2.1 Linear Regression & MLP
```
python -m train.train --arch linear --date True --norm True --steps 100
python -m train.train --arch mlp --date True --norm True --steps 100
```

#### 2.2.2 LSTM (Actually Including Inference Codes)
```
python keras_lstm.py
```
You could try the below command. But you need to finetune some hyperparameters in the `utils/config.py` file. Otherwise, the results won't be good
```
python -m train.train --arch lstm --norm True --steps 200
```

### 2.3 Forecast 2022 Values (Inference / Evaluation)
```
python -m eval.eval --arch linear --model_path 'save/linear_bs64_99.pth'
python -m eval.eval --arch mlp --hidden_dim 10 --model_path 'save/mlp_hid10_bs64_70.pth'
python keras_lstm_inference_only.py --model_path 'save/lstm_in20_out1_hid40_100'
```

### 2.4 Run the Flask App
Run the following command, and go to the returned web link
```
flask run
```

### 2.5 Misc.
#### 2.5.1 Train/Val Loss Curve (TensorBoard)
In `save/experiments`, you will see my training log files, (and yours if you retrained the model). To visualize the loss curves, run
```
tensorboard --logdir=save
```
Go to the returned link, probably [http://localhost:6006/](http://localhost:6006/). Then you will see an interface like below:

<img src='static/tensorboard.png' width=800>

I chose the model checkpoints based on the curves, 70 for MLP (right), 100 for Linear Regression (upper left), and 100 for LSTM (bottom).

#### 2.5.2 Exploratory Data Analysis
In `data/exploratory_data_analysis.ipynb`, I visualized some simple plots for this dataset, though the only takeaway was an linear positive relationship between the date and the receipt count.

#### 2.5.3 Test Cases
To test the implementation for `models.py` and `dataloader.py`, you can run
```
python -m utils.model
python -m utils.dataloader
```
Then you can check if the output are what you want.