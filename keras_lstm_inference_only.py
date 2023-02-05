import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.config import train_args, eval_args

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense


def load_data(args):
	raw_seq = pd.read_csv(args.data_path)['Receipt_Count'].to_numpy()
	# Because the model may be too complicated for the normalized data
	# Here we let mean and std be 0 and 1
	seq_mean, seq_std = 0, 1 #raw_seq.mean(axis=0), raw_seq.std(axis=0)
	# Normalize the data
	raw_seq = (raw_seq - seq_mean) / seq_std
	return raw_seq, seq_mean, seq_std


def main():
	args = eval_args()
	args.arch = 'lstm'

	# Set random seed
	tf.keras.utils.set_random_seed(args.seed+10)

	# Load data
	data, data_mean, data_std = load_data(args)

	model = tf.keras.models.load_model(args.model_path)
	model.compile(optimizer='adam', loss='mse')

	# Autoregressively forecast the future
	result = []
	x_input = data[-args.input_len:].reshape((1, args.input_len, args.input_dim))
	for _ in range(365):
		yhat = model.predict(x_input, verbose=0).reshape(1, 1, args.input_dim)
		x_input = np.concatenate([x_input[:, 1:, :], yhat], axis=1)
		result.append(yhat[0, 0, 0].tolist())

	# Denormalize the data
	data = data * data_std + data_mean
	results = np.array(result) * data_std + data_mean

	# Visualize the result
	# Daily  
	fig = plt.figure(figsize=(16, 8))
	plt.xlim(pd.Timestamp('2021-01-01'), pd.Timestamp('2023-01-01'))
	plt.ylim(5e6, 1.4e7)
	plt.xlabel("Date")
	plt.ylabel("Receipt Count")
	plt.title("Receipt Count Forecasting at Fetch Rewards - 2022")
	plt.style.use("seaborn")
	
	data_idx = [pd.Timestamp('2021-01-01') + pd.offsets.Day(i) for i in range(len(data))]
	plt.plot(data_idx, data)

	result_idx = [pd.Timestamp('2022-01-01') + pd.offsets.Day(i) for i in range(len(results))]

	x, y = [], []
	def animate(i):
		x.append(result_idx[i])
		y.append((results[i]))
		plt.plot(x,y, scaley=True, scalex=True, color="green", linestyle='--')

	ani = FuncAnimation(fig=fig, func=animate, frames=365, interval=20)
	ani.save(f"{args.save_dir}/{args.arch}_forecasting_2022.mp4", writer = 'ffmpeg', fps = 60)
	plt.close()

	# Monthly
	fig = plt.figure(figsize=(16, 8))
	plt.xlim(pd.Timestamp('2021-01-01'), pd.Timestamp('2023-01-01'))
	plt.ylim(1.5e8, 4e8)
	plt.xlabel("Date")
	plt.ylabel("Receipt Count")
	plt.title("Receipt Count Forecasting at Fetch Rewards - 2022 Monthly")
	plt.style.use("seaborn")
	
	# Prev data
	data_idx = [pd.Timestamp('2020-12-31') + pd.offsets.MonthEnd(i) for i in range(1, 13)]
	data_monthly, day_of_year = [], 0
	for d in data_idx:
		data_monthly.append(sum(data[day_of_year:day_of_year+d.day]))
		day_of_year += d.day
	data_idx = [d + pd.offsets.Day(1) for d in data_idx]
	plt.plot(data_idx, data_monthly)

	# Future data
	result_idx = [pd.Timestamp('2021-12-31') + pd.offsets.MonthEnd(i) for i in range(1, 13)]
	result_monthly, day_of_year = [], 0
	for d in result_idx:
		result_monthly.append(sum(results[day_of_year:day_of_year+d.day]))
		day_of_year += d.day
	result_idx = [d + pd.offsets.Day(1) for d in result_idx]

	# Connect years
	result_idx = [data_idx[-1]] + result_idx
	result_monthly = [data_monthly[-1]] + result_monthly

	# Animate
	x, y = [], []
	def animate(i):
		x.append(result_idx[i])
		y.append(result_monthly[i])
		plt.plot(x,y, scaley=True, scalex=True, color="green", linestyle='--')
		plt.annotate(f'{result_monthly[i]}', xy=(x[-1], y[-1]), xytext=(x[-1], y[-1]+5e7*(i%2*2-1)))

	ani = FuncAnimation(fig=fig, func=animate, frames=12, interval=20)
	ani.save(f"{args.save_dir}/{args.arch}_forecasting_2022_monthly.mp4", writer = 'ffmpeg', fps = 2)
	plt.close()


if __name__ == "__main__":
	main()
