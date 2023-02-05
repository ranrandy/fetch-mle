import torch
from utils.config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Evaluator:
    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.data = pd.read_csv(args.data_path)['Receipt_Count'].to_numpy().tolist()
        self.input = torch.arange(365, 730).float().reshape(-1, args.input_dim) 

        norm = np.loadtxt(f'utils/train_norm_{self.args.arch}.txt')
        self.data_mean = norm[0]
        self.data_std = norm[1]
        # self.normalized_data = (self.data - self.data_mean) / self.data_std
        self.input = (self.input - self.data_mean[1]) / self.data_std[1]
        self.results = []

        if args.arch == 'lstm':
            # LSTM parameters
            self.input_len = args.input_len
            self.output_len = args.output_len

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

    def forecast(self):
        self.model.eval()
        with torch.no_grad():

            if self.args.arch in ['linear', 'mlp']:
                X = self.input
                out = self.model(X)
                self.results = out.detach().numpy()

            elif self.args.arch == 'lstm':
                X = torch.Tensor(self.normalized_data[:self.input_len]).view(1, self.input_len, self.input_dim)
                for _ in range(0, self.args.days, self.output_len):
                    out = self.model(X)

                    # Connect the previous X with the new output
                    prev_X = X[:, self.output_len:, :].view(-1, self.input_len-self.output_len, self.input_dim)
                    X = torch.cat([prev_X, out], dim=1)
                    self.results.append(out[0].detach().numpy()[:, 0])

        self.results = self._inverse_normalize(self.results).tolist()
        self._visualize()

    def _inverse_normalize(self, data):
        return np.array(data).flatten() * self.data_std[0] + self.data_mean[0]

    def _visualize(self):
        # Daily  
        fig = plt.figure(figsize=(16, 8))
        plt.xlim(pd.Timestamp('2021-01-01'), pd.Timestamp('2023-01-01'))
        plt.ylim(5e6, 1.4e7)
        plt.xlabel("Date")
        plt.ylabel("Receipt Count")
        plt.title("Receipt Count Forecasting at Fetch Rewards - 2022")
        plt.style.use("seaborn")
        
        data_idx = [pd.Timestamp('2021-01-01') + pd.offsets.Day(i) for i in range(len(self.data))]
        plt.plot(data_idx, self.data)

        result_idx = [pd.Timestamp('2022-01-01') + pd.offsets.Day(i) for i in range(len(self.results))]

        x, y = [], []
        def animate(i):
            x.append(result_idx[i])
            y.append((self.results[i]))
            plt.plot(x,y, scaley=True, scalex=True, color="green", linestyle='--')

        ani = FuncAnimation(fig=fig, func=animate, frames=365, interval=20)
        ani.save(f"{self.args.save_dir}/{self.args.arch}_forecasting_2022_{self.args.model_path[-6:-4]}.mp4", writer = 'ffmpeg', fps = 60)
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
            data_monthly.append(sum(self.data[day_of_year:day_of_year+d.day]))
            day_of_year += d.day
        data_idx = [d + pd.offsets.Day(1) for d in data_idx]
        plt.plot(data_idx, data_monthly)

        # Future data
        result_idx = [pd.Timestamp('2021-12-31') + pd.offsets.MonthEnd(i) for i in range(1, 13)]
        result_monthly, day_of_year = [], 0
        for d in result_idx:
            result_monthly.append(sum(self.results[day_of_year:day_of_year+d.day]))
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
        ani.save(f"{self.args.save_dir}/{self.args.arch}_forecasting_2022_{self.args.model_path[-6:-4]}_monthly.mp4", writer = 'ffmpeg', fps = 2)
        plt.close()

        
        

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
