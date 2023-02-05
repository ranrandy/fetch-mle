import torch
import torch.nn as nn
from torch.optim import Adam

# from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import numpy as np


class Trainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.train_dataloader, self.val_dataloader, self.test_dataloader, self.mean, self.std = dataloaders
        self.args = args

        self.loss_fn = nn.MSELoss()
        self.device = args.device

        # Batch sizes
        self.train_bs = args.train_bs
        self.val_bs = args.val_bs
        self.test_bs = args.test_bs

        # LSTM Parameters
        if args.arch == 'lstm':
            self.input_len = args.input_len
            self.output_len = args.output_len

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim

    def _build_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones)

    def train(self):
        self._build_optimizer()
        self.writer = SummaryWriter(self.args.log_dir)

        self.step = 0
        self.num_steps = self.args.steps
        self.num_epochs = self.num_steps // len(self.train_dataloader) + 1

        for _ in range(self.num_epochs):
            if self.step >= self.num_steps:
                break
            for X, y in self.train_dataloader:
                self.model.train()
                if self.args.arch in ['linear', 'mlp']:
                    # -1 -> batch_size or last_batch_size
                    X = X.view(-1, self.input_dim).to(self.device)
                    y = y.view(-1, self.output_dim).to(self.device)
                elif self.args.arch == 'lstm':
                    X = X.view(-1, self.input_len, self.input_dim).to(self.device)
                    y = y.view(-1, self.output_len, self.output_dim).to(self.device)

                # Compute loss
                output = self.model(X)
                loss = self.loss_fn(output, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                
                # Log
                if self.step % self.args.log_interval == 0:
                    val_loss = self.validate()

                    print(f"step {self.step} -> train loss: {loss.item() / self.train_bs:>8f}")
                    print(f"step {self.step} -> val loss: {val_loss / self.val_bs:>8f} \n")
                    self.writer.add_scalars(f'Loss/{self.args.arch}', {
                        'train': loss.item() / self.train_bs,
                        'val': val_loss / self.val_bs,
                    }, self.step)

                # Save
                if self.step % self.args.save_interval == 0:
                    self.save()
                
                # Update step
                self.step += 1
        
        # Save latest
        if self.step % self.args.save_interval != 0:
            self.save()

    def validate(self):
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in self.val_dataloader:
                if self.args.arch == 'linear':
                    X = X.view(-1, self.input_dim).to(self.device)
                    y = y.view(-1, self.output_dim).to(self.device)
                elif self.args.arch == 'lstm':
                    X = X.view(-1, self.input_len, self.input_dim).to(self.device)
                    y = y.view(-1, self.output_len, self.output_dim).to(self.device)
                output = self.model(X)
                val_loss += self.loss_fn(output, y).item()
        return val_loss / len(self.val_dataloader)
    
    def save(self):
        if self.args.arch == 'linear':
            torch.save(
                self.model.state_dict(), 
                f'{self.args.save_dir}/{self.args.arch}_bs{self.train_bs}_{self.step}.pth'
            )
        elif self.args.arch == 'mlp':
            torch.save(
                self.model.state_dict(), 
                f'{self.args.save_dir}/{self.args.arch}_hid{self.hidden_dim}_bs{self.train_bs}_{self.step}.pth'
            )
        elif self.args.arch == 'lstm':
            torch.save(
                self.model.state_dict(), 
                f'{self.args.save_dir}/{self.args.arch}_in{self.input_len}_out{self.output_len}_hid{self.hidden_dim}_bs{self.train_bs}_{self.step}.pth'
            )

    def load(self):
        model = torch.load(self.args.model_path)

    def metrics(self, output, y):
        diff = output - y
        self.mse += np.mean(diff ** 2)
        self.mae += np.mean(np.abs(diff))
        self.rmse += np.sqrt(self.mse)
        self.r2 += 1 - (np.sum(diff ** 2) / np.sum((y - np.mean(y)) ** 2))

    def test(self):
        test_loss = 0
        self.mse, self.mae, self.rmse, self.r2 = 0, 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_dataloader:
                if self.args.arch in ['linear', 'mlp']:
                    X = X.view(-1, self.input_dim).to(self.device)
                    y = y.view(-1, self.output_dim).to(self.device)
                elif self.args.arch == 'lstm':
                    X = X.view(-1, self.input_len, self.input_dim).to(self.device)
                    y = y.view(-1, self.output_len, self.output_dim).to(self.device)         
                output = self.model(X)
                test_loss += self.loss_fn(output, y).item()

                # Evaluation Metrics
                self.metrics(output.cpu().numpy(), y.cpu().numpy())
                # print(self.mse, type(self.mse))
        
        test_loss /= len(self.test_dataloader)
        self.mse /= len(self.test_dataloader)
        self.mae /= len(self.test_dataloader)
        self.rmse /= len(self.test_dataloader)
        self.r2 /= len(self.test_dataloader)

        print(f"test loss: {test_loss / self.args.test_bs:>8f} \n")
        print(f"mean squared error: {self.mse / self.args.test_bs:>8f} \n")
        print(f"mean absolute error: {self.mae / self.args.test_bs:>8f} \n")
        print(f"root mean squared error: {self.rmse / self.args.test_bs:>8f} \n")
        print(f"r2 score: {self.r2 / self.args.test_bs:>8f} \n")
