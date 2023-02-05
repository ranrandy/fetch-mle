import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.config import *

import torch
from torch.utils.data import Dataset, DataLoader


# Only this function will be used by other files
def get_dataloader(args):
    # Load raw data
    data = load_data(args)
    if data is None:
        # sys.exit()
        return

    # Generate train, val and test sets
    train_data, val_data, test_data = split_data(data, args)
    
    # Normalize data
    train_mean, train_std = normalize_data(train_data, args)

    # Preprocess the data
    ### We only use the mean and std from the training set, 
    ### because we don't want to know any information from the val and test sets
    X_train, y_train = preprocess_data(train_data, args, train_mean, train_std)
    X_val, y_val = preprocess_data(val_data, args, train_mean, train_std)
    X_test, y_test = preprocess_data(test_data, args, train_mean, train_std)

    # Build torch datasets
    train_dataset = ReceiptCountDataset(X_train, y_train, args)
    val_dataset = ReceiptCountDataset(X_val, y_val, args)
    test_dataset = ReceiptCountDataset(X_test, y_test, args)

    # Build torch dataloaders
    train_dataloader = prepare_dataloader(train_dataset, args.train_bs, args, not (args.arch == 'lstm'))
    val_dataloader = prepare_dataloader(val_dataset, args.val_bs, args, False)
    test_dataloader = prepare_dataloader(test_dataset, args.test_bs, args, False)

    return train_dataloader, val_dataloader, test_dataloader, train_mean, train_std


def load_data(args):
    df = pd.read_csv(args.data_path, parse_dates=True, index_col="# Date")
    if args.date:
        df['day'] = df["Receipt_Count"].index.day_of_year
    if (args.arch == 'linear' or args.arch == 'mlp') and not args.date:
        print("Linear regression method must have the date as an input feature")
        return
    if (args.arch == 'lstm') and args.date:
        print("LSTM does not need the date as an input feature")
        return
    return df.to_numpy().reshape(len(df), -1).astype(np.float32)


def split_data(data, args):
    train_size, val_size = int(args.train_ratio * len(data)), int(args.val_ratio * len(data))
    
    if args.arch == 'linear' or args.arch == 'mlp':
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        # Manually split into sets
        train_data = data[indices[:train_size], :].reshape(train_size, -1)
        val_data = data[indices[train_size:train_size+val_size], :].reshape(val_size, -1)
        test_data = data[indices[train_size+val_size:], :].reshape(len(data)-train_size-val_size, -1)

    elif args.arch == 'lstm':
        train_data, val_data, test_data = \
            data[:train_size, :], data[train_size:train_size+val_size, :], data[train_size+val_size:, :]
        
        # Visualize dataset split using the Receipt_Count column
        plt.plot(range(train_size), train_data[:, 0], c='b')
        plt.plot(range(train_size, train_size+val_size), val_data[:, 0], c='y')
        plt.plot(range(train_size+val_size, len(data)), test_data[:, 0], c='g')
        plt.savefig(f'{args.save_dir}/dataset_split_lstm.png')
        plt.close()
    print(f"train:val:test = {len(train_data)}:{len(val_data)}:{len(test_data)}")
    return train_data, val_data, test_data


def normalize_data(train_data, args):
    train_mean, train_std = 0.0, 1.0
    if args.norm:
        train_mean, train_std = train_data.mean(axis=0), train_data.std(axis=0)
    np.savetxt(f"utils/train_norm_{args.arch}.txt", [train_mean, train_std])
    return train_mean, train_std


def preprocess_data(dataset, args,  mean, std):
    # Normalize
    dataset = (dataset - mean) / std

    if args.arch == 'linear' or args.arch == 'mlp':
        # Seperate into (X, y) pairs
        X = dataset[:, 1].reshape(-1, 1) # day_of_year
        y = dataset[:, 0].reshape(-1, 1) # Receipt_Count
        print(f"X shape -> {X.shape}, y shape -> {y.shape}")

    elif args.arch == 'lstm':
        # Init (X, y) pairs
        data_size = len(dataset) - (args.input_len + args.output_len)
        X = np.zeros((data_size, args.input_len, args.input_dim))
        y = np.zeros((data_size, args.output_len, args.output_dim))
        print(f"X shape -> {X.shape}, y shape -> {y.shape}")

        # Separate into (X, y) pairs
        for i in range(0, data_size, args.move):
            X[i] = dataset[i:i+args.input_len]
            y[i] = dataset[i+args.input_len:i+(args.input_len+args.output_len)]
    return X, y


class ReceiptCountDataset(Dataset):
    def __init__(self, X, y, args):
        self.args = args
        self.X, self.y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataloader(dataset, batch_size, args, shuffle):
    print("dataloader shuffled?", shuffle)
    data_loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return data_loader


# Test the implementation
if __name__=="__main__":
    from utils.config import train_args, eval_args
    t_args = train_args()
    e_args = eval_args()

    # train_dataloader, val_dataloader, test_dataloader, mean, std = get_dataloader(t_args)

    # for i, (data, target) in enumerate(train_dataloader):
    #     print(data.shape, target.shape)
    #     break
        
        
    # Test training
    case = 1
    for arch in ['linear', 'lstm', 'mlp']:
        t_args.arch = arch
        for b1 in [True, False]:
            t_args.date = b1
            for b2 in [True, False]:
                t_args.norm = b2
                print(f"\nCase {case}: arch={t_args.arch}, date={t_args.date}, norm={t_args.norm}")

                dataloaders = get_dataloader(t_args)
                if dataloaders is None:
                    case += 1
                    continue
                train_dataloader, val_dataloader, test_dataloader, mean, std = dataloaders
                for i, (data, target) in enumerate(train_dataloader):
                    print("data shape, target shape:", data.shape, target.shape)
                    # print("data example, target example", data[0], target[0])
                    print("mean, std:", mean, std)
                    break
                case += 1

    
    # Test evaluation
    print(f"\nCase {case}: evaluation")
    data = load_data(e_args)

    norm = np.loadtxt(f'utils/train_norm_{e_args.arch}.txt')
    mean, std = norm[0], norm[1]

    data = (data - mean) / std
    print("input shape & example:", data.shape, data[0])

    result = data # Evaluate
    result = result * std + mean
    print("output shape & example:", result.shape, result[0])



