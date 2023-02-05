from utils.config import eval_args
from utils.dataloader import get_dataloader
from utils.models import LSTM, Linear, MLP
from utils.fixseed import fixseed

from train.trainer import Trainer


def main():
    args = eval_args()

    # Fix all random seeds
    fixseed(args.seed)

    # Get the data
    dataloaders = get_dataloader(args)

    # Build the model
    if args.arch == 'linear':
        model = Linear(args).to(args.device)
    elif args.arch == 'lstm': # Deprecated, please try keras-lstm.py
        model = LSTM(args).to(args.device)
    elif args.arch == 'mlp':
        model = MLP(args).to(args.device)

    # Trainer
    trainer = Trainer(model, dataloaders, args)
    
    # Load the checkpoint
    trainer.load()

    # Test
    trainer.test()


if __name__=="__main__":
    main()