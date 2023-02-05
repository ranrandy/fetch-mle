from argparse import ArgumentParser


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--device", default='cpu', type=str, help="Device to use.")
    group.add_argument("--seed", default=12345, type=int, help="For fixing the random seed.")


def add_data_options(parser):
    group = parser.add_argument_group('data')
    group.add_argument("--data_path", default="data/data_daily.csv", type=str, help="Relative path to the data.")
    group.add_argument("--date", default=False, type=bool, help="If using the date as one feature.")
    group.add_argument("--norm", default=False, type=bool, help="If normalizing the input features.")

    group.add_argument("--train_ratio", default=0.7, type=float, help="Train set ratio.")
    group.add_argument("--val_ratio", default=0.1, type=float, help="Validation set ratio.")
    group.add_argument("--test_ratio", default=0.2, type=float, help="Test set ratio.")
    group.add_argument("--num_workers", default=4, type=int, help="Number of workers in the dataloader.")

    group.add_argument("--train_bs", default=64, type=int, help="Batch size during training.")
    group.add_argument("--val_bs", default=16, type=int, help="Batch size during validation")
    group.add_argument("--test_bs", default=16, type=int, help="Batch size during testing")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='linear', choices=['lstm', 'linear', 'mlp'], type=str, help="Architecture types.")
    
    # LSTM Parameters
    group.add_argument("--input_len", default=20, type=int, help="Length of the input.")
    group.add_argument("--output_len", default=1, type=int, help="Length of the output.")
    group.add_argument("--move", default=1, type=int, help="Interval for picking new sequences.")
    group.add_argument("--layer_dim", default=1, type=int, help="Stacked LSTM layers.")
    group.add_argument("--batch_first", default=True, type=bool, help="If batch_first in nn.LSTM")

    # Common Parameters
    group.add_argument("--input_dim", default=1, type=int, help="Number of input features.")
    group.add_argument("--hidden_dim", default=40, type=int, help="LSTM or MLP hidden dimension.")
    group.add_argument("--output_dim", default=1, type=int, help="Number of output features.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", default="save", type=str, help="Path to save checkpoints and results.")

    # Used in keras-lstm.py (Keras)
    group.add_argument("--epochs", default=1, type=int, help="Number of epochs to train.")

    # Used in train/train.py (PyTorch)
    group.add_argument("--steps", default=10, type=int, help="Number of steps to update the gradients.") 
    group.add_argument("--lr", default=0.1, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0, type=float, help="Optimizer weight decay.")

    group.add_argument("--log_interval", default=1, type=int, help="Interval for logging to the TensorBoard.")
    group.add_argument("--log_dir", default='save/experiments', type=str, help="Directory for logging.")
    group.add_argument("--save_interval", default=10, type=int, help="Interval for saving the model.")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", default="save/***.pth", type=str, help="Path to the ***.pth file.")
    group.add_argument("--eval_batch_size", default=16, type=int, help="Batch size during evaluation.")
    group.add_argument("--days", default=365, type=int, help="Number of days to forecast.")
    group.add_argument("--save_dir", default="static/save_eval", type=str, help="Path to save evaluation results.")


def common_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    return parser


def train_args():
    parser = common_args()
    add_training_options(parser)
    return parser.parse_args()


def eval_args():
    parser = common_args()
    add_evaluation_options(parser)
    return parser.parse_args()