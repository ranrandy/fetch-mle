from utils.config import eval_args
from utils.models import LSTM, Linear, MLP
from utils.fixseed import fixseed
from eval.evaluator import Evaluator


def main():
    args = eval_args()
    fixseed(args.seed)
        
    # Build the model
    if args.arch == 'linear':
        model = Linear(args).to(args.device)
    elif args.arch == 'lstm': # Deprecated
        model = LSTM(args).to(args.device)
    elif args.arch == 'mlp':
        model = MLP(args).to(args.device)

    # Evaluator
    evaluator = Evaluator(model, args)
    
    # Load the checkpoint and evaluate
    evaluator.load(args.model_path)
    evaluator.forecast()


if __name__=="__main__":
    main()