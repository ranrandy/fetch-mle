import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(args.input_dim, args.output_dim)

        self.initialize_weights(self.fc)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)

    def forward(self, x):
        '''
            x: (batch_size, input_dim)
        '''
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.output_dim),
        )
        for layer in self.mlp:
            self.initialize_weights(layer)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)

    def forward(self, x):
        '''
            x: (batch_size, input_dim)
        '''
        return self.mlp(x)


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.layer_dim = args.layer_dim

        self.lstm = nn.LSTM(args.input_dim, args.hidden_dim, args.layer_dim, batch_first=args.batch_first)
        self.fc = nn.Linear(args.hidden_dim, args.output_dim)

        self.initialize_weights(self.lstm)
        self.initialize_weights(self.fc)

    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

    def forward(self, x):
        '''
            x: (batch_size, input_len, input_dim)
        '''
        # h0, c0 are zeros by default
        out, _ = self.lstm(x)

        # We don't return the entire sequence, but only the output_len values
        # return shape: (batch_size, output_len, output_dim)
        return self.fc(out[:, -self.args.output_len:, :])



# Test the implementation
if __name__=="__main__":
    from utils.config import train_args
    args = train_args()
    print()

    # LSTM
    model = LSTM(args)
    print(model)
    x = torch.randn(16, 5, 1)
    print(model(x).shape)
    print()

    # MLP
    model = MLP(args)
    print(model)
    x = torch.randn(16, 1)
    print(model(x).shape)
    print()

    # Linear Regression
    model = Linear(args)
    print(model)
    x = torch.randn(16, 1)
    print(model(x).shape)
    print()
