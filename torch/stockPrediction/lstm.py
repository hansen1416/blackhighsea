import numpy
from sklearn.datasets import make_blobs
import torch

import preprocess
class MLP(torch.nn.Module):
    '''
        Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            # in_features, out_features
            # bias – If set to False, the layer will not learn an additive bias. Default: True
            torch.nn.Linear(32 * 32 * 3, 64),
            # torch.nn.Linear(20, 64),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


def blob_label(y, label, loc): # assign labels
    target = numpy.copy(y)
    for l in loc:
        target[y == l] = label
    return target


if __name__ == "__main__":

    ticker='000568'

    df = preprocess.get_stock_data(ticker)

    # print(df.tail())

    X, y, scaler_x, scaler_y = preprocess.scale_data(df)

    x_train = torch.from_numpy(X)

    print(x_train.shape)

    x_train, y_train = make_blobs(n_samples=40, n_features=32 * 32 * 3, \
        cluster_std=1.5, shuffle=True)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
    y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))

    print(x_train.size(), y_train.size())

    # Initialize the MLP
    mlp = MLP()
    # Define the loss function and optimizer
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    mlp.eval()

    mlp.train()
    epoch = 1

    for epoch in range(epoch):
        # sets the gradients to zero before we start backpropagation. 
        # This is a necessary step as PyTorch accumulates the gradients 
        # from the backward passes from the previous epochs.
        optimizer.zero_grad()
        # Forward pass
        y_pred = mlp(x_train)
        # Compute Loss
        loss = loss_function(y_pred.squeeze(), y_train)
    
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()

        # if i % 500 == 499:
        #     print('Loss after mini-batch %5d: %.3f' %
        #             (i + 1, current_loss / 500))
        #     current_loss = 0.0