"""Multi-task neural network with a masked loss (from notebook 05, section 4).

One network predicts all targets at once; missing target values are skipped in the
loss, so no values have to be filled in.
"""

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

HIDDEN_SIZES = (512, 256)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE = 25


def create_network(n_inputs, n_outputs):
    layers, size = [], n_inputs
    for hidden in HIDDEN_SIZES:
        layers += [nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(DROPOUT)]
        size = hidden
    layers.append(nn.Linear(size, n_outputs))
    return nn.Sequential(*layers)


# Mean squared error per target over the available labels, averaged over targets
def masked_loss(predictions, targets):
    mask = ~torch.isnan(targets)
    squared_error = torch.where(mask, predictions - torch.nan_to_num(targets), 0.0) ** 2
    counts = mask.sum(dim=0)
    per_target = squared_error.sum(dim=0) / counts.clamp(min=1)
    return per_target[counts > 0].mean()


# Train a network on X (numpy) and Y (numpy, NaN = missing) and return a predict function
def fit_network(X, Y, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Set aside rows for early stopping
    order = rng.permutation(len(X))
    n_stop = int(0.1 * len(X))
    stop_idx, fit_idx = order[:n_stop], order[n_stop:]

    # Scale the features and the targets using the fitting rows only
    preprocess = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X_fit = preprocess.fit_transform(X[fit_idx])
    X_stop = preprocess.transform(X[stop_idx])
    y_mean = np.nanmean(Y[fit_idx], axis=0)
    y_std = np.nanstd(Y[fit_idx], axis=0)

    to_tensor = lambda array: torch.tensor(array, dtype=torch.float32)
    X_fit, X_stop = to_tensor(X_fit), to_tensor(X_stop)
    Y_fit = to_tensor((Y[fit_idx] - y_mean) / y_std)
    Y_stop = to_tensor((Y[stop_idx] - y_mean) / y_std)

    network = create_network(X.shape[1], Y.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    best_loss, best_state, waited = np.inf, None, 0
    for epoch in range(MAX_EPOCHS):
        network.train()
        for batch in torch.randperm(len(X_fit)).split(BATCH_SIZE):
            loss = masked_loss(network(X_fit[batch]), Y_fit[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            stop_loss = masked_loss(network(X_stop), Y_stop).item()
        if stop_loss < best_loss:
            best_loss, waited = stop_loss, 0
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    network.load_state_dict(best_state)
    network.eval()

    def predict(X_new):
        with torch.no_grad():
            scaled = network(to_tensor(preprocess.transform(X_new))).numpy()
        return scaled * y_std + y_mean

    return predict
