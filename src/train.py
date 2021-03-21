import argparse
import json
import os
import pickle
import sys
import joblib
# import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier
from utils import load_model


def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = joblib.load(os.path.join(training_dir, 'train.joblib'))

    # first 6 are the targets
    train_y = torch.from_numpy(train_data[:, :6]).float()
    train_X = torch.from_numpy(train_data[:, 6:]).long()

    print(f'{train_y.shape}, {train_X.shape}')

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            # Forward
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding-dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--num-lstm-layers', type=int, default=1, metavar='N',
                        help='number of LSTM layers (default: 1)')
    parser.add_argument('--hidden-dims', type=int, default=[100], metavar='N',
                        nargs='+', help='size of the hidden dimensions (default: [100])')
    parser.add_argument('--vocab-size', type=int, default=10002, metavar='N',
                        help='size of the vocabulary (default: 10002)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.getenv('SM_HOSTS', '[]')))
    parser.add_argument('--current-host', type=str, default=os.getenv('SM_CURRENT_HOST', ''))
    parser.add_argument('--model-dir', type=str, default=os.getenv('SM_MODEL_DIR', 'model/'))
    parser.add_argument('--data-dir', type=str, default=os.getenv('SM_CHANNEL_TRAINING', 'processed/'))
    parser.add_argument('--num-gpus', type=int, default=os.getenv('SM_NUM_GPUS', '0'))

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTMClassifier(args.embedding_dim, args.num_lstm_layers, args.hidden_dims,
                           args.vocab_size).to(device)
    print(model)

    with open(os.path.join(args.data_dir, "vocab.joblib"), "rb") as f:
        model.word_dict = joblib.load(f)

    print("Model loaded with embedding_dim {}, lstm_layers {}, hidden_dims {}, vocab_size {}.".format(
        args.embedding_dim, args.num_lstm_layers, args.hidden_dims, args.vocab_size
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'lstm_layers': args.lstm_layers,
            'hidden_dim': args.hidden_dims,
            'vocab_size': args.vocab_size,
        }
        torch.save(model_info, f)

    # Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'vocab.joblib')
    with open(word_dict_path, 'wb') as f:
        joblib.dump(model.word_dict, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
