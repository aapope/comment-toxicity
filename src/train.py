import argparse
from collections import defaultdict
import json
import os
import pickle
import sys
import joblib
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from model import LSTMClassifier


def model_fn(model_dir):
    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        model_info['embedding_dim'],
        model_info['num_lstm_layers'],
        model_info['hidden_dims'],
        model_info['vocab_size']
    )

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'vocab.joblib')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = joblib.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model


def _get_data_loader(batch_size, data_dir, is_training=True):
    if is_training:
        print("Get train data loader.")
    else:
        print('Get validation/test data loader')

    train_data = joblib.load(os.path.join(data_dir, 'data.joblib'))

    # first 6 are the targets
    train_y = torch.from_numpy(train_data[:, :6]).float()
    train_X = torch.from_numpy(train_data[:, 6:]).long()
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    if is_training:
        print('Upsampling records with any positive labels...')
        # upsample examples where any label=1. should help the class imbalance...
        any_label = train_data[:, :6].max(axis=1)
        class_weights = [any_label.mean(), 1 - any_label.mean()]
        weights = [class_weights[any_label[i]] for i in range(any_label.shape[0])]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        return torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        return train_X, train_y


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


def log_eval_metrics(model, val_X, val_y, device):
    val_X = val_X.to(device)
    predict_y = model(val_X).cpu().detach().numpy()

    metrics = defaultdict(list)
    for label in range(6):
        print(label)
        val_y_class = val_y[:, label]
        predict_y_class = predict_y[:, label]
        predict_y_class_label = predict_y_class.round()

        metrics['accuracy'].append(accuracy_score(val_y_class, predict_y_class_label))
        metrics['precision'].append(precision_score(val_y_class, predict_y_class_label))
        metrics['recall'].append(recall_score(val_y_class, predict_y_class_label))
        metrics['roc_auc'].append(roc_auc_score(val_y_class, predict_y_class))

    avg_metrics = {}
    for name, results in metrics.items():
        print(f'{name} at class level: {results}')
        avg_metrics[name] = sum(results) / len(results)

    print(*[f'{name}: {result};' for name, result in avg_metrics.items()])
    
    
    

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
    parser.add_argument('--train-data-dir', type=str, default=os.getenv('SM_CHANNEL_TRAINING', 'processed/train/'))
    parser.add_argument('--val-data-dir', type=str, default=os.getenv('SM_CHANNEL_EVAL', 'processed/val/'))
    parser.add_argument('--num-gpus', type=int, default=os.getenv('SM_NUM_GPUS', '0'))

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_data_loader(args.batch_size, args.train_data_dir, is_training=True)
    val_X, val_y = _get_data_loader(args.batch_size, args.val_data_dir, is_training=False)

    # Build the model.
    model = LSTMClassifier(args.embedding_dim, args.num_lstm_layers, args.hidden_dims,
                           args.vocab_size).to(device)
    print(model)

    with open(os.path.join(args.train_data_dir, "vocab.joblib"), "rb") as f:
        model.word_dict = joblib.load(f)

    print("Model loaded with embedding_dim {}, lstm_layers {}, hidden_dims {}, vocab_size {}.".format(
        args.embedding_dim, args.num_lstm_layers, args.hidden_dims, args.vocab_size
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)
    log_eval_metrics(model, val_X, val_y, device)

    os.makedirs(args.model_dir, exist_ok=True)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'num_lstm_layers': args.num_lstm_layers,
            'hidden_dims': args.hidden_dims,
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
