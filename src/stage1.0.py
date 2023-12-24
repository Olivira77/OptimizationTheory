import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from utils.iostream import H5adInStream
from utils.preprocess import normalize_rows_and_columns


path = "../Data"
batch_size = 64

instreamer = H5adInStream(path=path)

instreamer.read_datasets()
labels_dict = instreamer.get_unique_label(obs_name="cell.type")
data_dict = instreamer.convert_data_to_numpy(obs_name="cell.type")

if __name__ == '__main__':
    for dataset_name, data in tqdm(data_dict.items()):
        print(f'Current dataset: {dataset_name}')
        if 'test' in dataset_name:
            X_test = torch.tensor(data[:, :-1], dtype=torch.float32)
            continue
        X_train, y_true = (torch.tensor(data[:, :-1], dtype=torch.float32, requires_grad=True),
                           torch.tensor(data[:, -1], dtype=torch.long))
        in_features = X_train.shape[1]
        out_features = len(labels_dict[dataset_name])
        
        train = TensorDataset(X_train, y_true)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        model = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in tqdm(range(5)):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            train_logits = model(X_train).detach().numpy()
            test_logits = model(X_test).detach().numpy()
            
            train_predictions = np.bincount(np.argmax(train_logits, axis=1))
            test_predictions = np.bincount(np.argmax(test_logits, axis=1))
            
            train_predictions = train_predictions / train_predictions.sum()
            test_predictions = test_predictions / test_predictions.sum()
            
            train_predictions.tofile(f'../public/Y/stage1.0/{dataset_name[:-6]}_train_pred.bin')
            test_predictions.tofile(f'../public/Y/stage1.0/{dataset_name[:-6]}_test_pred.bin')
            
            train_logits = normalize_rows_and_columns(train_logits)
            test_logits = normalize_rows_and_columns(test_logits)
            
            train_logits.tofile(f'../public/Y/stage1.0/{dataset_name[:-6]}_train_logits.bin')
            test_logits.tofile(f'../public/Y/stage1.0/{dataset_name[:-6]}_test_logits.bin')
            