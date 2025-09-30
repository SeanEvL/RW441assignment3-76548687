import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Select the dataset to run: "car", "nursery", "yeast", or "bean"
DATASET_TO_RUN = "car"

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        if hidden_size > 0:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_size, output_size)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.hidden_size > 0:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, lr=0.001, epochs=100, patience=5, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    return model

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, macro_f1

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    le = LabelEncoder()
    scaler = StandardScaler()
    print(f"Using device: {device}")
    print(f"Running baseline tuning for dataset: {DATASET_TO_RUN}")

    # Data Loading and Preprocessing
    if DATASET_TO_RUN == "car":
        df = pd.read_csv("car.csv")
        df['class'] = le.fit_transform(df['class'])
        features = pd.get_dummies(df.drop('class', axis=1), drop_first=False)
        X = features.values.astype(np.float32)
        y = df['class'].values.astype(np.int64)
    elif DATASET_TO_RUN == "nursery":
        df = pd.read_csv("nursery.csv")
        df['class'] = le.fit_transform(df['class'])
        features = pd.get_dummies(df.drop('class', axis=1), drop_first=False)
        X = features.values.astype(np.float32)
        y = df['class'].values.astype(np.int64)
    elif DATASET_TO_RUN == "yeast":
        df = pd.read_csv("yeast.csv")
        df = df.drop("sequence_name", axis=1)
        df = df.rename(columns={'localization_site': 'class'})
        df['class'] = le.fit_transform(df['class'])
        features = scaler.fit_transform(df.drop('class', axis=1))
        X = features.astype(np.float32)
        y = df['class'].values.astype(np.int64)
    elif DATASET_TO_RUN == "bean":
        df = pd.read_csv("dry_bean.csv")
        df = df.rename(columns={'Class': 'class'})
        df['class'] = le.fit_transform(df['class'])
        features = scaler.fit_transform(df.drop('class', axis=1))
        X = features.astype(np.float32)
        y = df['class'].values.astype(np.int64)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=1)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    input_size = X.shape[1]
    output_size = len(np.unique(y))
    hidden_sizes = [0, 10, 25, 50, 100]
    learning_rates = [0.0001, 0.001, 0.01]
    dropout_rates = [0.0, 0.1, 0.2, 0.3]
    
    best_val_f1 = 0
    best_config = None
    best_model = None

    for hidden in hidden_sizes:
        for lr in learning_rates:
            for dropout in dropout_rates:
                model = Net(input_size, hidden, output_size, dropout).to(device)
                model = train_model(model, train_loader, val_loader, lr=lr, epochs=100, patience=5, device=device)
                train_acc, train_f1 = evaluate(model, train_loader, device)
                val_acc, val_f1 = evaluate(model, val_loader, device)
                print(f"Hidden {hidden}, lr {lr}, dropout {dropout}: train acc {train_acc:.4f} f1 {train_f1:.4f}, val acc {val_acc:.4f} f1 {val_f1:.4f}")
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_config = (hidden, lr, dropout)
                    best_model = model
    
    test_acc, test_f1 = evaluate(best_model, test_loader, device)
    print(f"\nBest baseline for {DATASET_TO_RUN}: hidden {best_config[0]}, lr {best_config[1]}, dropout {best_config[2]}, test acc {test_acc:.4f} macro F1 {test_f1:.4f}")