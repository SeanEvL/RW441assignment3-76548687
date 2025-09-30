import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import copy

# Select the dataset to run: "car", "nursery", "yeast", or "bean"
DATASET_TO_RUN = "bean"

UNDERFIT_THRESHOLD_F1 = 0.90  
HIDDEN_NEURON_STEP = 5        
MAX_HIDDEN_NEURONS = 100      

if DATASET_TO_RUN == "car":
    LR = 0.01
    DROPOUT = 0.1
elif DATASET_TO_RUN == "nursery":
    LR = 0.001
    DROPOUT = 0.0
elif DATASET_TO_RUN == "yeast":
    LR = 0.001
    DROPOUT = 0.1
elif DATASET_TO_RUN == "bean":
    LR = 0.001
    DROPOUT = 0.0

EPOCHS = 500
PATIENCE = 10
BATCH_SIZE = 32

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_size > 0:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_size, output_size)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.hidden_size > 0 and self.fc1 is not None:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, lr, epochs, patience, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if len(all_labels) == 0:
        return 0.0, 0.0
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, macro_f1

def run_incremental_learning(X, y, class_order, device):
    history = []
    num_hidden = 0
    input_size = X.shape[1]
    
    for i in range(2, len(class_order) + 1):
        current_classes = class_order[:i]
        num_outputs = len(current_classes)
        print("-" * 50)
        print(f"Stage {i-1}: Training with {num_outputs} classes: {current_classes}")
        print("-" * 50)

        mask = np.isin(y, current_classes)
        X_current, y_current_original = X[mask], y[mask]
        
        label_map = {original_label: new_label for new_label, original_label in enumerate(current_classes)}
        y_current = np.array([label_map[label] for label in y_current_original], dtype=np.int64)
        
        X_temp, X_test_curr, y_temp, y_test_curr = train_test_split(X_current, y_current, test_size=0.2, stratify=y_current, random_state=1)
        X_train_curr, X_val_curr, y_train_curr, y_val_curr = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=1)
        
        train_ds = TensorDataset(torch.from_numpy(X_train_curr).float(), torch.from_numpy(y_train_curr))
        val_ds = TensorDataset(torch.from_numpy(X_val_curr).float(), torch.from_numpy(y_val_curr))
        test_ds = TensorDataset(torch.from_numpy(X_test_curr).float(), torch.from_numpy(y_test_curr))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
        
        while True:
            print(f"  Attempting training with {num_hidden} hidden neurons...")
            model = Net(input_size, num_hidden, num_outputs, dropout_rate=DROPOUT).to(device)
            model = train_model(model, train_loader, val_loader, lr=LR, epochs=EPOCHS, patience=PATIENCE, device=device)
            val_acc, val_f1 = evaluate(model, val_loader, device)
            print(f"    Validation -> Accuracy: {val_acc:.4f}, Macro F1: {val_f1:.4f}")
            
            if val_f1 < UNDERFIT_THRESHOLD_F1 and num_hidden < MAX_HIDDEN_NEURONS:
                print(f"    Underfitting detected (F1 {val_f1:.4f} < {UNDERFIT_THRESHOLD_F1}). Increasing architecture complexity.\n")
                num_hidden += HIDDEN_NEURON_STEP
            else:
                if num_hidden >= MAX_HIDDEN_NEURONS:
                    print(f"    Max hidden neurons ({MAX_HIDDEN_NEURONS}) reached. Finalizing architecture.")
                else:
                    print(f"    Performance is acceptable (F1 {val_f1:.4f} >= {UNDERFIT_THRESHOLD_F1}). Finalizing architecture for this stage.")
                
                test_acc, test_f1 = evaluate(model, test_loader, device)
                print(f"\n  Stage {i-1} Final -> Test Accuracy: {test_acc:.4f}, Test Macro F1: {test_f1:.4f} with {num_hidden} neurons.")
                history.append({'stage': i - 1, 'num_classes': num_outputs, 'classes': current_classes, 'hidden_neurons': num_hidden, 'test_accuracy': test_acc, 'test_f1': test_f1})
                break
    
    print("\n" + "="*50 + "\nIncremental Class Learning Finished\n" + "="*50)
    return model, history

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    le = LabelEncoder()
    scaler = StandardScaler()

    if DATASET_TO_RUN == "car":
        df = pd.read_csv("car.csv")
        df['class'] = le.fit_transform(df['class'])
        X = pd.get_dummies(df.drop('class', axis=1), drop_first=False).values
        y = df['class'].values
    elif DATASET_TO_RUN == "nursery":
        df = pd.read_csv("nursery.csv")
        df['class'] = le.fit_transform(df['class'])
        X = pd.get_dummies(df.drop('class', axis=1), drop_first=False).values
        y = df['class'].values
    elif DATASET_TO_RUN == "yeast":
        df = pd.read_csv("yeast.csv").drop("sequence_name", axis=1).rename(columns={'localization_site': 'class'})
        df['class'] = le.fit_transform(df['class'])
        X = scaler.fit_transform(df.drop('class', axis=1))
        y = df['class'].values
    elif DATASET_TO_RUN == "bean":
        df = pd.read_csv("dry_bean.csv").rename(columns={'Class': 'class'})
        df['class'] = le.fit_transform(df['class'])
        X = scaler.fit_transform(df.drop('class', axis=1))
        y = df['class'].values

    class_order = df['class'].value_counts().sort_values().index.tolist()
    print(f"Running ICL for dataset: {DATASET_TO_RUN}")
    print(f"Class order (minority to majority): {class_order}\n")
    
    final_model, training_history = run_incremental_learning(X, y, class_order, device)
    
    _, X_test_full, _, y_test_full = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    final_model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test_full).float().to(device)
        logits = final_model(test_tensor)
        predictions_remapped = torch.argmax(logits, dim=1).cpu().numpy()

    final_map_inv = {i: original_label for i, original_label in enumerate(class_order)}
    predictions_original = np.array([final_map_inv[p] for p in predictions_remapped])
    
    final_acc = accuracy_score(y_test_full, predictions_original)
    final_f1 = f1_score(y_test_full, predictions_original, average='macro', zero_division=0)

    print("\n--- ICL History ---")
    history_df = pd.DataFrame(training_history)
    print(history_df.to_string())
    print("\n--- Final ICL Model Performance on Full Test Set ---")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Macro F1-Score: {final_f1:.4f}")