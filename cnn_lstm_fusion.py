import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import gc
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set memory optimization flags and reproducibility
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
DELTA_T = 35  # Time difference between EEG and EMG in ms
WINDOW_SIZE = 200  # Window size for data processing
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
HIDDEN_DIM = 64

class MultimodalDataset(Dataset):
    def __init__(self, emg_data, eeg_data, labels, time_shift=DELTA_T):
        # Apply time shift to EMG data
        self.emg_data = torch.FloatTensor(emg_data)
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        self.time_shift = time_shift
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Shift EMG data by DELTA_T
        emg = self.emg_data[idx]
        eeg = self.eeg_data[idx]
        if self.time_shift > 0:
            emg = F.pad(emg[:, self.time_shift:], (0, self.time_shift))
        return emg, eeg, self.labels[idx]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.AvgPool1d(2)
        )
        
    def forward(self, x):
        return self.conv(x)

class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(CNNEncoder, self).__init__()
        self.cnn_blocks = nn.ModuleList([
            CNNBlock(input_channels, hidden_dim),
            CNNBlock(hidden_dim, hidden_dim * 2),
            CNNBlock(hidden_dim * 2, hidden_dim * 4),
            CNNBlock(hidden_dim * 4, hidden_dim * 8)
        ])
        
    def forward(self, x):
        for block in self.cnn_blocks:
            x = block(x)
        return x

class CNNLSTMFusion(nn.Module):
    def __init__(self, emg_channels, eeg_channels, hidden_dim, num_classes):
        super(CNNLSTMFusion, self).__init__()
        
        # CNN Encoders
        self.emg_encoder = CNNEncoder(emg_channels, hidden_dim)
        self.eeg_encoder = CNNEncoder(eeg_channels, hidden_dim)
        
        # LSTM layers
        lstm_input_dim = hidden_dim * 8 * 2  # Combined features from both modalities
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim * 4,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
    def forward(self, emg, eeg):
        # CNN feature extraction
        emg_features = self.emg_encoder(emg)
        eeg_features = self.eeg_encoder(eeg)
        
        # Combine features
        combined = torch.cat((emg_features, eeg_features), dim=1)
        
        # Reshape for LSTM
        batch_size = combined.size(0)
        seq_len = combined.size(2)
        combined = combined.permute(0, 2, 1)  # [batch, seq_len, features]
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(lstm_out)
        
        return output

def load_and_preprocess_data(emg_path, eeg_path, window_size=WINDOW_SIZE):
    print("Loading data...")
    
    # Load data
    emg_data = pd.read_csv(emg_path)
    eeg_data = pd.read_csv(eeg_path)
    
    # Extract features and labels
    emg_features = emg_data.iloc[:, :8].values  # 8 EMG channels
    eeg_features = eeg_data.iloc[:, :8].values  # 8 EEG channels
    
    # Standardize features
    print("Normalizing data...")
    emg_scaler = StandardScaler()
    eeg_scaler = StandardScaler()
    
    emg_features = emg_scaler.fit_transform(emg_features)
    eeg_features = eeg_scaler.fit_transform(eeg_features)
    
    # Create windowed data
    emg_windows = []
    eeg_windows = []
    window_labels = []
    sample_ids = []  # Track sample IDs for stratified splits
    
    # Find common samples between EMG and EEG data
    emg_samples = set(tuple(x) for x in emg_data[['subject', 'repetition', 'gesture']].drop_duplicates().values)
    eeg_samples = set(tuple(x) for x in eeg_data[['subject', 'repetition', 'gesture']].drop_duplicates().values)
    common_samples = emg_samples.intersection(eeg_samples)
    
    print(f"Found {len(common_samples)} common samples between EMG and EEG data.")
    
    for sample in common_samples:
        subject, repetition, gesture = sample
        
        # Get data for this sample
        emg_sample = emg_data[(emg_data['subject'] == subject) & 
                             (emg_data['repetition'] == repetition) & 
                             (emg_data['gesture'] == gesture)]
        
        eeg_sample = eeg_data[(eeg_data['subject'] == subject) & 
                             (eeg_data['repetition'] == repetition) & 
                             (eeg_data['gesture'] == gesture)]
        
        # Make sure both samples have data
        if len(emg_sample) == 0 or len(eeg_sample) == 0:
            continue
            
        # Extract features
        emg_sample_features = emg_sample.iloc[:, :8].values
        eeg_sample_features = eeg_sample.iloc[:, :8].values
        
        # Standardize using pre-fitted scalers
        emg_sample_features = emg_scaler.transform(emg_sample_features)
        eeg_sample_features = eeg_scaler.transform(eeg_sample_features)
        
        # Handle different lengths by using the shorter one
        min_length = min(len(emg_sample_features), len(eeg_sample_features))
        if min_length <= window_size:
            continue  # Skip if sample is too short
            
        emg_sample_features = emg_sample_features[:min_length]
        eeg_sample_features = eeg_sample_features[:min_length]
        
        # Create windows with fixed size
        for i in range(0, min_length - window_size, window_size // 2):
            emg_window = emg_sample_features[i:i + window_size]
            eeg_window = eeg_sample_features[i:i + window_size]
            
            # Only add if window is complete
            if len(emg_window) == window_size and len(eeg_window) == window_size:
                emg_windows.append(emg_window)
                eeg_windows.append(eeg_window)
                window_labels.append(gesture - 1)  # 0-indexed labels
                sample_ids.append(f"{subject}_{repetition}_{gesture}")
    
    if len(emg_windows) == 0:
        raise ValueError("No valid windows could be created. Check your data alignment.")
    
    print(f"Created {len(emg_windows)} windows from {len(common_samples)} samples.")
    
    # Convert to numpy arrays with explicit shape checking
    emg_windows = np.array(emg_windows)  # Shape: (n_windows, window_size, n_channels)
    eeg_windows = np.array(eeg_windows)  # Shape: (n_windows, window_size, n_channels)
    window_labels = np.array(window_labels)
    sample_ids = np.array(sample_ids)
    
    print(f"EMG windows shape: {emg_windows.shape}")
    print(f"EEG windows shape: {eeg_windows.shape}")
    
    return emg_windows, eeg_windows, window_labels, sample_ids

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    print("Starting training...")
    best_val_acc = 0.0
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for emg, eeg, labels in train_loader:
            emg, eeg, labels = emg.to(device), eeg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(emg, eeg)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for emg, eeg, labels in val_loader:
                emg, eeg, labels = emg.to(device), eeg.to(device), labels.to(device)
                outputs = model(emg, eeg)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)

def main():
    # Load and preprocess data
    emg_data, eeg_data, labels, sample_ids = load_and_preprocess_data(
        'data/processed/EMG-data.csv',
        'data/processed/EEG-data.csv'
    )
    
    # Split data
    X_emg_train, X_emg_test, X_eeg_train, X_eeg_test, y_train, y_test = train_test_split(
        emg_data, eeg_data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_emg_train, X_emg_val, X_eeg_train, X_eeg_val, y_train, y_val = train_test_split(
        X_emg_train, X_eeg_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create datasets
    train_dataset = MultimodalDataset(X_emg_train, X_eeg_train, y_train)
    val_dataset = MultimodalDataset(X_emg_val, X_eeg_val, y_val)
    test_dataset = MultimodalDataset(X_emg_test, X_eeg_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # Initialize model
    num_classes = len(np.unique(labels))
    model = CNNLSTMFusion(
        emg_channels=8,
        eeg_channels=8,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE/100)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Test best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for emg, eeg, labels in test_loader:
            emg, eeg, labels = emg.to(device), eeg.to(device), labels.to(device)
            outputs = model(emg, eeg)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'\nTest Accuracy: {test_acc:.2f}%')
    print(f'Test F1 Score: {test_f1:.4f}')
    print('\nConfusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()