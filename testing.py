# %% [markdown]
# # Multimodal EMG-EEG Gesture Classification using Deep Learning
# This notebook implements a multimodal deep learning approach combining EMG and EEG data for gesture classification.

# %% [markdown]
# ## Cell 1: Import Libraries and Set Environment
# Import necessary libraries and configure environment settings

# %%
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

# %% [markdown]
# ## Cell 2: Configuration and Hyperparameters
# Set up all configuration parameters, paths, and hyperparameters

# %%
# Data paths
EMG_DATA_PATH = 'data/processed/EMG-data.csv'
EEG_DATA_PATH = 'data/processed/EEG-data.csv'
MODEL_SAVE_PATH = 'best_model.pth'

# Data processing parameters
DELTA_T = 35  # Time difference between EEG and EMG in ms
WINDOW_SIZE = 200  # Window size for data processing

# Model hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
HIDDEN_DIM = 64
WEIGHT_DECAY = 1e-5

# Training settings
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# %% [markdown]
# ## Cell 3: Dataset Class Definition
# Implementation of the MultimodalDataset class for handling EMG and EEG data

# %%
class MultimodalDataset(Dataset):
    def __init__(self, emg_data, eeg_data, labels, time_shift=DELTA_T):
        # Data is already in shape (n_windows, n_channels, window_size)
        self.emg_data = torch.FloatTensor(emg_data)
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        self.time_shift = time_shift
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Shift EMG data by DELTA_T
        emg = self.emg_data[idx]  # Shape: (n_channels, window_size)
        eeg = self.eeg_data[idx]  # Shape: (n_channels, window_size)
        if self.time_shift > 0:
            emg = F.pad(emg[:, self.time_shift:], (0, self.time_shift))
        return emg, eeg, self.labels[idx]

# %% [markdown]
# ## Cell 4: Model Architecture Components
# Define the CNN blocks and encoder components

# %%
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
        # Input shape: [batch, channels, sequence]
        for block in self.cnn_blocks:
            x = block(x)
        return x

# %% [markdown]
# ## Cell 5: Main Model Architecture
# Implementation of the CNNLSTMFusion model combining EMG and EEG features

# %%
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
        # Input shapes are already [batch, channels, sequence]
        
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

# %% [markdown]
# ## Cell 6: Data Loading and Preprocessing
# Load and preprocess the EMG and EEG data

# %%
# Load and preprocess data
print("Loading and preprocessing data...")
emg_data, eeg_data, labels, sample_ids = load_and_preprocess_data(
    EMG_DATA_PATH,
    EEG_DATA_PATH
)

# Split data into train, validation, and test sets
X_emg_train, X_emg_test, X_eeg_train, X_eeg_test, y_train, y_test = train_test_split(
    emg_data, eeg_data, labels, 
    test_size=TRAIN_TEST_SPLIT, 
    random_state=RANDOM_SEED, 
    stratify=labels
)

X_emg_train, X_emg_val, X_eeg_train, X_eeg_val, y_train, y_val = train_test_split(
    X_emg_train, X_eeg_train, y_train, 
    test_size=VALIDATION_SPLIT, 
    random_state=RANDOM_SEED, 
    stratify=y_train
)

print("Data shapes:")
print(f"Training: EMG {X_emg_train.shape}, EEG {X_eeg_train.shape}")
print(f"Validation: EMG {X_emg_val.shape}, EEG {X_eeg_val.shape}")
print(f"Test: EMG {X_emg_test.shape}, EEG {X_eeg_test.shape}")

# %% [markdown]
# ## Cell 7: Create Data Loaders
# Prepare data loaders for training, validation, and testing

# %%
# Create datasets
train_dataset = MultimodalDataset(X_emg_train, X_eeg_train, y_train)
val_dataset = MultimodalDataset(X_emg_val, X_eeg_val, y_val)
test_dataset = MultimodalDataset(X_emg_test, X_eeg_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print("Data loaders created successfully")

# %% [markdown]
# ## Cell 8: Model Initialization
# Initialize the model, loss function, optimizer, and scheduler

# %%
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
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE/100)

print("Model initialized successfully")
print(f"Number of classes: {num_classes}")

# %% [markdown]
# ## Cell 9: Model Training
# Train the model using the prepared data loaders

# %%
# Train model
print("Starting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
print("Training completed")

# %% [markdown]
# ## Cell 10: Model Evaluation
# Evaluate the best model on the test set

# %%
# Load best model and evaluate
print("Evaluating best model on test set...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
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