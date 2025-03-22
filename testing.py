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

# New hyperparameters
WARMUP_EPOCHS = 5
DROPOUT_RATE = 0.3
ATTENTION_HEADS = 4
AUGMENTATION_PROB = 0.5
LABEL_SMOOTHING = 0.1

# %% [markdown]
# ## Cell 3: Helper Functions
# Define data loading and training functions

# %%
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
                # Transpose the windows to have shape (channels, time_steps)
                emg_windows.append(emg_window.T)  # Shape: (8, window_size)
                eeg_windows.append(eeg_window.T)  # Shape: (8, window_size)
                window_labels.append(gesture - 1)  # 0-indexed labels
                sample_ids.append(f"{subject}_{repetition}_{gesture}")
    
    if len(emg_windows) == 0:
        raise ValueError("No valid windows could be created. Check your data alignment.")
    
    print(f"Created {len(emg_windows)} windows from {len(common_samples)} samples.")
    
    # Convert to numpy arrays with explicit shape checking
    emg_windows = np.array(emg_windows)  # Shape: (n_windows, n_channels, window_size)
    eeg_windows = np.array(eeg_windows)  # Shape: (n_windows, n_channels, window_size)
    window_labels = np.array(window_labels)
    sample_ids = np.array(sample_ids)
    
    print(f"EMG windows shape: {emg_windows.shape}")
    print(f"EEG windows shape: {eeg_windows.shape}")
    
    return emg_windows, eeg_windows, window_labels, sample_ids

def augment_signal(signal, prob=AUGMENTATION_PROB):
    """Apply various augmentations to the signal."""
    if random.random() < prob:
        # Random scaling
        scale = random.uniform(0.8, 1.2)
        signal = signal * scale
    
    if random.random() < prob:
        # Add Gaussian noise
        noise = torch.randn_like(signal) * 0.05
        signal = signal + noise
    
    if random.random() < prob:
        # Random time shift
        shift = random.randint(-10, 10)
        signal = torch.roll(signal, shifts=shift, dims=-1)
    
    return signal

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    print("Starting training...")
    best_val_acc = 0.0
    scaler = GradScaler()
    
    # Learning rate warmup
    warmup_factor = LEARNING_RATE / WARMUP_EPOCHS
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Warmup learning rate
        if epoch < WARMUP_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_factor * (epoch + 1)
        
        for emg, eeg, labels in train_loader:
            emg, eeg, labels = emg.to(device), eeg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(emg, eeg)
                # Label smoothing
                loss = criterion(outputs, labels) * (1 - LABEL_SMOOTHING) + \
                       LABEL_SMOOTHING * torch.mean(outputs) / num_classes
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        if epoch >= WARMUP_EPOCHS:
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
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)

# %% [markdown]
# ## Cell 4: Dataset Class Definition
# Implementation of the MultimodalDataset class for handling EMG and EEG data

# %%
class MultimodalDataset(Dataset):
    def __init__(self, emg_data, eeg_data, labels, time_shift=DELTA_T, augment=False):
        self.emg_data = torch.FloatTensor(emg_data)
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        self.time_shift = time_shift
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        emg = self.emg_data[idx]
        eeg = self.eeg_data[idx]
        
        if self.augment:
            emg = augment_signal(emg)
            eeg = augment_signal(eeg)
        
        if self.time_shift > 0:
            emg = F.pad(emg[:, self.time_shift:], (0, self.time_shift))
        
        return emg, eeg, self.labels[idx]

# %% [markdown]
# ## Cell 5: Model Architecture Components
# Define the CNN blocks and encoder components

# %%
class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualCNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ELU()
        self.pool = nn.AvgPool1d(2)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        out = self.pool(out)
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=DROPOUT_RATE)
        self.norm = nn.LayerNorm([embed_dim])
        
    def forward(self, x):
        # Input shape: [batch, channels, seq_len]
        batch_size, channels, seq_len = x.size()
        
        # Reshape for attention [seq_len, batch, channels]
        x_reshaped = x.permute(2, 0, 1)
        
        # Apply self-attention
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Reshape back to [batch, channels, seq_len]
        attn_out = attn_out.permute(1, 2, 0)
        
        # Apply residual connection and normalization
        out = x + attn_out
        
        # Reshape for layer norm [batch, seq_len, channels]
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        
        # Reshape back to original format [batch, channels, seq_len]
        out = out.permute(0, 2, 1)
        
        return out

class ImprovedCNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(ImprovedCNNEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            ResidualCNNBlock(input_channels, hidden_dim),
            ResidualCNNBlock(hidden_dim, hidden_dim * 2),
            ResidualCNNBlock(hidden_dim * 2, hidden_dim * 4),
            ResidualCNNBlock(hidden_dim * 4, hidden_dim * 8)
        ])
        self.attention = MultiHeadAttention(hidden_dim * 8, ATTENTION_HEADS)
        
    def forward(self, x):
        # Apply CNN blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply attention
        x = self.attention(x)
        return x

# %% [markdown]
# ## Cell 6: Main Model Architecture
# Implementation of the CNNLSTMFusion model combining EMG and EEG features

# %%
class ImprovedCNNLSTMFusion(nn.Module):
    def __init__(self, emg_channels, eeg_channels, hidden_dim, num_classes):
        super(ImprovedCNNLSTMFusion, self).__init__()
        
        # Improved encoders
        self.emg_encoder = ImprovedCNNEncoder(emg_channels, hidden_dim)
        self.eeg_encoder = ImprovedCNNEncoder(eeg_channels, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = MultiHeadAttention(hidden_dim * 8, ATTENTION_HEADS)
        
        # LSTM with residual connections
        lstm_input_dim = hidden_dim * 8 * 2
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim * 4,
            num_layers=2,
            batch_first=True,
            dropout=DROPOUT_RATE,
            bidirectional=True
        )
        
        # Improved classifier with proper dimensions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(DROPOUT_RATE * 0.8),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
    def forward(self, emg, eeg):
        # Extract features with shape tracking
        # Input shape: [batch, channels, seq_len]
        emg_features = self.emg_encoder(emg)  # Shape: [batch, hidden_dim*8, seq_len]
        eeg_features = self.eeg_encoder(eeg)  # Shape: [batch, hidden_dim*8, seq_len]
        
        # Concatenate along channel dimension
        combined = torch.cat((emg_features, eeg_features), dim=1)  # Shape: [batch, hidden_dim*16, seq_len]
        
        # Apply cross-modal attention
        combined = self.cross_attention(combined)  # Shape: [batch, hidden_dim*16, seq_len]
        
        # Prepare for LSTM
        combined = combined.permute(0, 2, 1)  # Shape: [batch, seq_len, hidden_dim*16]
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)  # Shape: [batch, seq_len, hidden_dim*8]
        
        # Global pooling
        avg_pool = torch.mean(lstm_out, dim=1)  # Shape: [batch, hidden_dim*8]
        max_pool, _ = torch.max(lstm_out, dim=1)  # Shape: [batch, hidden_dim*8]
        
        # Combine pooling results
        pooled = (avg_pool + max_pool) / 2  # Shape: [batch, hidden_dim*8]
        
        # Classification
        output = self.classifier(pooled)  # Shape: [batch, num_classes]
        
        return output

# %% [markdown]
# ## Cell 7: Data Loading and Preprocessing
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
# ## Cell 8: Create Data Loaders
# Prepare data loaders for training, validation, and testing

# %%
# Create datasets
train_dataset = MultimodalDataset(X_emg_train, X_eeg_train, y_train, augment=True)
val_dataset = MultimodalDataset(X_emg_val, X_eeg_val, y_val, augment=False)
test_dataset = MultimodalDataset(X_emg_test, X_eeg_test, y_test, augment=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print("Data loaders created successfully")

# %% [markdown]
# ## Cell 9: Model Initialization
# Initialize the model, loss function, optimizer, and scheduler

# %%
# Initialize model
num_classes = len(np.unique(labels))
model = ImprovedCNNLSTMFusion(
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
# ## Cell 10: Model Training
# Train the model using the prepared data loaders

# %%
# Train model
print("Starting training...")
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
print("Training completed")

# %% [markdown]
# ## Cell 11: Model Evaluation
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