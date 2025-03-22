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
INITIAL_LR = 0.001  # Initial learning rate
MIN_LR = 1e-6      # Minimum learning rate
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 64

# Training settings
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Advanced training settings
GRAD_CLIP = 1.0    # Gradient clipping threshold
LABEL_SMOOTHING = 0.1  # Label smoothing factor
SCHEDULER_GAMMA = 0.95  # Learning rate decay factor

# Learning rate scheduler settings
NUM_EPOCHS = 300  # Will run for full 100 epochs
WARMUP_EPOCHS = 5
CYCLES = 3  # Number of cosine annealing cycles
CYCLE_LEN = NUM_EPOCHS // CYCLES  # Length of each cycle
T_MULT = 2  # Factor to increase cycle length after each cycle
ETA_MIN = MIN_LR  # Minimum learning rate for cosine annealing

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
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
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
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        print('-' * 60)

# %% [markdown]
# ## Cell 4: Dataset Class Definition
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
# ## Cell 5: Model Architecture Components
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
# ## Cell 6: Main Model Architecture
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
train_dataset = MultimodalDataset(X_emg_train, X_eeg_train, y_train)
val_dataset = MultimodalDataset(X_emg_val, X_eeg_val, y_val)
test_dataset = MultimodalDataset(X_emg_test, X_eeg_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print("Data loaders created successfully")

# %% [markdown]
# ## Cell 9: Model Initialization
# Initialize the model, loss function, and optimizer with advanced learning rate scheduling

# %%
# Initialize model
num_classes = len(np.unique(labels))
model = CNNLSTMFusion(
    emg_channels=8,
    eeg_channels=8,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes
).to(device)

# Define loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

# Define learning rate schedulers
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, 
    start_factor=0.1,
    end_factor=1.0,
    total_iters=WARMUP_EPOCHS
)

main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=CYCLE_LEN,
    T_mult=T_MULT,
    eta_min=MIN_LR
)

print("Model initialized successfully")
print(f"Number of classes: {num_classes}")

# %% [markdown]
# ## Cell 10: Model Training
# Train the model for full 100 epochs using advanced learning rate scheduling

# %%
# Train model
print("Starting training...")
best_val_acc = 0.0
best_val_f1 = 0.0
best_epoch = 0
scaler = GradScaler()

# Training history
history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': [],
    'lr': [], 'grad_norm': []
}

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_preds = []
    train_labels_list = []
    epoch_grad_norm = 0.0
    
    # Training phase
    for batch_idx, (emg, eeg, labels) in enumerate(train_loader):
        emg, eeg, labels = emg.to(device), eeg.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(emg, eeg)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        epoch_grad_norm += grad_norm.item()
        
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        train_preds.extend(predicted.cpu().numpy())
        train_labels_list.extend(labels.cpu().numpy())
        
        # Print batch progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}] - '
                  f'Loss: {loss.item():.4f}')
    
    # Update learning rate schedulers
    if epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        main_scheduler.step()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels_list = []
    
    with torch.no_grad():
        for emg, eeg, labels in val_loader:
            emg, eeg, labels = emg.to(device), eeg.to(device), labels.to(device)
            outputs = model(emg, eeg)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            val_preds.extend(predicted.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())
    
    # Calculate metrics
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    train_f1 = f1_score(train_labels_list, train_preds, average='weighted')
    val_f1 = f1_score(val_labels_list, val_preds, average='weighted')
    current_lr = optimizer.param_groups[0]['lr']
    avg_grad_norm = epoch_grad_norm / len(train_loader)
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['lr'].append(current_lr)
    history['grad_norm'].append(avg_grad_norm)
    
    # Print progress
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
    print(f'LR: {current_lr:.6f}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
    print(f'Gradient Norm: {avg_grad_norm:.4f}')
    
    # Save best model based on both accuracy and F1 score
    if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
            'main_scheduler_state_dict': main_scheduler.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'train_f1': train_f1,
            'history': history,
            'hyperparameters': {
                'label_smoothing': LABEL_SMOOTHING,
                'grad_clip': GRAD_CLIP,
                'initial_lr': INITIAL_LR,
                'batch_size': BATCH_SIZE,
                'hidden_dim': HIDDEN_DIM
            }
        }, MODEL_SAVE_PATH)
        print(f'New best model saved with validation accuracy: {val_acc:.2f}% and F1: {val_f1:.4f}')
    
    print('-' * 80)

print(f"Best model was saved at epoch {best_epoch+1}")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Best validation F1 score: {best_val_f1:.4f}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

# Plot losses
plt.subplot(2, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracies
plt.subplot(2, 3, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

# Plot F1 scores
plt.subplot(2, 3, 3)
plt.plot(history['train_f1'], label='Train F1')
plt.plot(history['val_f1'], label='Validation F1')
plt.title('F1 Score History')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

# Plot learning rate
plt.subplot(2, 3, 4)
plt.plot(history['lr'])
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')

# Plot gradient norms
plt.subplot(2, 3, 5)
plt.plot(history['grad_norm'])
plt.title('Gradient Norm History')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')

plt.tight_layout()
plt.show()

print("Training completed")

# %% [markdown]
# ## Cell 11: Model Evaluation
# Evaluate the best model on the test set

# %%
# Load best model and evaluate
print("Evaluating best model on test set...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH)['model_state_dict'])
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