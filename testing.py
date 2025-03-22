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

# Create a custom dataset for multimodal data - Fixed to load data to GPU only when needed
class MultimodalDataset(Dataset):
    def __init__(self, emg_data, eeg_data, labels):
        self.emg_data = torch.FloatTensor(emg_data)
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.emg_data[idx], self.eeg_data[idx], self.labels[idx]

# Define the improved EMG encoder network with residual connections and attention
class EMGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EMGEncoder, self).__init__()
        # Deeper architecture with residual blocks and attention
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # First residual block
        self.res1 = ResidualBlock(hidden_dim, hidden_dim*2)
        self.se1 = SEBlock(hidden_dim*2)
        
        # Second residual block
        self.res2 = ResidualBlock(hidden_dim*2, hidden_dim*4)
        self.se2 = SEBlock(hidden_dim*4)
        
        # Self-attention block
        self.attention = SelfAttention(hidden_dim*4)
        
        # Global pooling and feature combination
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim*8, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels]
        x = x.permute(0, 2, 1)  # [batch_size, channels, sequence_length]
        
        # Initial projection
        x = self.input_proj(x)
        
        # First residual block with SE
        x = self.res1(x)
        x = self.se1(x)
        
        # Second residual block with SE
        x = self.res2(x)
        x = self.se2(x)
        
        # Self-attention
        x = self.attention(x)
        
        # Global pooling
        avg_pool = self.gap(x).squeeze(-1)
        max_pool = self.gmp(x).squeeze(-1)
        
        # Concatenate and project
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.output_proj(x)
        
        return x

# Define the improved EEG encoder network with residual connections and attention
class EEGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EEGEncoder, self).__init__()
        # Mirror the EMG encoder architecture for symmetry
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # First residual block
        self.res1 = ResidualBlock(hidden_dim, hidden_dim*2)
        self.se1 = SEBlock(hidden_dim*2)
        
        # Second residual block
        self.res2 = ResidualBlock(hidden_dim*2, hidden_dim*4)
        self.se2 = SEBlock(hidden_dim*4)
        
        # Self-attention block
        self.attention = SelfAttention(hidden_dim*4)
        
        # Global pooling and feature combination
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim*8, hidden_dim*4),
            nn.LayerNorm(hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels]
        x = x.permute(0, 2, 1)  # [batch_size, channels, sequence_length]
        
        # Initial projection
        x = self.input_proj(x)
        
        # First residual block with SE
        x = self.res1(x)
        x = self.se1(x)
        
        # Second residual block with SE
        x = self.res2(x)
        x = self.se2(x)
        
        # Self-attention
        x = self.attention(x)
        
        # Global pooling
        avg_pool = self.gap(x).squeeze(-1)
        max_pool = self.gmp(x).squeeze(-1)
        
        # Concatenate and project
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.output_proj(x)
        
        return x

# Residual block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        x = F.relu(x)
        return x

# Squeeze-and-Excitation block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# Self-attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(dim, dim, 1)
        self.key = nn.Conv1d(dim, dim, 1)
        self.value = nn.Conv1d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out + x  # Residual connection

# Cross-attention for multimodal fusion
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = torch.bmm(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        return out + x1  # Residual connection

# Define the improved multimodal fusion network
class MultimodalNet(nn.Module):
    def __init__(self, emg_input_dim, eeg_input_dim, hidden_dim, num_classes):
        super(MultimodalNet, self).__init__()
        
        self.emg_encoder = EMGEncoder(emg_input_dim, hidden_dim)
        self.eeg_encoder = EEGEncoder(eeg_input_dim, hidden_dim)
        
        # Cross-attention for multimodal fusion
        self.cross_attention = CrossAttention(hidden_dim*4)
        
        # Improved fusion network with residual connections
        fusion_dim = hidden_dim * 8  # Combined features from both modalities
        
        self.fusion = nn.Sequential(
            ResidualMLP(fusion_dim, hidden_dim * 4),
            nn.Dropout(0.3),
            ResidualMLP(hidden_dim * 4, hidden_dim * 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
    def forward(self, emg, eeg):
        # Extract features from both modalities
        emg_features = self.emg_encoder(emg)
        eeg_features = self.eeg_encoder(eeg)
        
        # Cross-attention between modalities
        emg_attended = self.cross_attention(emg_features, eeg_features)
        eeg_attended = self.cross_attention(eeg_features, emg_features)
        
        # Concatenate attended features
        combined = torch.cat((emg_attended, eeg_attended), dim=1)
        
        # Final classification
        output = self.fusion(combined)
        return output

# Residual MLP block for the fusion network
class ResidualMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.gelu(x)  # Using GELU for better performance
        
        x = self.fc2(x)
        x = self.ln2(x)
        
        x += residual
        x = F.gelu(x)
        return x

# Add noise to data for augmentation
def add_noise(data, noise_factor=0.05):
    noise = torch.randn(data.shape).to(data.device) * noise_factor * torch.std(data)
    return data + noise

# Process and load the data
def load_and_process_data(emg_path, eeg_path, window_size=50, stride=25):
    print("Loading data...")
    
    # Load data
    emg_data = pd.read_csv(emg_path)
    eeg_data = pd.read_csv(eeg_path)
    
    # Extract features and labels
    emg_features = emg_data.iloc[:, :8].values
    eeg_features = eeg_data.iloc[:, :8].values
    
    # Standardize the features
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
        
        # Create windows
        for i in range(0, min_length - window_size, stride):
            emg_windows.append(emg_sample_features[i:i+window_size])
            eeg_windows.append(eeg_sample_features[i:i+window_size])
            window_labels.append(gesture - 1)  # 0-indexed labels
            sample_ids.append(f"{subject}_{repetition}_{gesture}")  # Track which sample this comes from
    
    if len(emg_windows) == 0:
        raise ValueError("No valid windows could be created. Check your data alignment.")
    
    print(f"Created {len(emg_windows)} windows from {len(common_samples)} samples.")
    
    return np.array(emg_windows), np.array(eeg_windows), np.array(window_labels), np.array(sample_ids)

# Training function with memory optimizations, mixed precision, and metrics tracking
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0
    best_f1 = 0.0
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Keep track of metrics
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for i, (emg_inputs, eeg_inputs, labels) in enumerate(train_loader):
            # Move data to device
            emg_inputs, eeg_inputs, labels = emg_inputs.to(device), eeg_inputs.to(device), labels.to(device)
            
            # Data augmentation (add noise) in training only
            if random.random() < 0.5:  # 50% chance to apply noise
                emg_inputs = add_noise(emg_inputs, noise_factor=0.03)
                eeg_inputs = add_noise(eeg_inputs, noise_factor=0.03)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                outputs = model(emg_inputs, eeg_inputs)
                loss = criterion(outputs, labels)
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels for F1 score calculation
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Free up memory
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
        train_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for emg_inputs, eeg_inputs, labels in val_loader:
                # Move data to device
                emg_inputs, eeg_inputs, labels = emg_inputs.to(device), eeg_inputs.to(device), labels.to(device)
                
                # Forward pass (no mixed precision needed for validation)
                outputs = model(emg_inputs, eeg_inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # Free memory
                del outputs, loss
        
        torch.cuda.empty_cache()
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
        val_f1 = f1_score(all_targets, all_preds, average='weighted')
        val_precision = precision_score(all_targets, all_preds, average='weighted')
        val_recall = recall_score(all_targets, all_preds, average='weighted')
        
        # Store metrics
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
        print(f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        
        # Save the best model based on F1 score - SIMPLIFIED to avoid serialization issues
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            
            # SIMPLIFIED: Save only the model state dict
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            
            # Save metrics separately
            metrics = {
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc,
            }
            np.save('best_model_metrics.npy', metrics)
            
            # Save history separately
            np.save('training_history.npy', history)
            print(f'Saved model with F1: {val_f1:.4f}, Accuracy: {val_acc:.2f}%')
    
    return model, history

# Function to evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for emg_inputs, eeg_inputs, labels in test_loader:
            # Move data to device
            emg_inputs, eeg_inputs, labels = emg_inputs.to(device), eeg_inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(emg_inputs, eeg_inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    print("Test Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }

def main():
    # Make sure numpy is imported inside the function 
    import numpy as np
    import torch
    import gc
    from sklearn.model_selection import train_test_split
    
    # Hyperparameters
    window_size = 50
    batch_size = 32  # Adjusted batch size for mixed precision
    learning_rate = 0.001
    num_epochs = 3
    hidden_dim = 64
    
    # Set paths to your data files
    emg_path = 'data/processed/EMG-data.csv'
    eeg_path = 'data/processed/EEG-data.csv'
    
    try:
        # Load and process data
        emg_windows, eeg_windows, window_labels, sample_ids = load_and_process_data(
            emg_path, eeg_path, window_size=window_size
        )
        
        # Calculate number of unique classes
        num_classes = len(np.unique(window_labels))
        print(f"Number of classes: {num_classes}")
        print(f"EMG windows shape: {emg_windows.shape}")
        print(f"EEG windows shape: {eeg_windows.shape}")
        
        # Proper train/val/test split (60/20/20) - stratified by sample_id to prevent data leakage
        unique_samples = np.unique(sample_ids)
        
        samples_train, samples_temp = train_test_split(
            unique_samples, test_size=0.4, random_state=42
        )
        samples_val, samples_test = train_test_split(
            samples_temp, test_size=0.5, random_state=42
        )
        
        train_mask = np.isin(sample_ids, samples_train)
        val_mask = np.isin(sample_ids, samples_val)
        test_mask = np.isin(sample_ids, samples_test)
        
        X_emg_train, X_eeg_train, y_train = emg_windows[train_mask], eeg_windows[train_mask], window_labels[train_mask]
        X_emg_val, X_eeg_val, y_val = emg_windows[val_mask], eeg_windows[val_mask], window_labels[val_mask]
        X_emg_test, X_eeg_test, y_test = emg_windows[test_mask], eeg_windows[test_mask], window_labels[test_mask]
        
        print(f"Training set: {len(X_emg_train)} samples")
        print(f"Validation set: {len(X_emg_val)} samples")
        print(f"Test set: {len(X_emg_test)} samples")
        
        # Free up memory
        del emg_windows, eeg_windows, window_labels, sample_ids
        gc.collect()
        torch.cuda.empty_cache()
        
        # Create datasets
        train_dataset = MultimodalDataset(X_emg_train, X_eeg_train, y_train)
        val_dataset = MultimodalDataset(X_emg_val, X_eeg_val, y_val)
        test_dataset = MultimodalDataset(X_emg_test, X_eeg_test, y_test)
        
        # Free up more memory
        del X_emg_train, X_emg_val, X_emg_test, X_eeg_train, X_eeg_val, X_eeg_test, y_train, y_val, y_test
        gc.collect()
        torch.cuda.empty_cache()
        
        # Create data loaders with modified parameters
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            pin_memory=True, num_workers=0
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=0
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=0
        )

        # Initialize model
        emg_input_dim = 8  # Number of EMG channels
        eeg_input_dim = 8  # Number of EEG channels
        
        model = MultimodalNet(
            emg_input_dim=emg_input_dim,
            eeg_input_dim=eeg_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
        # Define optimizer and loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Define cosine annealing learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/100)
        
        # Train the model
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs
        )
        
        print("Training completed!")
        
        # FIXED: Simplified model loading approach
        try:
            # Only load the state dict, not the full checkpoint
            state_dict = torch.load('best_multimodal_model.pth', map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded model state dictionary")
            
            # Load metrics separately if available
            try:
                metrics = np.load('best_model_metrics.npy', allow_pickle=True).item()
                best_epoch = metrics.get('epoch', 0)
                best_val_f1 = metrics.get('val_f1', 0.0)
                best_val_acc = metrics.get('val_acc', 0.0)
                
                print(f"Loaded best model from epoch {best_epoch+1} with validation F1: {best_val_f1:.4f}, "
                      f"validation accuracy: {best_val_acc:.2f}%")
            except Exception as e:
                print(f"Could not load metrics, but model weights were loaded successfully: {e}")
                best_epoch = 0
                best_val_f1 = 0.0
                best_val_acc = 0.0
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing with the current model state from the last epoch")
            best_epoch = num_epochs - 1
            best_val_f1 = history['val_f1'][-1] if history['val_f1'] else 0.0
            best_val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
        
        # Evaluate the model on test set
        test_results = evaluate_model(model, test_loader)
        
        print(f"Test F1 Score: {test_results['f1']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
        
        # Save final results
        final_results = {
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'best_val_acc': best_val_acc,
            'test_results': test_results,
            'history': history
        }
        
        # Save as numpy file for later analysis
        np.save('model_results.npy', final_results)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()