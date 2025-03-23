#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG-Based Gesture Classification Using CNN, LSTM, and Hybrid CNN-LSTM Models with PyTorch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper Functions
def print_color(text: str, color: str) -> None:
    """
    Prints text in specified ANSI color for better readability.
    
    Args:
        text (str): Text to be printed
        color (str): Color name ('red', 'green', 'yellow', 'blue', 'magenta')
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m"
    }
    print(f"{colors.get(color, '')}{text}\033[0m")


def plot_eeg_channels(data: np.ndarray, title: str = "EEG Signals", 
                     sample_idx: int = 0, channels: int = 8) -> None:
    """
    Plot EEG channels from a window of data.
    
    Args:
        data (np.ndarray): EEG data array with shape (windows, time_steps, channels)
        title (str): Title for the plot
        sample_idx (int): Index of the window to plot
        channels (int): Number of EEG channels
    """
    plt.figure(figsize=(12, 16))
    window_data = data[sample_idx]
    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.plot(window_data[:, i])
        plt.title(f'{title}, Channel {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()


# Data Preparation
def prepare_eeg_data():
    """
    Load and prepare EEG data for training models.
    
    Returns:
        Tuple: Processed data tensors ready for model training
    """
    # Load the dataset from a CSV file
    df = pd.read_csv("EEG-data.csv")
    
    # Adjust gesture labels to be zero-indexed
    df['gesture'] = df['gesture'] - 1
    
    # Count number of unique gestures
    num_classes = df['gesture'].nunique()
    
    # Display initial data and dataset structure
    print_color("Head of DataFrame:", "green")
    print(df.head())
    print_color("Shape of DataFrame:", "green")
    print(df.shape)
    
    # Check and display any null values in the dataset
    null_count = df.isnull().sum()
    print_color("Null values in each column:", "yellow")
    print(null_count)
    
    # List and display unique gestures and subjects
    print_color("Unique gestures (after adjustment to 0-index):", "blue")
    print(sorted(df["gesture"].unique()))
    print_color("Unique subjects:", "blue")
    print(sorted(df["subject"].unique()))
    
    # Windowing Setup
    # Set window size and step size for slicing the data
    WINDOW_SIZE = 100  # Number of samples per window
    STEP_SIZE = 50     # Interval at which new windows are created
    
    # Initialize lists to store windowed data and corresponding labels
    X_list = []
    y_list = []
    
    # Group the data by gesture, extracting features for each gesture
    for gesture_id in sorted(df["gesture"].unique()):
        gesture_df = df[df["gesture"] == gesture_id]
        gesture_data = gesture_df[
            ["Channel_1", "Channel_2", "Channel_3", "Channel_4",
             "Channel_5", "Channel_6", "Channel_7", "Channel_8"]
        ].values  # Extract channel data as numpy array
    
        # Generate overlapping windows of data
        for start_idx in range(0, len(gesture_data) - WINDOW_SIZE + 1, STEP_SIZE):
            window_data = gesture_data[start_idx:start_idx + WINDOW_SIZE]
            X_list.append(window_data)
            y_list.append(gesture_id)
    
    # Convert lists to numpy arrays
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    
    # Save arrays for future use
    os.makedirs('data/processed', exist_ok=True)
    np.save("data/processed/X_eeg.npy", X_array)
    np.save("data/processed/y_eeg.npy", y_array)
    
    # Display shapes of the prepared datasets
    print_color("Shape of X_array:", "red")
    print(X_array.shape)
    print_color("Shape of y_array:", "red")
    print(y_array.shape)
    
    # Print statistics about the data
    print_color("Data statistics in X_array:", "green")
    print("Mean:", np.mean(X_array, axis=(0, 1)))
    print("Standard Deviation:", np.std(X_array, axis=(0, 1)))
    print("Max value:", np.max(X_array))
    print("Min value:", np.min(X_array))
    
    # Normalize the data
    X_mean = np.mean(X_array)
    X_std = np.std(X_array)
    X_array = (X_array - X_mean) / X_std
    
    # Convert to PyTorch tensors and move to device
    X_tensor = torch.FloatTensor(X_array).to(device)
    y_tensor = torch.LongTensor(y_array).to(device)
    
    # Print statistics about the tensor data
    print_color("Shape of X_tensor:", "red")
    print(X_tensor.shape)
    print_color("Shape of y_tensor:", "red")
    print(y_tensor.shape)
    print_color("Data statistics in X_tensor:", "green")
    print(f"Mean: {X_tensor.mean().item():.4f}")
    print(f"Standard Deviation: {X_tensor.std().item():.4f}")
    print(f"Max value: {X_tensor.max().item():.4f}")
    print(f"Min value: {X_tensor.min().item():.4f}")
    
    return X_tensor, y_tensor, num_classes


# Model Definitions
class CNNModel(nn.Module):
    """
    CNN model for EEG classification.
    
    Structure:
    - 1D Convolutional layers for feature extraction
    - Batch Normalization for stable learning
    - Max Pooling for dimension reduction
    - Dropout for regularization
    - Fully connected layers for classification
    """
    def __init__(self, num_classes: int = 7, input_channels: int = 8):
        super(CNNModel, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the output size after the three pooling layers
        # Input size: (batch_size, 8, 100)
        # After 3 pooling layers with stride 2: 100 -> 50 -> 25 -> 12
        self.flatten_size = 128 * 12
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, time_steps, channels)
        # Reshape for 1D convolution (batch_size, channels, time_steps)
        x = x.permute(0, 2, 1)
        
        # Convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


class LSTMModel(nn.Module):
    """
    LSTM model for EEG classification.
    
    Structure:
    - LSTM layers for sequence modeling
    - Dropout for regularization
    - Fully connected layer for classification
    """
    def __init__(self, num_classes: int = 7, input_channels: int = 8, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, time_steps, channels)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # We use the output from the last time step for classification
        # For bidirectional, concatenated output
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout before final layer
        lstm_out = self.dropout(lstm_out)
        
        # Final classification layer
        out = self.fc(lstm_out)
        
        return out


class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for EEG classification.
    
    Structure:
    - 1D Convolutional layers for feature extraction
    - LSTM layers for sequence modeling
    - Fully connected layer for classification
    """
    def __init__(self, num_classes: int = 7, input_channels: int = 8, hidden_size: int = 128):
        super(CNNLSTMModel, self).__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after CNN
        self.cnn_output_size = 64  # Number of channels after CNN
        self.seq_length_after_cnn = 25  # 100 -> 50 -> 25 after max pooling
        
        # LSTM sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, time_steps, channels)
        # Reshape for 1D convolution (batch_size, channels, time_steps)
        x = x.permute(0, 2, 1)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Reshape for LSTM (batch_size, seq_length, features)
        x = x.permute(0, 2, 1)
        
        # LSTM sequence modeling
        lstm_out, _ = self.lstm(x)
        
        # Use output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Final classification
        out = self.fc(lstm_out)
        
        return out


# Training and Evaluation Functions
def train_model(model: nn.Module, train_loader: DataLoader, 
               criterion: nn.Module, optimizer: torch.optim.Optimizer, 
               device: torch.device, epochs: int = 10) -> List[float]:
    """
    Train a model using the provided data loader.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating model weights
        device: Device to perform computations on (CPU/GPU)
        epochs: Number of training epochs
    
    Returns:
        List of training loss values per epoch
    """
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss/total, 
                'acc': 100.*correct/total
            })
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to perform computations on (CPU/GPU)
    
    Returns:
        Tuple of (test loss, accuracy, confusion matrix)
    """
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Collect for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / total
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_targets, all_preds))
    
    return test_loss, accuracy, conf_matrix


def plot_training_results(losses: List[float], title: str = "Training Loss") -> None:
    """
    Plot training results.
    
    Args:
        losses: List of loss values per epoch
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
    """
    if class_names is None:
        class_names = [f'Gesture {i}' for i in range(conf_matrix.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def main():
    # Data preparation
    print_color("Preparing EEG data...", "blue")
    X_tensor, y_tensor, num_classes = prepare_eeg_data()
    
    # Split data into train, validation, test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=RANDOM_SEED, stratify=y_tensor.cpu()
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train.cpu()
    )
    
    print_color(f"Train set: {X_train.shape[0]} samples", "green")
    print_color(f"Validation set: {X_val.shape[0]} samples", "green")
    print_color(f"Test set: {X_test.shape[0]} samples", "green")
    
    # Create data loaders
    BATCH_SIZE = 64
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Training settings
    LEARNING_RATE = 0.001
    EPOCHS = 20
    
    # Loss function with class weights if needed
    criterion = nn.CrossEntropyLoss()
    
    # Create directory for models
    os.makedirs("models", exist_ok=True)
    
    # Train CNN model
    print_color("\nTraining CNN Model...", "magenta")
    cnn_model = CNNModel(num_classes=num_classes).to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    cnn_losses = train_model(cnn_model, train_loader, criterion, cnn_optimizer, device, EPOCHS)
    cnn_train_time = time.time() - start_time
    
    print(f"CNN training completed in {cnn_train_time:.2f} seconds")
    plot_training_results(cnn_losses, "CNN Training Loss")
    
    # Evaluate CNN model
    print_color("\nEvaluating CNN Model...", "magenta")
    cnn_test_loss, cnn_accuracy, cnn_conf_matrix = evaluate_model(cnn_model, test_loader, criterion, device)
    
    # Save CNN model
    torch.save(cnn_model.state_dict(), "models/cnn_eeg_model.pth")
    
    # Train LSTM model
    print_color("\nTraining LSTM Model...", "magenta")
    lstm_model = LSTMModel(num_classes=num_classes).to(device)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    lstm_losses = train_model(lstm_model, train_loader, criterion, lstm_optimizer, device, EPOCHS)
    lstm_train_time = time.time() - start_time
    
    print(f"LSTM training completed in {lstm_train_time:.2f} seconds")
    plot_training_results(lstm_losses, "LSTM Training Loss")
    
    # Evaluate LSTM model
    print_color("\nEvaluating LSTM Model...", "magenta")
    lstm_test_loss, lstm_accuracy, lstm_conf_matrix = evaluate_model(lstm_model, test_loader, criterion, device)
    
    # Save LSTM model
    torch.save(lstm_model.state_dict(), "models/lstm_eeg_model.pth")
    
    # Train CNN-LSTM hybrid model
    print_color("\nTraining CNN-LSTM Hybrid Model...", "magenta")
    cnn_lstm_model = CNNLSTMModel(num_classes=num_classes).to(device)
    cnn_lstm_optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    cnn_lstm_losses = train_model(cnn_lstm_model, train_loader, criterion, cnn_lstm_optimizer, device, EPOCHS)
    cnn_lstm_train_time = time.time() - start_time
    
    print(f"CNN-LSTM training completed in {cnn_lstm_train_time:.2f} seconds")
    plot_training_results(cnn_lstm_losses, "CNN-LSTM Training Loss")
    
    # Evaluate CNN-LSTM model
    print_color("\nEvaluating CNN-LSTM Hybrid Model...", "magenta")
    cnn_lstm_test_loss, cnn_lstm_accuracy, cnn_lstm_conf_matrix = evaluate_model(cnn_lstm_model, test_loader, criterion, device)
    
    # Save CNN-LSTM model
    torch.save(cnn_lstm_model.state_dict(), "models/cnn_lstm_eeg_model.pth")
    
    # Compare model performances
    print_color("\nModel Performance Comparison:", "blue")
    models = ["CNN", "LSTM", "CNN-LSTM"]
    accuracies = [cnn_accuracy, lstm_accuracy, cnn_lstm_accuracy]
    train_times = [cnn_train_time, lstm_train_time, cnn_lstm_train_time]
    
    for model, acc, time_taken in zip(models, accuracies, train_times):
        print(f"{model}: Accuracy = {acc:.2f}%, Training Time = {time_taken:.2f} seconds")
    
    # Plot confusion matrices
    plot_confusion_matrix(cnn_conf_matrix)
    plot_confusion_matrix(lstm_conf_matrix)
    plot_confusion_matrix(cnn_lstm_conf_matrix)


if __name__ == "__main__":
    main() 