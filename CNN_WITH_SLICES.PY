from EyeHandler import EyeFileHandler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

def plot_bscan(image_array):
    plt.figure(figsize=(10, 5))
    plt.imshow(image_array, cmap='gray', aspect='auto')
    plt.show()

def plot_bscan_with_layers(image_array, y_coordinates):
    plt.figure(figsize=(10, 5))
    plt.imshow(image_array, cmap='gray', aspect='auto')
    x_values = np.arange(1024)
    plt.plot(x_values, y_coordinates, color='red', linewidth=2)
    plt.show()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 64)
        self.down2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(64, 64)
        self.down3 = nn.MaxPool2d(2)
        
        # Bottom
        self.bottom = DoubleConv(64, 64)

        # Decoder
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        d1 = self.down1(c1)
        
        c2 = self.conv2(d1)
        d2 = self.down2(c2)
        
        c3 = self.conv3(d2)
        d3 = self.down3(c3)
        
        # Bottom
        bottom = self.bottom(d3)
        
        # Decoder
        u3 = self.up3(bottom)
        u3 = torch.cat([u3, c3], dim=1)
        c6 = self.conv6(u3)
        
        u2 = self.up2(c6)
        u2 = torch.cat([u2, c2], dim=1)
        c7 = self.conv7(u2)
        
        u1 = self.up1(c7)
        u1 = torch.cat([u1, c1], dim=1)
        c8 = self.conv8(u1)
        
        output = self.out(c8)
        
        return output

class RetinalDataset(Dataset):
    def __init__(self, image_paths, label_paths, slice_width=64, use_slicing=True):
        """
        Dataset class that can handle both full images and sliced images
        
        Parameters:
        - image_paths: List of image arrays
        - label_paths: List of label arrays
        - slice_width: Width of each slice (default: 64)
        - use_slicing: Whether to use image slicing (default: True)
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.slice_width = slice_width
        self.use_slicing = use_slicing
        self.num_slices = 1024 // slice_width if use_slicing else 1

    def __len__(self):
        return len(self.image_paths) * self.num_slices

    def __getitem__(self, idx):
        image_idx = idx // self.num_slices
        slice_idx = idx % self.num_slices
        
        image = self.image_paths[image_idx]
        label = self.label_paths[image_idx]
        
        if self.use_slicing:
            # Extract slice
            start_col = slice_idx * self.slice_width
            end_col = start_col + self.slice_width
            image = image[:, start_col:end_col]
            label = label[start_col:end_col]
        
        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        # Create binary mask
        width = self.slice_width if self.use_slicing else 1024
        mask = torch.zeros((1, 496, width), dtype=torch.float32)
        for x in range(width):
            y = int(label[x])
            if 0 <= y < 496:
                mask[0, y, x] = 1.0

        return image, mask

def get_y_coordinates(prob_map):
    """Convert probability map to y-coordinates"""
    prob_map = torch.sigmoid(prob_map).cpu().numpy()[0, 0]
    y_coords = np.argmax(prob_map, axis=0)
    return y_coords

def predict_layer(model, image, device, use_slicing=True, slice_width=64):
    """
    Predict layer coordinates using either sliced or full image processing
    """
    model.eval()
    
    if use_slicing:
        num_slices = image.shape[1] // slice_width
        predictions = []
        
        with torch.no_grad():
            for i in range(num_slices):
                start_col = i * slice_width
                end_col = start_col + slice_width
                image_slice = image[:, start_col:end_col]
                
                slice_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                prob_map = model(slice_tensor)
                pred_y_coords = get_y_coordinates(prob_map)
                
                predictions.append(pred_y_coords)
        
        return np.concatenate(predictions)
    else:
        with torch.no_grad():
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            prob_map = model(image_tensor)
            return get_y_coordinates(prob_map)

def train_model(model, train_loader, num_epochs, device):
    """Train the model"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}")

def replace_nan_with_local_avg_2d(arr):
    """
    Replace NaN values in a 2D numpy array with the average of their nearest 2 non-NaN neighbors.
    For edge NaN values, use the closest non-NaN value.
    Processes each column independently.
    
    Parameters:
    arr (numpy.ndarray): Input 2D array of shape (m,n) containing NaN values
    
    Returns:
    numpy.ndarray: Array with NaN values replaced by local averages
    """
    # Make a copy to avoid modifying the original array
    result = arr.copy()
    
    # Process each column independently
    for col in range(result.shape[1]):
        column = result[:, col]
        nan_indices = np.where(np.isnan(column))[0]
        
        for idx in nan_indices:
            # Find nearest non-NaN values before and after the current position
            left_idx = right_idx = idx
            
            # Search up
            while left_idx >= 0 and np.isnan(column[left_idx]):
                left_idx -= 1
                
            # Search down
            while right_idx < len(column) and np.isnan(column[right_idx]):
                right_idx += 1
                
            # Calculate replacement value based on available neighbors
            if left_idx >= 0 and right_idx < len(column):
                # Both neighbors available - use average
                result[idx, col] = (column[left_idx] + column[right_idx]) / 2
            elif left_idx >= 0:
                # Only upper neighbor available - use that value
                result[idx, col] = column[left_idx]
            elif right_idx < len(column):
                # Only lower neighbor available - use that value
                result[idx, col] = column[right_idx]
            else:
                # If no neighbors available (should never happen as long as column has at least one non-NaN value)
                result[idx, col] = np.nan                
    return result



def main():
    # Load data
    handler = EyeFileHandler('01-026_Baseline_OD.eye')
    layer_height_data = np.load('annotations/layers/layer_heights.npy')
    pixel_map_data = np.load('raw_volume.npy')
    temp = replace_nan_with_local_avg_2d(layer_height_data[1])
    layer_height_data[1] = temp
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create datasets with slicing
    train_dataset = RetinalDataset(pixel_map_data, layer_height_data[1], use_slicing=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Initialize model
    model = UNet(in_channels=1).to(device)
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, num_epochs=2, device=device)
    print(np.sum(np.isnan(layer_height_data[1][3])))
    print("Training completed!")
    
    # Make predictions
    print("Making predictions...")
    test_image = pixel_map_data[9]
    
    # Predict using slicing
    predicted_coords_sliced = predict_layer(model, test_image, device, use_slicing=True)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot original image with predicted layer
    plt.subplot(2, 1, 1)
    plt.imshow(test_image, cmap="gray")
    plt.plot(np.arange(1024), predicted_coords_sliced, color="red", linewidth=2)
    plt.title("Predicted Layer (Using Slicing)")
    
    # Plot ground truth for comparison
    plt.subplot(2, 1, 2)
    plt.imshow(test_image, cmap="gray")
    plt.plot(np.arange(1024), layer_height_data[0][0], color="green", linewidth=2)
    plt.title("Ground Truth")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
