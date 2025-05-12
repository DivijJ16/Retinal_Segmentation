import numpy as np
import heapq
import torch
import matplotlib.pyplot as plt
import os
from EyeHandler import EyeFileHandler
from TEST import Final_U_NET, replace_nan_with_local_avg_2d

# Import functions and classes from the TEST.py file

# Custom RetinalDataset class with NaN handling
class RetinalDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, label_paths, slice_width=64, use_slicing=True):
        """
        Dataset class that can handle both full images and sliced images
        with proper NaN handling
        
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
        
        # Create binary mask - with NaN handling
        width = self.slice_width if self.use_slicing else 1024
        mask = torch.zeros((1, 496, width), dtype=torch.float32)
        
        for x in range(width):
            # Handle NaN values by skipping them
            if x < len(label) and not np.isnan(label[x]):
                y = int(label[x])
                if 0 <= y < 496:
                    mask[0, y, x] = 1.0

        return image, mask

def train_model(model, train_loader, val_loader, num_epochs, device, model_base_path, start_epoch=0):
    initial_lr = 0.1
    
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Lists to store loss history
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Learning rate scheduler logic
        if epoch > 0 and epoch % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print(f"Epoch {epoch}: Learning rate updated to {param_group['lr']}")
        
        # Training phase
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
                print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        # Calculate average validation loss for this epoch
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model checkpoint after each epoch
        checkpoint_path = f"{model_base_path}_epoch_{epoch+1}.pth"
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_loss_history': train_losses,
            'val_loss_history': val_losses
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")
        
        # Also save as latest model for easy resume
        latest_path = f"{model_base_path}_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Plot training and validation loss
        if (epoch + 1) % 5 == 0 or epoch == start_epoch + num_epochs - 1:  # Plot every 5 epochs and at the end
            plt.figure(figsize=(10, 6))
            epochs_range = list(range(start_epoch + 1, epoch + 2))
            plt.plot(epochs_range, train_losses, label='Training Loss')
            plt.plot(epochs_range, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(f"{model_base_path}_loss_plot_epoch_{epoch+1}.png")
            plt.close()
    
    return model, train_losses, val_losses


def weight_assigner_from_probmap(prob_map, W_min=1e-5):
    """
    Convert probability map to edge weights for shortest path.
    Higher probability = lower weight (more likely to be part of the path)
    
    Parameters:
    - prob_map: Probability map from the UNet model (height x width)
    - W_min: Minimum weight to ensure no zero weights
    
    Returns:
    - weights: 3D array of shape (height, width, 8) containing edge weights
    """
    height, width = prob_map.shape
    weights = np.zeros((height, width, 8))
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    # Invert probabilities: high probability = low weight
    inv_probs = 1 - prob_map + W_min
    
    for y in range(height):
        for x in range(width):
            for i in range(8):
                ny, nx = y + dy[i], x + dx[i]
                if 0 <= ny < height and 0 <= nx < width:
                    # Average the inverted probabilities of current and neighbor pixels
                    weights[y, x, i] = (inv_probs[y, x] + inv_probs[ny, nx]) / 2
    return weights

def shortest_path(weights, start=None, end=None):
    """
    Implementation of Dijkstra's algorithm for finding shortest path.
    
    Parameters:
    - weights: 3D array of shape (height, width, 8) representing edge weights
    - start: Starting point (y, x). If None, uses (0, 0)
    - end: Ending point (y, x). If None, uses (height-1, width-1)
    
    Returns:
    - path: List of (y, x) coordinates representing the shortest path
    - distances: 2D array of distances from start to each pixel
    """
    height, width, _ = weights.shape
    
    # Default start and end points
    if start is None:
        start = (0, 0)
    if end is None:
        end = (height - 1, width - 1)
    
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    distances = np.full((height, width), np.inf)
    predecessors = np.empty((height, width), dtype=object)
    visited = np.full((height, width), False)
    
    heap = []
    distances[start] = 0
    heapq.heappush(heap, (0, start))
    
    while heap:
        current_dist, (y, x) = heapq.heappop(heap)
        
        if visited[y, x]:
            continue
        visited[y, x] = True

        if (y, x) == end:
            break
        
        for i, (dy, dx) in enumerate(neighbors):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if visited[ny, nx]:
                    continue
                weight = weights[y, x, i]
                new_dist = current_dist + weight
                if new_dist < distances[ny, nx]:
                    distances[ny, nx] = new_dist
                    predecessors[ny, nx] = (y, x)
                    heapq.heappush(heap, (new_dist, (ny, nx)))
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current[0], current[1]]
    return path[::-1], distances

def find_layer_across_width(prob_map, start_col=0):
    """
    Find layer coordinates by running shortest path across the width of the image
    
    Parameters:
    - prob_map: Probability map from UNet (height x width)
    - start_col: Starting column index
    
    Returns:
    - y_coords: Array of y-coordinates representing the detected layer
    """
    height, width = prob_map.shape
    weights = weight_assigner_from_probmap(prob_map)
    y_coords = np.zeros(width, dtype=int)
    
    # Find the starting point (highest probability in the first column)
    start_y = np.argmax(prob_map[:, start_col])
    start = (start_y, start_col)
    
    # For each column, find the end point and compute shortest path
    for end_col in range(start_col + 1, width):
        # Find likely endpoint in current column
        end_y = np.argmax(prob_map[:, end_col])
        end = (end_y, end_col)
        
        # Find shortest path between current start and end
        path, _ = shortest_path(weights, start=start, end=end)
        
        # Extract y-coordinates for all columns in this segment
        for y, x in path:
            if start_col <= x < end_col:
                y_coords[x] = y
                
        y_coords = cubic_spline_smoothing(y_coords)
        
        # Update start for next segment
        start = end
    
    return y_coords

def preprocess_annotations(annotations):
    """
    Preprocess annotation data to handle NaN values
    
    Parameters:
    - annotations: List of annotation arrays which may contain NaNs
    
    Returns:
    - clean_annotations: List of annotation arrays with NaNs replaced by interpolated values
    """
    clean_annotations = []
    
    for annotation in annotations:
        # Create a 2D array with the annotation as a single row
        annotation_2d = annotation.reshape(1, -1)
        
        # Apply the replace_nan_with_local_avg_2d function
        clean_annotation = replace_nan_with_local_avg_2d(annotation_2d)[0]
        
        # If any NaNs still remain (e.g., at edges), replace with nearest non-NaN value
        nan_mask = np.isnan(clean_annotation)
        if np.any(nan_mask):
            # Get indices of NaN values
            nan_indices = np.where(nan_mask)[0]
            
            # Get indices of non-NaN values
            valid_indices = np.where(~nan_mask)[0]
            
            if len(valid_indices) > 0:
                # For each NaN value, find the nearest non-NaN value
                for idx in nan_indices:
                    # Find nearest non-NaN index
                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                    clean_annotation[idx] = clean_annotation[nearest_idx]
            else:
                # If all values are NaN, replace with a default value (e.g., middle of image)
                clean_annotation[:] = 248  # Middle of 496 height
                
        clean_annotations.append(clean_annotation)
    
    return clean_annotations

def extract_layer_data(handler, target_layer, layer_names):
    """
    Extract layer data with handling for different layer arrangements
    
    Parameters:
    - handler: EyeFileHandler instance
    - target_layer: Name of target layer (e.g., 'EZ')
    - layer_names: List of available layer names
    
    Returns:
    - annotations: List of layer annotations for the target layer
    """
    annotations = []
    num_bscans = handler.raw_volume.shape[0]
    
    # Check if target layer exists in the available layers
    if target_layer not in layer_names:
        print(f"Warning: Layer '{target_layer}' not found in {handler.file_path}")
        # Return empty annotations for all B-scans
        return [np.full(handler.raw_volume.shape[2], np.nan) for _ in range(num_bscans)]
    
    # Get the index of the target layer
    target_layer_index = layer_names.index(target_layer)
    
    # Try different possible arrangements of layer data
    for i in range(num_bscans):
        try:
            # First possible arrangement: [layer_index, bscan_index, width]
            annotations.append(handler.layer_heights[target_layer_index, i, :])
        except IndexError:
            try:
                # Second possible arrangement: [bscan_index, layer_index, width]
                annotations.append(handler.layer_heights[i, target_layer_index, :])
            except IndexError:
                print(f"Error: Could not extract layer heights for scan {i} in {handler.file_path}")
                annotations.append(np.full(handler.raw_volume.shape[2], np.nan))
    
    return annotations

def cubic_spline_smoothing(y_coords, smoothing_factor=0.01):
    """
    Apply cubic spline smoothing to the segmented layer
    
    Parameters:
    - y_coords: Array of y-coordinates for the segmented layer
    - smoothing_factor: Smoothing factor for the spline (lower = smoother)
    
    Returns:
    - smoothed_coords: Smoothed array of y-coordinates
    """
    from scipy.interpolate import UnivariateSpline
    
    # Create x-coordinates
    x_coords = np.arange(len(y_coords))
    
    # Remove any NaN values
    valid_indices = ~np.isnan(y_coords)
    if np.sum(valid_indices) < 4:  # Need at least 4 points for cubic spline
        return y_coords
    
    x_valid = x_coords[valid_indices]
    y_valid = y_coords[valid_indices]
    
    # Apply spline fitting
    spline = UnivariateSpline(x_valid, y_valid, s=smoothing_factor * len(y_valid))
    smoothed_coords = spline(x_coords)
    
    # Ensure coordinates are integers
    smoothed_coords = np.round(smoothed_coords).astype(int)
    
    return smoothed_coords

def main():
    # Directory containing eye files
    eye_files_dir = '.EYE Files'  # Change this to your directory path
    target_layer = 'RPE'  # Target layer to predict
    
    # Define model base path for checkpoints
    model_base_path = f"unet_{target_layer}"
    
    # Get all .eye files in the directory
    eye_files = [f for f in os.listdir(eye_files_dir) if f.endswith('.eye')]
    print(f"Found {len(eye_files)} eye files")
    
    # Lists to store data from all files
    all_bscans = []
    all_annotations = []
    file_indices = []  # To keep track of which file each B-scan comes from
    
    # Process each eye file
    for file_idx, eye_file in enumerate(eye_files):
        file_path = os.path.join(eye_files_dir, eye_file)
        print(f"Processing {file_path}...")
        
        try:
            handler = EyeFileHandler(file_path)
            handler.load_raw_volume()
            handler.load_layer_data()
            
            layer_names = handler.layer_names()
            print(f"  Available layers: {layer_names}")
            
            # Extract B-scans
            bscans = [handler.raw_volume[i] for i in range(handler.raw_volume.shape[0])]
            
            # Extract annotations for the target layer
            annotations = extract_layer_data(handler, target_layer, layer_names)
            
            # Add data to global lists
            all_bscans.extend(bscans)
            all_annotations.extend(annotations)
            file_indices.extend([file_idx] * len(bscans))
            
            print(f"  Added {len(bscans)} B-scans from {eye_file}")
            
        except Exception as e:
            print(f"Error processing {eye_file}: {str(e)}")
    
    # Preprocess annotations to handle NaN values
    print("Preprocessing annotations to handle NaN values...")
    clean_annotations = preprocess_annotations(all_annotations)
    
    # Print information about NaN values
    total_nan = 0
    for i, (ann, clean_ann) in enumerate(zip(all_annotations, clean_annotations)):
        nan_count = np.sum(np.isnan(ann))
        if nan_count > 0:
            total_nan += nan_count
            file_idx = file_indices[i]
            print(f"B-scan from file {eye_files[file_idx]}: {nan_count} NaN values found and replaced")
    
    print(f"Total NaN values handled: {total_nan}")
    
    # Create train/validation split (80/20) - ensuring we split by files, not by B-scans
    unique_file_indices = list(set(file_indices))
    num_files = len(unique_file_indices)
    train_size = int(0.8 * num_files)
    
    # Randomly sample file indices for training
    import random
    random.seed(42)  # For reproducibility
    train_file_indices = random.sample(unique_file_indices, train_size)
    
    # Separate train and validation data
    train_bscans = []
    train_annotations = []
    val_bscans = []
    val_annotations = []
    
    for i, (bscan, annotation) in enumerate(zip(all_bscans, clean_annotations)):
        file_idx = file_indices[i]
        if file_idx in train_file_indices:
            train_bscans.append(bscan)
            train_annotations.append(annotation)
        else:
            val_bscans.append(bscan)
            val_annotations.append(annotation)
    
    print(f"Training data: {len(train_bscans)} B-scans from {train_size} files")
    print(f"Validation data: {len(val_bscans)} B-scans from {num_files - train_size} files")
    
    # Configure model training settings
    use_slicing = True
    slice_width = 64
    batch_size = 8
    num_epochs = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = UNet(in_channels=1, num_classes=1).to(device)
    
    # Create datasets and data loaders
    train_dataset = RetinalDataset(train_bscans, train_annotations, slice_width, use_slicing)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    # Create validation dataset for loss tracking
    val_dataset = RetinalDataset(val_bscans, val_annotations, slice_width, use_slicing)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Check if we have a checkpoint to resume from
    latest_checkpoint_path = f"{model_base_path}_latest.pth"
    start_epoch = 0
    train_loss_history = []
    val_loss_history = []
    
    try:
        # Try to load the latest checkpoint
        print(f"Looking for latest checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Initialize with default LR
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        
        # Load loss history if available
        if 'train_loss_history' in checkpoint and 'val_loss_history' in checkpoint:
            train_loss_history = checkpoint['train_loss_history']
            val_loss_history = checkpoint['val_loss_history']
            print(f"Loaded loss history of {len(train_loss_history)} epochs")
            
            # Plot existing loss history
            plt.figure(figsize=(10, 6))
            epochs_range = list(range(1, len(train_loss_history) + 1))
            plt.plot(epochs_range, train_loss_history, label='Training Loss')
            plt.plot(epochs_range, val_loss_history, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss (Resumed)')
            plt.legend()
            plt.savefig(f"{model_base_path}_loss_plot_resumed.png")
            plt.show()
            
        print(f"Resuming training from epoch {start_epoch}")
        
    except FileNotFoundError:
        print("No checkpoint found. Starting training from scratch.")
    
    # Check if we're continuing training or need to train a new model
    remaining_epochs = num_epochs - start_epoch
    
    if remaining_epochs > 0:
        print(f"Training for {remaining_epochs} more epochs...")
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, remaining_epochs, device, 
            model_base_path, start_epoch
        )
        
        # Extend loss history with new values
        train_loss_history.extend(train_losses)
        val_loss_history.extend(val_losses)
    else:
        print(f"Model already trained for {start_epoch} epochs, which is >= {num_epochs}.")
        print("Skipping training. Set num_epochs higher if you want to train more.")
    
    # Save final model with proper name for compatibility with rest of the code
    final_model_path = f"{model_base_path}_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")
    
    # Plot final loss curves
    plt.figure(figsize=(10, 6))
    epochs_range = list(range(1, len(train_loss_history) + 1))
    plt.plot(epochs_range, train_loss_history, label='Training Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{model_base_path}_final_loss_plot.png")
    plt.show()
    
    # Create unsliced validation dataset for evaluation
    test_dataset = RetinalDataset(val_bscans, val_annotations, slice_width, use_slicing=False)
    
    # Test on validation set
    model.eval()
    mae_unet_total = 0
    mae_combined_total = 0
    valid_count = 0
    
    for val_idx in range(len(val_bscans)):
        # Get test image and ground truth
        test_image = val_bscans[val_idx]
        ground_truth = val_annotations[val_idx]
        
        # Skip B-scans with all NaN ground truth
        if np.all(np.isnan(ground_truth)):
            continue
        
        # Method 1: Direct layer prediction using UNet
        with torch.no_grad():
            image_tensor = torch.tensor(test_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            prob_map_raw = model(image_tensor)
            prob_map = torch.sigmoid(prob_map_raw).cpu().numpy()[0, 0]
            predicted_layer_unet = np.argmax(prob_map, axis=0)
        
        # Method 2: Generate probability map from UNet, then use shortest path
        predicted_layer_combined = find_layer_across_width(prob_map)
        
        # Calculate accuracy metrics
        mae_unet = np.mean(np.abs(ground_truth - predicted_layer_unet))
        mae_combined = np.mean(np.abs(ground_truth - predicted_layer_combined))
        
        mae_unet_total += mae_unet
        mae_combined_total += mae_combined
        valid_count += 1
        
        # Only visualize the first few validation samples to avoid clutter
        if val_idx < 3:  # Show results for first 3 validation samples
            plt.figure(figsize=(15, 10))
            
            # Original B-scan with ground truth
            plt.subplot(3, 1, 1)
            plt.imshow(test_image, cmap='gray', aspect='auto')
            plt.plot(np.arange(len(ground_truth)), ground_truth, color='cyan', linewidth=1.5, label="Ground Truth")
            plt.title("B-scan with Ground Truth")
            plt.legend()
            
            # UNet prediction
            plt.subplot(3, 1, 2)
            plt.imshow(test_image, cmap='gray', aspect='auto')
            plt.plot(np.arange(len(predicted_layer_unet)), predicted_layer_unet, color='red', linewidth=1.5, label="UNet Prediction")
            plt.title("B-scan with UNet Prediction")
            plt.legend()
            
            # Combined UNet + Shortest Path
            plt.subplot(3, 1, 3)
            plt.imshow(test_image, cmap='gray', aspect='auto')
            plt.plot(np.arange(len(predicted_layer_combined)), predicted_layer_combined, color='green', linewidth=1.5, label="UNet + Shortest Path")
            plt.title("B-scan with UNet + Shortest Path")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{model_base_path}_prediction_sample_{val_idx}.png")
            plt.show()
            
            # Visualize probability map
            plt.figure(figsize=(10, 5))
            plt.imshow(prob_map, cmap='viridis', aspect='auto')
            plt.colorbar(label='Probability')
            plt.title("Probability Map from UNet")
            plt.savefig(f"{model_base_path}_probmap_sample_{val_idx}.png")
            plt.show()
    
    # Calculate average performance across validation set
    if valid_count > 0:
        avg_mae_unet = mae_unet_total / valid_count
        avg_mae_combined = mae_combined_total / valid_count
        
        print(f"Average UNet Mean Absolute Error: {avg_mae_unet:.2f} pixels")
        print(f"Average Combined Method Mean Absolute Error: {avg_mae_combined:.2f} pixels")
        
        improvement = avg_mae_unet - avg_mae_combined
        percent_improvement = (improvement / avg_mae_unet * 100) if avg_mae_unet > 0 else 0
        print(f"Improvement: {improvement:.2f} pixels ({percent_improvement:.2f}%)")
        
        # Save final evaluation metrics
        with open(f"{model_base_path}_evaluation_metrics.txt", "w") as f:
            f.write(f"Average UNet Mean Absolute Error: {avg_mae_unet:.2f} pixels\n")
            f.write(f"Average Combined Method Mean Absolute Error: {avg_mae_combined:.2f} pixels\n")
            f.write(f"Improvement: {improvement:.2f} pixels ({percent_improvement:.2f}%)\n")
    else:
        print("No valid validation samples found.")

if __name__ == "__main__":
    main()