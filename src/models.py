"""
NFL Big Data Bowl 2026 - Model Architectures and Training
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using baseline models only.")

class BaselineLinearModel(BaseEstimator, RegressorMixin):
    """Simple linear regression baseline."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.weights_x = None
        self.weights_y = None
        self.bias_x = 0
        self.bias_y = 0
        
    def fit(self, X, y):
        """Fit linear model to predict x,y coordinates."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Separate x and y targets
        y_x = y[:, 0]  # x coordinates
        y_y = y[:, 1]  # y coordinates
        
        # Simple linear regression (using normal equation)
        XtX_inv = np.linalg.pinv(X_scaled.T @ X_scaled)
        
        self.weights_x = XtX_inv @ X_scaled.T @ y_x
        self.weights_y = XtX_inv @ X_scaled.T @ y_y
        
        # Calculate bias
        y_x_pred = X_scaled @ self.weights_x
        y_y_pred = X_scaled @ self.weights_y
        self.bias_x = np.mean(y_x - y_x_pred)
        self.bias_y = np.mean(y_y - y_y_pred)
        
        return self
    
    def predict(self, X):
        """Predict x,y coordinates."""
        X_scaled = self.scaler.transform(X)
        
        pred_x = X_scaled @ self.weights_x + self.bias_x
        pred_y = X_scaled @ self.weights_y + self.bias_y
        
        return np.column_stack([pred_x, pred_y])

class PhysicsInformedModel(BaseEstimator, RegressorMixin):
    """Physics-informed baseline using kinematic equations."""
    
    def __init__(self, max_acceleration=5.0, max_speed=12.0):
        self.max_acceleration = max_acceleration  # m/s^2
        self.max_speed = max_speed  # m/s
        self.dt = 0.1  # 10 fps = 0.1 seconds per frame
        
    def fit(self, X, y):
        """Physics model doesn't need training."""
        return self
    
    def predict(self, X):
        """Predict using physics-based motion."""
        predictions = []
        
        for i, row in enumerate(X):
            # Extract current state
            x_0, y_0 = row[0], row[1]  # Assuming first two features are x, y
            v_x_0 = row[2] if len(row) > 2 else 0  # velocity x
            v_y_0 = row[3] if len(row) > 3 else 0  # velocity y
            
            # Simple kinematic prediction (assume constant acceleration toward ball)
            # This is a placeholder - would need actual ball landing coordinates
            ball_x = row[4] if len(row) > 4 else x_0  # ball_land_x
            ball_y = row[5] if len(row) > 5 else y_0  # ball_land_y
            
            # Calculate desired acceleration toward ball
            dist_to_ball = np.sqrt((ball_x - x_0)**2 + (ball_y - y_0)**2)
            if dist_to_ball > 0:
                a_x = self.max_acceleration * (ball_x - x_0) / dist_to_ball
                a_y = self.max_acceleration * (ball_y - y_0) / dist_to_ball
            else:
                a_x, a_y = 0, 0
            
            # Predict next position using kinematic equations
            x_pred = x_0 + v_x_0 * self.dt + 0.5 * a_x * self.dt**2
            y_pred = y_0 + v_y_0 * self.dt + 0.5 * a_y * self.dt**2
            
            predictions.append([x_pred, y_pred])
        
        return np.array(predictions)

if TORCH_AVAILABLE:
    class NFLDataset(Dataset):
        """PyTorch dataset for NFL tracking data."""
        
        def __init__(self, features, targets):
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.targets[idx]

    class LSTMModel(nn.Module):
        """LSTM model for sequential player movement prediction."""
        
        def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2):
            super(LSTMModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            # x shape: (batch, sequence, features)
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Use last timestep output
            out = self.dropout(lstm_out[:, -1, :])
            out = self.fc(out)
            
            return out

    class TransformerModel(nn.Module):
        """Transformer model for player movement prediction."""
        
        def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, output_size=2):
            super(TransformerModel, self).__init__()
            
            self.input_projection = nn.Linear(input_size, d_model)
            self.positional_encoding = nn.Parameter(torch.randn(100, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.output_projection = nn.Linear(d_model, output_size)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            # x shape: (batch, sequence, features)
            seq_len = x.size(1)
            
            # Project input and add positional encoding
            x = self.input_projection(x)
            x += self.positional_encoding[:seq_len, :].unsqueeze(0)
            
            # Apply transformer
            x = self.transformer(x)
            
            # Use last timestep for prediction
            x = self.dropout(x[:, -1, :])
            out = self.output_projection(x)
            
            return out

    class NFLModelTrainer:
        """Trainer class for PyTorch models."""
        
        def __init__(self, model, device='cpu'):
            self.model = model.to(device)
            self.device = device
            self.scaler = StandardScaler()
            
        def train(self, X_train, y_train, X_val=None, y_val=None, 
                 epochs=50, batch_size=32, lr=0.001):
            """Train the model."""
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            
            # Create datasets
            train_dataset = NFLDataset(X_train_scaled, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            if X_val is not None:
                val_dataset = NFLDataset(X_val_scaled, y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                if X_val is not None:
                    self.model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch_features, batch_targets in val_loader:
                            batch_features = batch_features.to(self.device)
                            batch_targets = batch_targets.to(self.device)
                            
                            outputs = self.model(batch_features)
                            loss = criterion(outputs, batch_targets)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)
                    scheduler.step(val_loss)
                    
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            return train_losses, val_losses
        
        def predict(self, X):
            """Make predictions."""
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                predictions = self.model(X_tensor).cpu().numpy()
            
            return predictions

def calculate_rmse(y_true, y_pred):
    """Calculate RMSE for x,y coordinate predictions."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    rmse = calculate_rmse(y_test, predictions)
    
    # Separate RMSE for x and y
    rmse_x = np.sqrt(mean_squared_error(y_test[:, 0], predictions[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(y_test[:, 1], predictions[:, 1]))
    
    return {
        'rmse_total': rmse,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y
    }

if __name__ == "__main__":
    # Example usage
    print("NFL Big Data Bowl 2026 - Model Testing")
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, 2)  # x, y coordinates
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test baseline model
    print("\nTesting Baseline Linear Model...")
    baseline = BaselineLinearModel()
    baseline.fit(X_train, y_train)
    baseline_results = evaluate_model(baseline, X_test, y_test)
    print(f"Baseline RMSE: {baseline_results['rmse_total']:.4f}")
    
    # Test physics model
    print("\nTesting Physics-Informed Model...")
    physics = PhysicsInformedModel()
    physics.fit(X_train, y_train)
    physics_results = evaluate_model(physics, X_test, y_test)
    print(f"Physics RMSE: {physics_results['rmse_total']:.4f}")
    
    if TORCH_AVAILABLE:
        print("\nTesting LSTM Model...")
        # Reshape for sequence data (assuming sequence length of 1 for this test)
        X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        lstm_model = LSTMModel(input_size=n_features)
        trainer = NFLModelTrainer(lstm_model)
        trainer.train(X_train_seq, y_train, epochs=10, batch_size=32)
        
        lstm_results = evaluate_model(trainer, X_test_seq, y_test)
        print(f"LSTM RMSE: {lstm_results['rmse_total']:.4f}")
    
    print("\nModel testing complete!")