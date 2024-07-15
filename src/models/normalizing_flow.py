import torch
import torch.nn as nn
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import imageio
import wandb
import datetime

class ConditionalNormalizingFlow:
    def __init__(self, context_size, latent_size=1, num_flow_layers=4, hidden_units=128, hidden_layers=3):
        self.context_size = context_size
        self.latent_size = latent_size
        self.num_flow_layers = num_flow_layers
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.base_dist = nf.distributions.DiagGaussian(latent_size, trainable=False)
        
        self.flows = []
        for _ in range(num_flow_layers):
            self.flows += [nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
            self.flows += [nf.flows.LULinearPermute(latent_size)]
        
        self.model = nf.ConditionalNormalizingFlow(self.base_dist, self.flows).to(self.device)
        self.print_initialization_details()

    def train(self, X_train, 
                    y_train, 
                    X_val, 
                    y_val, 
                    max_iter=15000, 
                    batch_size=128, 
                    lr=3e-4, 
                    weight_decay=1e-5,
                    project="wildfire flow",
                    name="test_ex",
                    save_dir=None,
                    log_wandb=False):
                # Move all data to the appropriate device
        X_train, y_train = self.to_device((X_train, y_train))
        X_val, y_val = self.to_device((X_val, y_val))

        self.log_wandb = log_wandb
        # Print shapes of training and validation sets
        print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Validation set shape: X_val: {X_val.shape}, y_val: {y_val.shape}")
                # Print data types
        print(f"Training set dtypes: X_train: {X_train.dtype}, y_train: {y_train.dtype}")
        print(f"Validation set dtypes: X_val: {X_val.dtype}, y_val: {y_val.dtype}")
        
        # Print some statistics
        print(f"Training set stats: X_train min: {X_train.min()}, X_train max: {X_train.max()}")
        print(f"Training set stats: y_train min: {y_train.min()}, y_train max: {y_train.max()}")
        print(f"Validation set stats: X_val min: {X_val.min()}, X_val max: {X_val.max()}")
        print(f"Validation set stats: y_val min: {y_val.min()}, y_val max: {y_val.max()}")
        
        # Assert that the number of features in training and validation sets are the same
        assert X_train.shape[1] == X_val.shape[1], "Number of features in training and validation sets do not match"
        
        if save_dir is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = f"wildfire-ignition-generator/data/model_runs/{timestamp}_{name}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save directory: {save_dir}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if log_wandb:
            print(f"Wandb Project {project}, Name: {name}")
            wandb.init(project=project, name=name)
            wandb.config.update({
                "latent_size": self.latent_size,
                "num_flow_layers": self.num_flow_layers,
                "hidden_units_per_layer": self.hidden_units,
                "num_hidden_layers": self.hidden_layers,
                "context_size": self.context_size,
                "max_iterations": max_iter,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay
            })
        
        train_loss_hist = []
        val_loss_hist = []
        
        for it in tqdm(range(max_iter)):
            self.model.train()
            optimizer.zero_grad()
            
            indices = np.random.choice(len(X_train), batch_size)
            context = X_train[indices]
            targets = torch.log1p(y_train[indices]).unsqueeze(1)  # Log transform
            
            # Assert the context shape
            assert context.shape[1] == self.context_size, f"Expected context size {self.context_size}, got {context.shape[1]}"
            
            train_loss = -self.model.log_prob(targets, context).mean()
            train_loss.backward()
            optimizer.step()
            
            train_loss_hist.append(train_loss.item())
            if log_wandb:
                wandb.log({"train_loss": train_loss.item(), "iteration": it + 1})
            
            with torch.no_grad():
                val_targets = torch.log1p(y_val).unsqueeze(1)  # Log transform
                val_loss = -self.model.log_prob(val_targets, X_val).mean()
                val_loss_hist.append(val_loss.item())
                if log_wandb:
                    wandb.log({"val_loss": val_loss.item(), "iteration": it + 1})
            
            if it % 250 == 0:
                print(f"Iteration {it+1}: Train Loss = {train_loss.item()}, Val Loss = {val_loss.item()}")
                self.plot_and_save(X_val, y_val, it, save_dir, log_wandb)
            
            if it % 750 == 0:
                self.save_model(os.path.join(save_dir, f'model_{it+1:04d}.pth'))
        
        self.create_training_gif(save_dir, log_wandb)
        self.plot_loss_history(train_loss_hist, val_loss_hist, save_dir, log_wandb)
        if log_wandb:
            wandb.finish()
    
    def to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return [self.to_device(item) for item in data]
        elif isinstance(data, dict):
            return {key: self.to_device(value) for key, value in data.items()}
        else:
            return data

    def plot_and_save(self, X_val, y_val, iteration, save_dir, log_wandb):
        self.model.eval()
        with torch.no_grad():
            samples, _ = self.model.sample(len(X_val), context=X_val)
            samples = torch.expm1(samples)  # Inverse log transform
            samples = torch.clamp(samples, min=4)
        
        plt.figure(figsize=(12, 8))
        plt.hist(samples.cpu().numpy(), bins=50, density=True, alpha=0.6, color='b', label='CNF Generated Size')
        plt.hist(y_val.cpu().numpy(), bins=50, density=True, alpha=0.6, color='g', label='Validation Data Size')
        plt.title(f'Comparison of Learned Distribution to Original Data\nIteration {iteration+1}')
        plt.xlabel('Wildfire Size (Acres)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir, f'plot_{iteration+1:04d}.png'))
        plt.show()
        plt.close()
        
        if log_wandb:
            wandb.log({"generated_vs_validation_plot": wandb.Image(os.path.join(save_dir, f'plot_{iteration+1:04d}.png'))})
        
        self.plot_qq(y_val, samples, iteration, save_dir, log_wandb)
    
    def plot_qq(self, y_val, samples, iteration, save_dir, log_wandb):
        y_val_np = y_val.cpu().numpy()
        samples_np = samples.cpu().numpy()
        
        quantiles = np.linspace(0, 99, 50)
        actual_quantiles = np.percentile(y_val_np, quantiles)
        predicted_quantiles = np.percentile(samples_np, quantiles)
        
        plt.figure(figsize=(10, 6))
        plt.plot(actual_quantiles, predicted_quantiles, 'o')
        plt.plot([min(actual_quantiles), max(actual_quantiles)], [min(actual_quantiles), max(actual_quantiles)], 'r--')
        plt.xlabel('Actual Quantiles')
        plt.ylabel('Predicted Quantiles')
        plt.title('Q-Q Plot')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'qq_plot_{iteration+1:04d}.png'))
        plt.close()
        
        if log_wandb:
            wandb.log({"qq_plot": wandb.Image(os.path.join(save_dir, f'qq_plot_{iteration+1:04d}.png'))})
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        if self.log_wandb:
            wandb.save(path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

        
    def create_training_gif(self, save_dir, log_wandb):
        with imageio.get_writer(os.path.join(save_dir, 'training_progress.gif'), mode='I', duration=1.0, loop=0) as writer:
            for i in range(0, 10000, 250):
                filename = os.path.join(save_dir, f'plot_{i+1:04d}.png')
                if os.path.exists(filename):
                    image = imageio.imread(filename)
                    for _ in range(10):
                        writer.append_data(image)
        
        if log_wandb:
            wandb.log({"training_progress_gif": wandb.Video(os.path.join(save_dir, 'training_progress.gif'))})
    
    def plot_loss_history(self, train_loss_hist, val_loss_hist, save_dir, log_wandb):
        def moving_average_and_ci(data, window=50):
            rolling_mean = np.convolve(data, np.ones(window)/window, mode='valid')
            rolling_std = np.std([data[i:i+window] for i in range(len(data) - window + 1)], axis=1)
            ci_upper = rolling_mean + rolling_std
            ci_lower = rolling_mean - rolling_std
            return rolling_mean, ci_upper, ci_lower

        train_mean, train_ci_upper, train_ci_lower = moving_average_and_ci(train_loss_hist)
        val_mean, val_ci_upper, val_ci_lower = moving_average_and_ci(val_loss_hist)

        plt.figure(figsize=(10, 6))
        plt.plot(train_mean, label='Train Loss', color='blue')
        plt.fill_between(range(len(train_mean)), train_ci_lower, train_ci_upper, color='blue', alpha=0.2)
        plt.plot(val_mean, label='Validation Loss', color='orange', linewidth=2)
        plt.fill_between(range(len(val_mean)), val_ci_lower, val_ci_upper, color='orange', alpha=0.2)
        plt.legend()
        plt.title('Training and Validation Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'train_loss.png'))
        plt.close()

        if log_wandb:
            wandb.log({"train_loss_plot": wandb.Image(os.path.join(save_dir, 'train_loss.png'))})

    def print_initialization_details(self):
        print(f"ConditionalNormalizingFlow initialized with the following parameters:")
        print(f"Context size: {self.context_size}")
        print(f"Latent size: {self.latent_size}")
        print(f"Number of flow layers: {self.num_flow_layers}")
        print(f"Hidden units per layer: {self.hidden_units}")
        print(f"Number of hidden layers: {self.hidden_layers}")
        print(f"Device: {self.device}")

# Usage example
if __name__ == "__main__":
    print('Training Flow')
    # Assume X_train, y_train, X_val, y_val are prepared and loaded
    # context_size = X_train.shape[1]
    
    # cnf = ConditionalNormalizingFlow(context_size)
    # cnf.train(X_train, y_train, X_val, y_val, log_wandb=False)
