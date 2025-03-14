from typing import Dict, List, Optional, Tuple
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_curve, auc

from hep_foundation.base_model import BaseModel
from hep_foundation.variational_autoencoder import VariationalAutoEncoder
from hep_foundation.plot_utils import (
    set_science_style, get_figure_size, get_color_cycle,
    FONT_SIZES, LINE_WIDTHS
)
class ModelTester:
    """Class for evaluating trained models with various tests"""
    
    def __init__(
        self,
        model: BaseModel,
        test_dataset: tf.data.Dataset,
        signal_datasets: Optional[Dict[str, tf.data.Dataset]] = None,
        experiment_id: str = None,
        base_path: Path = Path("experiments")
    ):
        """
        Initialize the model tester
        
        Args:
            model: Trained model to evaluate
            test_dataset: Dataset of background events for testing
            signal_datasets: Dictionary of signal datasets for comparison
            experiment_id: ID of the experiment (e.g. '001_vae_test')
            base_path: Base path where experiments are stored
        """
        self.model = model
        self.test_dataset = test_dataset
        self.signal_datasets = signal_datasets or {}
        
        # Setup paths
        self.base_path = Path(base_path)
        if experiment_id is None:
            raise ValueError("experiment_id must be provided")
        
        self.experiment_path = self.base_path / experiment_id
        self.testing_path = self.experiment_path / "testing"
        self.testing_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Verify experiment data exists
        self.experiment_data_path = self.experiment_path / "experiment_data.json"
        if not self.experiment_data_path.exists():
            raise ValueError(f"No experiment data found at {self.experiment_data_path}")
            
        # Load existing experiment data
        with open(self.experiment_data_path, 'r') as f:
            self.experiment_data = json.load(f)
            
        # Initialize test results storage
        self.test_results = {}
        
    def _update_experiment_data(self, test_results: Dict) -> None:
        """Update experiment data with new test results"""
        # Add or update test results in experiment data
        if 'test_results' not in self.experiment_data:
            self.experiment_data['test_results'] = {}
        
        self.experiment_data['test_results'].update(test_results)
        
        # Save updated experiment data
        with open(self.experiment_data_path, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
            
    def _calculate_losses(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reconstruction (MSE) and KL losses for a dataset
        
        Args:
            dataset: Dataset to evaluate
            
        Returns:
            Tuple of (reconstruction_losses, kl_losses) arrays
        """
        if not isinstance(self.model, VariationalAutoEncoder):
            raise ValueError("Model must be a VariationalAutoEncoder")
        
        reconstruction_losses = []
        kl_losses = []
        
        # Log dataset info
        total_batches = 0
        total_events = 0
        
        logging.info("\nCalculating losses for dataset...")
        for batch in dataset:
            total_batches += 1
            total_events += batch.shape[0]
            
            # Get encoder outputs
            z_mean, z_log_var, z = self.model.encoder(batch)
            
            # Get reconstructions
            reconstructions = self.model.decoder(z)
            
            # Flatten input and reconstruction for loss calculation
            flat_inputs = tf.reshape(batch, [-1, tf.reduce_prod(batch.shape[1:])])
            flat_reconstructions = tf.reshape(reconstructions, [-1, tf.reduce_prod(reconstructions.shape[1:])])
            
            # Calculate losses per event (not taking the mean)
            recon_losses_batch = tf.reduce_sum(
                tf.square(flat_inputs - flat_reconstructions),
                axis=1
            ).numpy()
            
            kl_losses_batch = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            ).numpy()
            
            # Append individual event losses
            reconstruction_losses.extend(recon_losses_batch.tolist())
            kl_losses.extend(kl_losses_batch.tolist())
        
        logging.info(f"Dataset stats:")
        logging.info(f"  Total batches: {total_batches}")
        logging.info(f"  Total events: {total_events}")
        logging.info(f"  Events per batch: {total_events/total_batches if total_batches > 0 else 0:.1f}")
        
        return np.array(reconstruction_losses), np.array(kl_losses)

    def _calculate_separation_metrics(
        self,
        background_losses: np.ndarray,
        signal_losses: np.ndarray,
        loss_type: str
    ) -> Dict:
        """
        Calculate metrics for separation between background and signal
        
        Args:
            background_losses: Array of losses for background events
            signal_losses: Array of losses for signal events
            loss_type: String identifier for the type of loss
            
        Returns:
            Dictionary of separation metrics
        """
        # Calculate basic statistics
        metrics = {
            'background_mean': float(np.mean(background_losses)),
            'background_std': float(np.std(background_losses)),
            'signal_mean': float(np.mean(signal_losses)),
            'signal_std': float(np.std(signal_losses)),
            'separation': float(
                abs(np.mean(signal_losses) - np.mean(background_losses)) /
                np.sqrt(np.std(signal_losses)**2 + np.std(background_losses)**2)
            )
        }
        
        # Add ROC curve metrics
        from sklearn.metrics import roc_curve, auc
        labels = np.concatenate([np.zeros(len(background_losses)), np.ones(len(signal_losses))])
        scores = np.concatenate([background_losses, signal_losses])
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        metrics['roc_auc'] = float(roc_auc)
        
        # Option 1: Don't store ROC curve data in JSON at all
        # Just store the AUC value which is what we primarily care about
        
        # Option 2: Downsample the ROC curve to reduce size
        # Only include a maximum of 20 points to keep the JSON file manageable
        if len(fpr) > 20:
            # Get indices for approximately 20 evenly spaced points
            indices = np.linspace(0, len(fpr) - 1, 20).astype(int)
            # Make sure to always include the endpoints
            if indices[0] != 0:
                indices[0] = 0
            if indices[-1] != len(fpr) - 1:
                indices[-1] = len(fpr) - 1
            
            metrics['roc_curve'] = {
                'fpr': fpr[indices].tolist(),
                'tpr': tpr[indices].tolist(),
                'thresholds': thresholds[indices].tolist() if len(thresholds) > 0 else []
            }
        else:
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        return metrics

    def run_anomaly_detection_test(self) -> Dict:
        """
        Evaluate model's anomaly detection capabilities
        
        Compares reconstruction error (MSE) and KL divergence distributions
        between background and signal datasets.
        
        Returns:
            Dictionary containing test metrics and results
        """
        logging.info("\nRunning anomaly detection test...")
        
        if not isinstance(self.model, VariationalAutoEncoder):
            raise ValueError("Anomaly detection test requires a VariationalAutoEncoder")
        
        # Create testing/anomaly_detection directory
        test_dir = self.testing_path / "anomaly_detection"
        test_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = test_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Log dataset info before testing
        logging.info("\nDataset information before testing:")
        for batch in self.test_dataset:
            logging.info(f"Background test dataset batch shape: {batch.shape}")
            break
        for signal_name, signal_dataset in self.signal_datasets.items():
            for batch in signal_dataset:
                logging.info(f"{signal_name} signal dataset batch shape: {batch.shape}")
                break
        
        # Calculate losses for background dataset
        logging.info("\nCalculating losses for background dataset...")
        bg_recon_losses, bg_kl_losses = self._calculate_losses(self.test_dataset)
        
        # Calculate losses for each signal dataset
        signal_results = {}
        for signal_name, signal_dataset in self.signal_datasets.items():
            logging.info(f"\nCalculating losses for signal dataset: {signal_name}")
            sig_recon_losses, sig_kl_losses = self._calculate_losses(signal_dataset)
            
            # Calculate separation metrics
            recon_metrics = self._calculate_separation_metrics(
                bg_recon_losses, sig_recon_losses, "reconstruction"
            )
            kl_metrics = self._calculate_separation_metrics(
                bg_kl_losses, sig_kl_losses, "kl_divergence"
            )
            
            signal_results[signal_name] = {
                'reconstruction_metrics': recon_metrics,
                'kl_divergence_metrics': kl_metrics,
                'n_events': len(sig_recon_losses)
            }
            
            # Create comparison plots
            self._plot_loss_distributions(
                bg_recon_losses, sig_recon_losses,
                bg_kl_losses, sig_kl_losses,
                signal_name, plots_dir
            )
        
        # Prepare test results
        test_results = {
            'anomaly_detection': {
                'timestamp': str(datetime.now()),
                'background_events': len(bg_recon_losses),
                'signal_results': signal_results,
                'plots_directory': str(plots_dir)
            }
        }
        
        # Update experiment data with test results
        self._update_experiment_data(test_results)
        
        return test_results

    def _plot_loss_distributions(
        self,
        bg_recon_losses: np.ndarray,
        sig_recon_losses: np.ndarray,
        bg_kl_losses: np.ndarray,
        sig_kl_losses: np.ndarray,
        signal_name: str,
        plots_dir: Path
    ) -> None:
        """Create plots comparing background and signal loss distributions"""
        # Set style
        set_science_style(use_tex=False)
        colors = get_color_cycle('high_contrast')
        
        # 1. Loss distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size('double'))
        
        # Calculate percentile-based limits for better visualization
        recon_upper_limit = np.percentile(np.concatenate([bg_recon_losses, sig_recon_losses]), 99)
        kl_upper_limit = np.percentile(np.concatenate([bg_kl_losses, sig_kl_losses]), 99)
        
        # Reconstruction loss
        ax1.hist(bg_recon_losses, bins=50, alpha=0.5, color=colors[0],
                 label='Background', density=True, range=(0, recon_upper_limit))
        ax1.hist(sig_recon_losses, bins=50, alpha=0.5, color=colors[1],
                 label=signal_name, density=True, range=(0, recon_upper_limit))
        ax1.set_xlabel('Reconstruction Loss')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # KL divergence
        ax2.hist(bg_kl_losses, bins=50, alpha=0.5, color=colors[0],
                 label='Background', density=True, range=(0, kl_upper_limit))
        ax2.hist(sig_kl_losses, bins=50, alpha=0.5, color=colors[1],
                 label=signal_name, density=True, range=(0, kl_upper_limit))
        ax2.set_xlabel('KL Divergence')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add percentile information to the plot titles
        ax1.set_title(f'Reconstruction Loss (99th percentile: {recon_upper_limit:.1f})')
        ax2.set_title(f'KL Divergence (99th percentile: {kl_upper_limit:.1f})')
        
        plt.suptitle(f'Loss Distributions: Background vs {signal_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f'loss_distributions_{signal_name}.pdf')
        
        
        # 2. ROC curves
        plt.figure(figsize=get_figure_size('single'))
        
        # ROC for reconstruction loss
        labels = np.concatenate([np.zeros(len(bg_recon_losses)), np.ones(len(sig_recon_losses))])
        scores = np.concatenate([bg_recon_losses, sig_recon_losses])
        fpr_recon, tpr_recon, _ = roc_curve(labels, scores)
        roc_auc_recon = auc(fpr_recon, tpr_recon)
        
        # ROC for KL loss
        scores_kl = np.concatenate([bg_kl_losses, sig_kl_losses])
        fpr_kl, tpr_kl, _ = roc_curve(labels, scores_kl)
        roc_auc_kl = auc(fpr_kl, tpr_kl)
        
        plt.plot(fpr_recon, tpr_recon, color=colors[0],
                 label=f'Reconstruction (AUC = {roc_auc_recon:.3f})')
        plt.plot(fpr_kl, tpr_kl, color=colors[1],
                 label=f'KL Divergence (AUC = {roc_auc_kl:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: Background vs {signal_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(plots_dir / f'roc_curves_{signal_name}.pdf')
        plt.close() 