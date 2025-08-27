import json
from pathlib import Path
from typing import Optional

import tensorflow as tf

from hep_foundation.config.config_loader import PipelineConfigLoader
from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.models.dnn_predictor import DNNPredictorConfig
from hep_foundation.pipeline.downstream_model_manager import DownstreamModelManager
from hep_foundation.pipeline.foundation_pipeline_utils import log_evaluation_summary
from hep_foundation.plots.foundation_plot_manager import FoundationPlotManager


class SignalClassificationEvaluator:
    """
    Handles signal classification evaluation functionality for foundation models.
    """

    def __init__(
        self,
        processed_datasets_dir: str,
        logger=None,
        foundation_plot_manager=None,
    ):
        """
        Initialize the signal classification evaluator.

        Args:
            processed_datasets_dir: Directory for processed datasets
            logger: Logger instance (optional, will create one if not provided)
            foundation_plot_manager: FoundationPlotManager instance (optional, will create one if not provided)
        """
        self.processed_datasets_dir = Path(processed_datasets_dir)
        self.logger = logger or get_logger(__name__)
        self.foundation_plot_manager = (
            foundation_plot_manager or FoundationPlotManager()
        )
        self.downstream_manager = DownstreamModelManager(logger=self.logger)

    def evaluate_signal_classification(
        self,
        dataset_config: DatasetConfig,
        dnn_model_config: DNNPredictorConfig,
        dnn_training_config: TrainingConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
        data_sizes: list = None,
        fixed_epochs: int = 10,
        seed: Optional[int] = None,
    ) -> bool:
        """
        Evaluate foundation model for signal classification using data efficiency study.

        This method creates balanced datasets mixing background and signal data with binary labels,
        then trains three models (From Scratch, Fine-Tuned, Fixed) on increasing amounts of training data
        to show how pre-trained weights help with limited signal data.

        Args:
            dataset_config: Configuration for dataset processing
            dnn_model_config: Configuration for DNN model
            dnn_training_config: Configuration for DNN training
            task_config: Configuration for task processing
            delete_catalogs: Whether to delete catalogs after processing
            foundation_model_path: Path to the foundation model encoder
            data_sizes: List of training data sizes to test (e.g., [1000, 2000, 5000, 10000])
            fixed_epochs: Number of epochs to train each model for each data size
            seed: Random seed for reproducible weight initialization (optional)
        """

        if not foundation_model_path:
            self.logger.error(
                "Foundation model path must be provided for signal classification evaluation."
            )
            return False

        if not dataset_config.signal_keys:
            self.logger.error(
                "Signal keys must be configured for signal classification evaluation."
            )
            return False

        foundation_model_dir = Path(foundation_model_path)
        classification_dir = foundation_model_dir / "testing" / "signal_classification"
        classification_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize data manager
            data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

            # 1. Load background dataset (no labels needed)
            self.logger.info("Loading background dataset...")
            background_train, background_val, background_test = (
                data_manager.load_atlas_datasets(
                    dataset_config=dataset_config,
                    validation_fraction=dataset_config.validation_fraction,
                    test_fraction=dataset_config.test_fraction,
                    batch_size=dnn_training_config.batch_size,
                    shuffle_buffer=dataset_config.shuffle_buffer,
                    include_labels=False,  # No labels needed for background
                    delete_catalogs=delete_catalogs,
                )
            )

            # 2. Prepare background histogram path for comparison plot generation
            background_hist_data_path_for_comparison = None
            dataset_id = data_manager.generate_dataset_id(dataset_config)
            potential_background_hist_path = (
                data_manager.get_dataset_dir(dataset_id)
                / "plot_data"
                / "atlas_dataset_features_hist_data.json"
            )
            if potential_background_hist_path.exists():
                background_hist_data_path_for_comparison = (
                    potential_background_hist_path
                )
                self.logger.info(
                    f"Found background histogram data for comparison at {potential_background_hist_path}"
                )
            else:
                self.logger.warning(
                    f"Background histogram data for comparison not found at {potential_background_hist_path}. Comparison plot may be skipped by DatasetManager."
                )

            # 3. Load first signal dataset with proper train/val/test splits
            signal_key = dataset_config.signal_keys[0]
            self.logger.info(f"Loading signal dataset with splits: {signal_key}")

            signal_datasets_splits = data_manager.load_signal_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=dnn_training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=False,  # No labels needed for signals
                background_hist_data_path=background_hist_data_path_for_comparison,  # Pass the path here
                split=True,  # Enable splitting
            )

            if signal_key not in signal_datasets_splits:
                self.logger.error(f"Signal dataset '{signal_key}' not found")
                return False

            signal_train, signal_val, signal_test = signal_datasets_splits[signal_key]

            # 4. Create balanced labeled datasets
            self.logger.info("Creating balanced labeled datasets...")

            def create_balanced_labeled_dataset(bg_dataset, sig_dataset):
                """Create a balanced dataset with binary labels"""
                # Convert to unbatched datasets for easier manipulation
                bg_unbatched = bg_dataset.unbatch()
                sig_unbatched = sig_dataset.unbatch()

                # Add labels: 0 for background, 1 for signal
                bg_labeled = bg_unbatched.map(
                    lambda x: (x, tf.constant(0.0, dtype=tf.float32))
                )
                sig_labeled = sig_unbatched.map(
                    lambda x: (x, tf.constant(1.0, dtype=tf.float32))
                )

                # Combine and shuffle
                combined = bg_labeled.concatenate(sig_labeled)
                combined = combined.shuffle(buffer_size=10000, seed=42)

                # Rebatch
                return combined.batch(dnn_training_config.batch_size)

            # Create balanced train, validation, and test datasets using proper signal splits
            labeled_train_dataset = create_balanced_labeled_dataset(
                background_train, signal_train
            )
            labeled_val_dataset = create_balanced_labeled_dataset(
                background_val, signal_val
            )
            labeled_test_dataset = create_balanced_labeled_dataset(
                background_test, signal_test
            )

            # Count total training events
            total_train_events = 0
            for batch in labeled_train_dataset:
                batch_size = tf.shape(batch[0])[0]
                total_train_events += batch_size.numpy()

            # Count total test events for plot labeling
            total_test_events = 0
            for batch in labeled_test_dataset:
                batch_size = tf.shape(batch[0])[0]
                total_test_events += batch_size.numpy()

            self.logger.info(f"Total training events available: {total_train_events}")
            self.logger.info(f"Total test events: {total_test_events}")

            # Filter data_sizes to only include sizes <= total_train_events
            data_sizes = [
                size
                for size in (data_sizes or [total_train_events])
                if size <= total_train_events
            ]
            self.logger.info(f"Data sizes to test: {data_sizes}")

            # 5. Load Pre-trained Foundation Encoder & its Config
            self.logger.info(
                f"Loading foundation model configuration from: {foundation_model_dir}"
            )

            # Load the VAE model config from the YAML config file
            vae_config_path = foundation_model_dir / "_experiment_config.yaml"
            if not vae_config_path.exists():
                self.logger.error(
                    f"Foundation model config not found at: {vae_config_path}"
                )
                return False

            config_loader = PipelineConfigLoader()
            vae_config_data = config_loader.load_config(vae_config_path)

            if "foundation_model_training" in vae_config_data:
                foundation_config = vae_config_data["foundation_model_training"]
                original_vae_model_config = {
                    "model_type": foundation_config["model"]["model_type"],
                    "architecture": foundation_config["model"]["architecture"],
                    "hyperparameters": foundation_config["model"]["hyperparameters"],
                }
            elif "models" in vae_config_data and "vae" in vae_config_data["models"]:
                # Backward compatibility with old format
                original_vae_model_config = vae_config_data["models"]["vae"]
            else:
                self.logger.error(
                    f"Could not find VAE model config in: {vae_config_path}"
                )
                return False

            vae_arch_config = original_vae_model_config["architecture"]
            latent_dim = vae_arch_config["latent_dim"]
            encoder_hidden_layers = vae_arch_config.get("encoder_layers", [])
            encoder_activation = vae_arch_config.get("activation", "relu")

            # Load the pre-trained deterministic encoder directly
            pretrained_deterministic_encoder_path = (
                foundation_model_dir
                / "models"
                / "foundation_model"
                / "deterministic_encoder"
            )
            if not pretrained_deterministic_encoder_path.exists():
                self.logger.error(
                    f"Pretrained deterministic encoder not found at {pretrained_deterministic_encoder_path}"
                )
                return False

            self.logger.info(
                f"Loading pre-trained deterministic encoder from: {pretrained_deterministic_encoder_path}"
            )
            pretrained_deterministic_encoder = tf.keras.models.load_model(
                pretrained_deterministic_encoder_path
            )

            self.logger.info(
                f"Loaded deterministic encoder with output shape: {pretrained_deterministic_encoder.output.shape}"
            )

            original_input_shape = (task_config.input.get_total_feature_size(),)
            classification_output_shape = (1,)  # Binary classification

            # 5. Run data efficiency study for signal classification
            from hep_foundation.pipeline.downstream_model_manager import (
                DownstreamModelType,
            )

            # Initialize results dictionary for classification (with accuracy)
            results = {
                "data_sizes": data_sizes,
                "From_Scratch_loss": [],
                "Fine_Tuned_loss": [],
                "Fixed_Encoder_loss": [],
                "From_Scratch_accuracy": [],
                "Fine_Tuned_accuracy": [],
                "Fixed_Encoder_accuracy": [],
            }

            # Run experiments for each data size
            for data_size_index, data_size in enumerate(data_sizes):
                self.logger.info(f"{'=' * 50}")
                self.logger.info(f"Training with {data_size} events")
                self.logger.info(f"{'=' * 50}")

                # Create subset of training data
                train_subset = self.downstream_manager.create_subset_dataset(
                    labeled_train_dataset,
                    data_size,
                    dnn_training_config.batch_size,
                    data_size_index,
                )

                # Format data size for better labeling
                data_size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )

                # Test all three model types
                model_types = [
                    DownstreamModelType.FROM_SCRATCH,
                    DownstreamModelType.FINE_TUNED,
                    DownstreamModelType.FIXED_ENCODER,
                ]

                for model_type in model_types:
                    self.logger.info(
                        f"Building {model_type.value.replace('_', ' ').title()} model..."
                    )

                    # Create the model
                    model = self.downstream_manager.create_downstream_model(
                        model_type=model_type,
                        input_shape=original_input_shape,
                        output_shape=classification_output_shape,
                        pretrained_encoder=pretrained_deterministic_encoder,
                        encoder_hidden_layers=encoder_hidden_layers,
                        latent_dim=latent_dim,
                        dnn_model_config=dnn_model_config,
                        encoder_activation=encoder_activation,
                        output_activation="sigmoid",
                    )

                    # Train and evaluate
                    # Preserve the model name created by downstream_manager which contains "Classifier"
                    # This is crucial for ModelTrainer to detect it's a classification task
                    base_model_name = model.name
                    model_name = f"{base_model_name}_{data_size_label}"

                    training_config = {
                        "batch_size": dnn_training_config.batch_size,
                        "learning_rate": dnn_training_config.learning_rate,
                    }

                    evaluation_results = (
                        self.downstream_manager.train_and_evaluate_model(
                            model=model,
                            model_name=model_name,
                            train_dataset=train_subset,
                            val_dataset=labeled_val_dataset,
                            test_dataset=labeled_test_dataset,
                            training_config=training_config,
                            fixed_epochs=fixed_epochs,
                            data_size=data_size,
                            output_dir=classification_dir,
                            save_training_history=True,
                            return_accuracy=True,
                            seed=seed,
                        )
                    )

                    # Store results
                    test_loss, test_accuracy = evaluation_results
                    model_key = (
                        model_type.value.replace("_", " ").title().replace(" ", "_")
                    )
                    results[f"{model_key}_loss"].append(test_loss)
                    results[f"{model_key}_accuracy"].append(test_accuracy)

            # 6. Save results and create plots
            self.logger.info("Creating data efficiency plots...")

            # Save results to JSON
            results_file = (
                classification_dir
                / "signal_classification_data_efficiency_results.json"
            )
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {results_file}")

            # Create combined training history and data efficiency plots
            training_histories_dir = classification_dir / "training_histories"

            # Plot 1: Loss comparison
            loss_plot_file = (
                classification_dir
                / "signal_classification_loss_evaluation_combined.png"
            )
            self.foundation_plot_manager.create_combined_downstream_evaluation_plot(
                training_histories_dir,
                results,
                loss_plot_file,
                plot_type="classification_loss",
                metric_name="Test Loss (Binary Crossentropy)",
                total_test_events=total_test_events,
                signal_key=signal_key,
                title_prefix="Signal Classification Evaluation: Training History & Loss",
            )

            # Plot 2: Accuracy comparison
            accuracy_plot_file = (
                classification_dir
                / "signal_classification_accuracy_evaluation_combined.png"
            )
            self.foundation_plot_manager.create_combined_downstream_evaluation_plot(
                training_histories_dir,
                results,
                accuracy_plot_file,
                plot_type="classification_accuracy",
                metric_name="Test Accuracy",
                total_test_events=total_test_events,
                signal_key=signal_key,
                title_prefix="Signal Classification Evaluation: Training History & Accuracy",
            )

            # 7. Display summary
            log_evaluation_summary(
                results, evaluation_type="signal_classification", signal_key=signal_key
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Signal classification evaluation failed: {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Detailed traceback:")
            return False
