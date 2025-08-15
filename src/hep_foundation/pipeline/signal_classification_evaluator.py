import json
from pathlib import Path
from typing import Optional

import tensorflow as tf

from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.models.base_model import CustomKerasModelWrapper
from hep_foundation.models.dnn_predictor import DNNPredictorConfig
from hep_foundation.models.model_factory import ModelFactory
from hep_foundation.pipeline.foundation_pipeline_utils import log_evaluation_summary
from hep_foundation.plots.foundation_plot_manager import FoundationPlotManager
from hep_foundation.training.model_trainer import ModelTrainer


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

            from hep_foundation.config.config_loader import PipelineConfigLoader

            config_loader = PipelineConfigLoader()
            vae_config_data = config_loader.load_config(vae_config_path)

            if "models" in vae_config_data and "vae" in vae_config_data["models"]:
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

            # Helper function to build classifier head
            def build_classifier_head(name_suffix: str) -> tf.keras.Model:
                # Create a copy of the DNN config with modified architecture for binary classification
                import copy

                classifier_config = copy.deepcopy(dnn_model_config)

                # Update the architecture for the classifier head
                classifier_config.architecture.update(
                    {
                        "input_shape": (latent_dim,),
                        "output_shape": classification_output_shape,
                        "output_activation": "sigmoid",  # Binary classification
                        "name": f"{dnn_model_config.architecture.get('name', 'classifier')}_{name_suffix}",
                    }
                )

                classifier_model_wrapper = ModelFactory.create_model(
                    model_type="dnn_predictor", config=classifier_config
                )
                classifier_model_wrapper.build()
                return classifier_model_wrapper.model

            # Helper function to create subset dataset
            def create_subset_dataset(dataset, num_events, data_size_index=None):
                """Create a subset of the dataset with exactly num_events events"""
                # Convert to unbatched dataset to count events precisely
                unbatched = dataset.unbatch()
                # Use different seed for each data size to ensure independent sampling
                # This prevents smaller datasets from being strict subsets of larger ones
                seed = 42 + (data_size_index or 0)  # Different seed for each data size
                shuffled = unbatched.shuffle(buffer_size=50000, seed=seed)
                subset = shuffled.take(num_events)
                # Rebatch with original batch size
                return subset.batch(dnn_training_config.batch_size)

            # Helper function to train and evaluate a model for a specific data size
            def train_and_evaluate_for_size(
                model_name: str,
                combined_keras_model: tf.keras.Model,
                train_subset,
                data_size: int,
                save_training_history: bool = False,
                label_distributions_dir_param: Optional[Path] = None,
                label_variable_names_param: Optional[list] = None,
            ):
                self.logger.info(
                    f"Training {model_name} model with {data_size} events..."
                )

                # Wrap the Keras model with CustomKerasModelWrapper for ModelTrainer
                wrapped_model_for_trainer = CustomKerasModelWrapper(
                    combined_keras_model, name=model_name
                )

                trainer_config_dict = {
                    "batch_size": dnn_training_config.batch_size,
                    "epochs": fixed_epochs,  # Use fixed epochs for fair comparison
                    "learning_rate": dnn_training_config.learning_rate,
                    "early_stopping": {
                        "patience": fixed_epochs + 1,
                        "min_delta": 0,
                    },  # Disable early stopping
                }

                trainer = ModelTrainer(
                    model=wrapped_model_for_trainer, training_config=trainer_config_dict
                )

                # Train with reduced verbosity for evaluation
                _ = trainer.train(  # Unused return value
                    dataset=train_subset,
                    validation_data=labeled_val_dataset,
                    callbacks=[],  # No callbacks for speed
                    training_history_dir=classification_dir / "training_histories"
                    if save_training_history
                    else None,
                    model_name=model_name,
                    dataset_id=f"signal_classification_eval_{data_size}",
                    experiment_id="signal_classification_evaluation",
                    verbose_training="minimal",  # Reduce verbosity for evaluation models
                    save_individual_history=True,  # Save individual files for comparison plots
                )

                # Evaluate on test set
                test_metrics = trainer.evaluate(labeled_test_dataset)
                test_loss = test_metrics.get("test_loss", 0.0)
                test_accuracy = test_metrics.get("test_binary_accuracy", 0.0)

                self.logger.info(
                    f"{model_name} with {data_size} events - Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}"
                )
                return test_loss, test_accuracy

            # Store results for plotting
            results = {
                "data_sizes": data_sizes,
                "From_Scratch_loss": [],
                "Fine_Tuned_loss": [],
                "Fixed_Encoder_loss": [],
                "From_Scratch_accuracy": [],
                "Fine_Tuned_accuracy": [],
                "Fixed_Encoder_accuracy": [],
            }

            # 5. Run experiments for each data size
            for data_size_index, data_size in enumerate(data_sizes):
                self.logger.info(f"{'=' * 50}")
                self.logger.info(f"Training with {data_size} events")
                self.logger.info(f"{'=' * 50}")

                # Create subset of training data
                train_subset = create_subset_dataset(
                    labeled_train_dataset, data_size, data_size_index
                )

                # Enable training history saving for all data sizes
                should_save_history = True
                self.logger.info(
                    f"Enabling training history saving for all models with {data_size} events"
                )

                # Format data size for better labeling
                data_size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )

                # --- Model 1: From Scratch ---
                self.logger.info("Building From Scratch model...")
                scratch_encoder_layers = []
                for units in encoder_hidden_layers:
                    scratch_encoder_layers.append(
                        tf.keras.layers.Dense(units, activation=encoder_activation)
                    )
                scratch_encoder_layers.append(
                    tf.keras.layers.Dense(latent_dim, name="scratch_latent_space")
                )

                scratch_encoder_part = tf.keras.Sequential(
                    scratch_encoder_layers, name="scratch_encoder"
                )
                scratch_classifier_dnn = build_classifier_head("from_scratch")

                model_inputs = tf.keras.Input(
                    shape=original_input_shape, name="input_features"
                )
                encoded_scratch = scratch_encoder_part(model_inputs)
                predictions_scratch = scratch_classifier_dnn(encoded_scratch)
                model_from_scratch = tf.keras.Model(
                    inputs=model_inputs,
                    outputs=predictions_scratch,
                    name="Classifier_From_Scratch",
                )

                scratch_loss, scratch_accuracy = train_and_evaluate_for_size(
                    f"From_Scratch_{data_size_label}",
                    model_from_scratch,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=None,  # Signal classification doesn't need label distribution analysis
                    label_variable_names_param=None,  # Signal classification uses binary labels
                )
                results["From_Scratch_loss"].append(scratch_loss)
                results["From_Scratch_accuracy"].append(scratch_accuracy)

                # --- Model 2: Fine-Tuned ---
                self.logger.info("Building Fine-Tuned model...")
                fine_tuned_input = tf.keras.Input(
                    shape=original_input_shape, name="fine_tuned_input"
                )
                fine_tuned_encoded = pretrained_deterministic_encoder(fine_tuned_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fine_tuned_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32),
                    name="dtype_cast_fine_tuned_classifier",
                )(fine_tuned_encoded)

                fine_tuned_encoder_part = tf.keras.Model(
                    inputs=fine_tuned_input,
                    outputs=fine_tuned_encoded_cast,
                    name="fine_tuned_pretrained_encoder",
                )
                fine_tuned_encoder_part.trainable = True

                fine_tuned_classifier_dnn = build_classifier_head("fine_tuned")

                model_inputs_ft = tf.keras.Input(
                    shape=original_input_shape, name="input_features_ft"
                )
                encoded_ft = fine_tuned_encoder_part(model_inputs_ft)
                predictions_ft = fine_tuned_classifier_dnn(encoded_ft)
                model_fine_tuned = tf.keras.Model(
                    inputs=model_inputs_ft,
                    outputs=predictions_ft,
                    name="Classifier_Fine_Tuned",
                )

                finetuned_loss, finetuned_accuracy = train_and_evaluate_for_size(
                    f"Fine_Tuned_{data_size_label}",
                    model_fine_tuned,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=None,  # Signal classification doesn't need label distribution analysis
                    label_variable_names_param=None,  # Signal classification uses binary labels
                )
                results["Fine_Tuned_loss"].append(finetuned_loss)
                results["Fine_Tuned_accuracy"].append(finetuned_accuracy)

                # --- Model 3: Fixed Encoder ---
                self.logger.info("Building Fixed Encoder model...")
                fixed_input = tf.keras.Input(
                    shape=original_input_shape, name="fixed_input"
                )
                fixed_encoded = pretrained_deterministic_encoder(fixed_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fixed_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32), name="dtype_cast_fixed_classifier"
                )(fixed_encoded)

                fixed_encoder_part = tf.keras.Model(
                    inputs=fixed_input,
                    outputs=fixed_encoded_cast,
                    name="fixed_pretrained_encoder",
                )
                fixed_encoder_part.trainable = False

                fixed_classifier_dnn = build_classifier_head("fixed_encoder")

                model_inputs_fx = tf.keras.Input(
                    shape=original_input_shape, name="input_features_fx"
                )
                encoded_fx = fixed_encoder_part(model_inputs_fx)
                predictions_fx = fixed_classifier_dnn(encoded_fx)
                model_fixed = tf.keras.Model(
                    inputs=model_inputs_fx,
                    outputs=predictions_fx,
                    name="Classifier_Fixed_Encoder",
                )

                fixed_loss, fixed_accuracy = train_and_evaluate_for_size(
                    f"Fixed_Encoder_{data_size_label}",
                    model_fixed,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=None,  # Signal classification doesn't need label distribution analysis
                    label_variable_names_param=None,  # Signal classification uses binary labels
                )
                results["Fixed_Encoder_loss"].append(fixed_loss)
                results["Fixed_Encoder_accuracy"].append(fixed_accuracy)

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

            # Create combined training history plot if we saved training histories
            training_histories_dir = classification_dir / "training_histories"
            combined_plot_path = (
                classification_dir / "signal_classification_training_comparison.png"
            )
            self.foundation_plot_manager.create_training_history_comparison_plot_from_directory(
                training_histories_dir,
                combined_plot_path,
                title_prefix="Signal Classification Model Training Comparison",
                validation_only=True,
            )

            # Create the data efficiency plots
            # Plot 1: Loss comparison
            loss_plot_file = classification_dir / "signal_classification_loss_plot.png"
            self.foundation_plot_manager.create_data_efficiency_plot(
                results,
                loss_plot_file,
                plot_type="classification_loss",
                metric_name="Test Loss (Binary Crossentropy)",
                total_test_events=total_test_events,
                signal_key=signal_key,
            )

            # Plot 2: Accuracy comparison
            accuracy_plot_file = (
                classification_dir / "signal_classification_accuracy_plot.png"
            )
            self.foundation_plot_manager.create_data_efficiency_plot(
                results,
                accuracy_plot_file,
                plot_type="classification_accuracy",
                metric_name="Test Accuracy",
                total_test_events=total_test_events,
                signal_key=signal_key,
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
