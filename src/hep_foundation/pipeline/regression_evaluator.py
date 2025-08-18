from pathlib import Path
from typing import Optional

import numpy as np
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
from hep_foundation.plots.histogram_manager import HistogramManager
from hep_foundation.training.model_trainer import ModelTrainer


class RegressionEvaluator:
    """
    Handles regression evaluation functionality for foundation models.
    """

    def __init__(
        self,
        processed_datasets_dir: str,
        logger=None,
        histogram_manager=None,
        foundation_plot_manager=None,
    ):
        """
        Initialize the regression evaluator.

        Args:
            processed_datasets_dir: Directory for processed datasets
            logger: Logger instance (optional, will create one if not provided)
            histogram_manager: HistogramManager instance (optional, will create one if not provided)
            foundation_plot_manager: FoundationPlotManager instance (optional, will create one if not provided)
        """
        self.processed_datasets_dir = Path(processed_datasets_dir)
        self.logger = logger or get_logger(__name__)
        self.histogram_manager = histogram_manager or HistogramManager()
        self.foundation_plot_manager = (
            foundation_plot_manager or FoundationPlotManager()
        )

    def _denormalize_labels(
        self,
        normalized_labels: np.ndarray,
        norm_params: dict,
        label_config_index: int = 0,
    ) -> np.ndarray:
        """
        Denormalize label values using stored normalization parameters.

        Args:
            normalized_labels: Normalized label array of shape (n_samples, n_features)
            norm_params: Normalization parameters dictionary
            label_config_index: Index of the label configuration (default 0)

        Returns:
            Denormalized label array with original scale and units
        """
        if (
            "labels" not in norm_params
            or len(norm_params["labels"]) <= label_config_index
        ):
            self.logger.warning(
                "No label normalization parameters found, returning normalized values"
            )
            return normalized_labels

        label_norm_params = norm_params["labels"][label_config_index]
        denormalized = normalized_labels.copy()

        # For now, assume all labels are aggregated features (this is the common case)
        # If scalar features are present, they would need separate handling
        if "aggregated" in label_norm_params:
            for agg_name, params in label_norm_params["aggregated"].items():
                means = np.array(params["means"])
                stds = np.array(params["stds"])

                # Apply denormalization: denormalized = normalized * std + mean
                denormalized = denormalized * stds + means
                break  # Assuming single aggregator for labels (most common case)
        elif "scalar" in label_norm_params:
            # Handle scalar features if present
            for feature_name, params in label_norm_params["scalar"].items():
                mean = params["mean"]
                std = params["std"]
                # Apply denormalization for scalar features
                denormalized = denormalized * std + mean
                break  # Assuming single scalar feature for labels

        return denormalized

    def evaluate_regression(
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
        Evaluate foundation model for regression tasks using data efficiency study.

        This method trains three models (From Scratch, Fine-Tuned, Fixed) on increasing amounts of training data
        to show how pre-trained weights help with limited data and demonstrate the value of the foundation model.

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
                "Foundation model path must be provided for data efficiency evaluation."
            )
            return False

        foundation_model_dir = Path(foundation_model_path)
        regression_dir = foundation_model_dir / "testing" / "regression_evaluation"
        regression_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize data manager
            data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

            # 1. Load full dataset with regression labels
            self.logger.info("Loading full dataset with regression labels...")
            # train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
            #     dataset_config=dataset_config,
            #     validation_fraction=dataset_config.validation_fraction,
            #     test_fraction=dataset_config.test_fraction,
            #     batch_size=dnn_training_config.batch_size,
            #     shuffle_buffer=dataset_config.shuffle_buffer,
            #     include_labels=True,
            #     delete_catalogs=delete_catalogs,
            # )
            # Start of testing code
            signal_datasets_splits = data_manager.load_signal_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=dnn_training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=True,
                split=True,  # Enable splitting
            )
            signal_key = "wprime_taunu"

            if signal_key not in signal_datasets_splits:
                self.logger.error(f"Signal dataset '{signal_key}' not found")
                return False

            train_dataset, val_dataset, test_dataset = signal_datasets_splits[
                signal_key
            ]

            # End of testing code

            # Access normalization parameters for denormalizing labels
            import json

            import h5py

            dataset_path = data_manager.get_current_dataset_path()
            norm_params = None
            try:
                with h5py.File(dataset_path, "r") as f:
                    norm_params = json.loads(f.attrs["normalization_params"])
                self.logger.info(
                    "Successfully loaded normalization parameters for label denormalization"
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not load normalization parameters: {e}. Labels will remain normalized."
                )
                norm_params = None

            # Count total training events
            total_train_events = 0
            for batch in train_dataset:
                batch_size = tf.shape(batch[0])[0]
                total_train_events += batch_size.numpy()

            # Count total test events for plot labeling
            total_test_events = 0
            for batch in test_dataset:
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

            # 2. Load Pre-trained Foundation Encoder & its Config
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
            self.logger.info("Deterministic encoder layers:")
            for layer in pretrained_deterministic_encoder.layers:
                self.logger.info(
                    f"  {layer.name}: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}"
                )

            original_input_shape = (task_config.input.get_total_feature_size(),)
            regression_output_shape = (task_config.labels[0].get_total_feature_size(),)

            # Helper function to build regressor head
            def build_regressor_head(name_suffix: str) -> tf.keras.Model:
                # Create a copy of the DNN config with modified architecture
                import copy

                regressor_config = copy.deepcopy(dnn_model_config)

                # Update the architecture for the regressor head
                regressor_config.architecture.update(
                    {
                        "input_shape": (latent_dim,),
                        "output_shape": regression_output_shape,
                        "name": f"{dnn_model_config.architecture.get('name', 'regressor')}_{name_suffix}",
                    }
                )

                regressor_model_wrapper = ModelFactory.create_model(
                    model_type="dnn_predictor", config=regressor_config
                )
                regressor_model_wrapper.build()
                return regressor_model_wrapper.model

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
                    validation_data=val_dataset,
                    callbacks=[],  # No callbacks for speed
                    training_history_dir=regression_dir / "training_histories"
                    if save_training_history
                    else None,
                    model_name=model_name,
                    dataset_id=f"regression_eval_{data_size}",
                    experiment_id="regression_evaluation",
                    verbose_training="minimal",  # Reduce verbosity for evaluation models
                    save_individual_history=True,  # Save individual files for comparison plots
                )

                # Evaluate on test set
                test_metrics = trainer.evaluate(test_dataset)
                test_loss = test_metrics.get(
                    "test_loss", test_metrics.get("test_mse", 0.0)
                )

                # Generate predictions and save histogram data (1000 samples)
                try:
                    self.logger.info(
                        f"Generating predictions for {model_name} model..."
                    )
                    predictions_list = []
                    actual_labels_list = []
                    samples_collected = 0
                    max_prediction_samples = 1000

                    for batch in test_dataset:
                        if samples_collected >= max_prediction_samples:
                            break

                        if isinstance(batch, tuple) and len(batch) == 2:
                            features_batch, labels_batch = batch

                            predictions_batch = combined_keras_model.predict(
                                features_batch, verbose=0
                            )

                            # Extract actual labels from batch structure
                            if isinstance(labels_batch, tuple):
                                actual_labels_batch = None
                                for item in labels_batch:
                                    if hasattr(item, "shape") and hasattr(
                                        item, "numpy"
                                    ):
                                        actual_labels_batch = item.numpy()
                                        break
                                if (
                                    actual_labels_batch is None
                                    and len(labels_batch) > 0
                                ):
                                    actual_labels_batch = labels_batch[0]
                            else:
                                actual_labels_batch = (
                                    labels_batch.numpy()
                                    if hasattr(labels_batch, "numpy")
                                    else labels_batch
                                )

                            # Remove extra dimensions if present
                            if (
                                hasattr(actual_labels_batch, "ndim")
                                and actual_labels_batch.ndim == 3
                                and actual_labels_batch.shape[0] == 1
                            ):
                                actual_labels_batch = actual_labels_batch.squeeze(
                                    axis=0
                                )

                            # Convert to numpy and collect samples
                            predictions_np = np.array(predictions_batch)
                            actual_labels_np = np.array(actual_labels_batch)

                            batch_size = predictions_np.shape[0]
                            samples_to_take = min(
                                batch_size, max_prediction_samples - samples_collected
                            )

                            predictions_list.extend(predictions_np[:samples_to_take])
                            actual_labels_list.extend(
                                actual_labels_np[:samples_to_take]
                            )
                            samples_collected += samples_to_take

                    # Convert to numpy arrays
                    predictions_array = np.array(predictions_list)
                    actual_labels_array = np.array(actual_labels_list)

                    if len(predictions_array) > 0 and len(actual_labels_array) > 0:
                        # Denormalize labels and predictions to get original scale/units
                        if norm_params is not None:
                            try:
                                self.logger.info(
                                    f"Denormalizing labels and predictions for {model_name}..."
                                )
                                predictions_array = self._denormalize_labels(
                                    predictions_array, norm_params, label_config_index=0
                                )
                                actual_labels_array = self._denormalize_labels(
                                    actual_labels_array,
                                    norm_params,
                                    label_config_index=0,
                                )
                                self.logger.info(
                                    f"Successfully denormalized {len(predictions_array)} prediction/label pairs"
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to denormalize labels for {model_name}: {e}. Using normalized values."
                                )
                        else:
                            self.logger.info(
                                f"No normalization parameters available for {model_name}, using normalized values"
                            )
                        # Get label variable names (required for histogram keys)
                        if not label_variable_names_param:
                            self.logger.warning(
                                "No label variable names found, using generic names"
                            )
                            num_vars = (
                                predictions_array.shape[1]
                                if predictions_array.ndim > 1
                                else 1
                            )
                            current_label_names = [
                                f"variable_{i}" for i in range(num_vars)
                            ]
                        else:
                            current_label_names = label_variable_names_param

                        # Create data dictionaries for histogram manager (simplified - no special cases)
                        predictions_data = {}
                        differences_data = {}

                        for i, var_name in enumerate(current_label_names):
                            if i < predictions_array.shape[1]:
                                predictions_data[var_name] = predictions_array[
                                    :, i
                                ].tolist()
                                differences_data[var_name] = (
                                    predictions_array[:, i] - actual_labels_array[:, i]
                                ).tolist()

                        # Prepare save directory and file paths
                        hist_save_dir = label_distributions_dir_param or (
                            regression_dir / "label_distributions"
                        )
                        hist_save_dir.mkdir(parents=True, exist_ok=True)

                        data_size_label = (
                            f"{data_size // 1000}k"
                            if data_size >= 1000
                            else str(data_size)
                        )

                        # Extract base model name to avoid duplication (model_name already contains data_size_label)
                        # e.g., "From_Scratch_1k" -> "From_Scratch"
                        base_model_name = (
                            model_name.rsplit("_", 1)[0]
                            if "_" in model_name
                            else model_name
                        )

                        # Save predictions histogram using HistogramManager
                        pred_file_path = (
                            hist_save_dir
                            / f"{base_model_name}_{data_size_label}_predictions_hist.json"
                        )
                        self.histogram_manager.save_to_hist_file(
                            data=predictions_data,
                            file_path=pred_file_path,
                            nbins=50,
                            use_percentile_file=False,
                            update_percentile_file=False,
                            use_percentile_cache=True,  # Use cache for coordinated bins
                        )

                        # Save differences histogram using HistogramManager with separate percentile index
                        diff_file_path = (
                            hist_save_dir
                            / f"{base_model_name}_{data_size_label}_diff_predictions_hist.json"
                        )
                        self.histogram_manager.save_to_hist_file(
                            data=differences_data,
                            file_path=diff_file_path,
                            nbins=50,
                            use_percentile_file=False,
                            update_percentile_file=False,
                            use_percentile_cache=False,  # Don't use coordinated bins for differences
                        )

                        self.logger.info(
                            f"Saved histogram data for {model_name} with {len(predictions_array)} samples"
                        )
                    else:
                        self.logger.warning(
                            f"No predictions or labels generated for {model_name}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Failed to generate predictions for {model_name}: {str(e)}"
                    )

                self.logger.info(
                    f"{model_name} with {data_size} events - Test Loss: {test_loss:.6f}"
                )
                return test_loss

            # Store results for plotting
            results = {
                "data_sizes": data_sizes,
                "From_Scratch": [],
                "Fine_Tuned": [],
                "Fixed_Encoder": [],
            }

            # Save training histories for all data sizes (not just the largest)
            if not data_sizes:
                self.logger.warning(
                    "No valid data sizes remain after filtering. Using total training events as fallback."
                )
                # Will save histories for all data sizes processed

            # 3. Set up label distributions directory
            label_distributions_dir = regression_dir / "label_distributions"
            label_distributions_dir.mkdir(parents=True, exist_ok=True)

            # Extract label variable names from task config
            label_variable_names = []
            if hasattr(task_config, "labels") and task_config.labels:
                first_label_config = task_config.labels[0]
                if (
                    hasattr(first_label_config, "feature_array_aggregators")
                    and first_label_config.feature_array_aggregators
                ):
                    first_aggregator = first_label_config.feature_array_aggregators[0]
                    if hasattr(first_aggregator, "input_branches"):
                        for branch_selector in first_aggregator.input_branches:
                            if hasattr(branch_selector, "branch") and hasattr(
                                branch_selector.branch, "name"
                            ):
                                label_variable_names.append(branch_selector.branch.name)

            self.logger.info(f"Extracted label variable names: {label_variable_names}")

            # Save actual labels histogram ONCE before model training
            if label_variable_names:
                self.logger.info("Saving actual test labels histogram...")
                actual_labels_list = []
                samples_collected = 0
                max_samples = 1000

                for batch in test_dataset:
                    if samples_collected >= max_samples:
                        break

                    if isinstance(batch, tuple) and len(batch) == 2:
                        features_batch, labels_batch = batch

                        # Extract actual labels from batch structure
                        if isinstance(labels_batch, tuple):
                            actual_labels_batch = None
                            for item in labels_batch:
                                if hasattr(item, "shape") and hasattr(item, "numpy"):
                                    actual_labels_batch = item.numpy()
                                    break
                            if actual_labels_batch is None and len(labels_batch) > 0:
                                actual_labels_batch = labels_batch[0]
                        else:
                            actual_labels_batch = (
                                labels_batch.numpy()
                                if hasattr(labels_batch, "numpy")
                                else labels_batch
                            )

                        # Remove extra dimensions if present
                        if (
                            hasattr(actual_labels_batch, "ndim")
                            and actual_labels_batch.ndim == 3
                            and actual_labels_batch.shape[0] == 1
                        ):
                            actual_labels_batch = actual_labels_batch.squeeze(axis=0)

                        actual_labels_np = np.array(actual_labels_batch)
                        batch_size = actual_labels_np.shape[0]
                        samples_to_take = min(
                            batch_size, max_samples - samples_collected
                        )

                        actual_labels_list.extend(actual_labels_np[:samples_to_take])
                        samples_collected += samples_to_take

                # Convert to numpy and organize by variable names
                actual_labels_array = np.array(actual_labels_list)

                # Denormalize actual test labels to get original scale/units
                if norm_params is not None:
                    try:
                        self.logger.info("Denormalizing actual test labels...")
                        actual_labels_array = self._denormalize_labels(
                            actual_labels_array, norm_params, label_config_index=0
                        )
                        self.logger.info(
                            f"Successfully denormalized {len(actual_labels_array)} actual label samples"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to denormalize actual test labels: {e}. Using normalized values."
                        )
                else:
                    self.logger.info(
                        "No normalization parameters available, using normalized values for actual test labels"
                    )

                actual_data = {}

                for i, var_name in enumerate(label_variable_names):
                    if i < actual_labels_array.shape[1]:
                        actual_data[var_name] = actual_labels_array[:, i].tolist()

                # Save actual labels histogram
                actual_labels_path = (
                    label_distributions_dir / "actual_test_labels_hist.json"
                )
                self.histogram_manager.save_to_hist_file(
                    data=actual_data,
                    file_path=actual_labels_path,
                    nbins=50,
                    use_percentile_file=False,
                    update_percentile_file=False,
                    use_percentile_cache=True,  # Use cache for coordinated bins
                )
                self.logger.info("Saved actual test labels histogram")

            # 4. Run experiments for each data size
            for data_size_index, data_size in enumerate(data_sizes):
                self.logger.info(f"{'=' * 50}")
                self.logger.info(f"Training with {data_size} events")
                self.logger.info(f"{'=' * 50}")

                # Create subset of training data
                train_subset = create_subset_dataset(
                    train_dataset, data_size, data_size_index
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
                scratch_regressor_dnn = build_regressor_head("from_scratch")

                model_inputs = tf.keras.Input(
                    shape=original_input_shape, name="input_features"
                )
                encoded_scratch = scratch_encoder_part(model_inputs)
                predictions_scratch = scratch_regressor_dnn(encoded_scratch)
                model_from_scratch = tf.keras.Model(
                    inputs=model_inputs,
                    outputs=predictions_scratch,
                    name="Regressor_From_Scratch",
                )

                # Save training history for all data sizes
                should_save_history = True

                self.logger.info(
                    f"Enabling training history saving for all models with {data_size} events"
                )

                # Format data size for better labeling
                data_size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )

                scratch_loss = train_and_evaluate_for_size(
                    f"From_Scratch_{data_size_label}",
                    model_from_scratch,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=label_distributions_dir,
                    label_variable_names_param=label_variable_names,
                )
                results["From_Scratch"].append(scratch_loss)

                # --- Model 2: Fine-Tuned ---
                self.logger.info("Building Fine-Tuned model...")
                # Create a functional copy of the deterministic encoder for fine-tuning
                # We can't use clone_model with QKeras layers, so we'll create a new model
                # that uses the same layers but allows training
                fine_tuned_input = tf.keras.Input(
                    shape=original_input_shape, name="fine_tuned_input"
                )
                fine_tuned_encoded = pretrained_deterministic_encoder(fine_tuned_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fine_tuned_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32), name="dtype_cast_fine_tuned"
                )(fine_tuned_encoded)

                fine_tuned_encoder_part = tf.keras.Model(
                    inputs=fine_tuned_input,
                    outputs=fine_tuned_encoded_cast,
                    name="fine_tuned_pretrained_encoder",
                )
                fine_tuned_encoder_part.trainable = True

                fine_tuned_regressor_dnn = build_regressor_head("fine_tuned")

                model_inputs_ft = tf.keras.Input(
                    shape=original_input_shape, name="input_features_ft"
                )
                encoded_ft = fine_tuned_encoder_part(model_inputs_ft)
                predictions_ft = fine_tuned_regressor_dnn(encoded_ft)
                model_fine_tuned = tf.keras.Model(
                    inputs=model_inputs_ft,
                    outputs=predictions_ft,
                    name="Regressor_Fine_Tuned",
                )

                finetuned_loss = train_and_evaluate_for_size(
                    f"Fine_Tuned_{data_size_label}",
                    model_fine_tuned,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=label_distributions_dir,
                    label_variable_names_param=label_variable_names,
                )
                results["Fine_Tuned"].append(finetuned_loss)

                # --- Model 3: Fixed Encoder ---
                self.logger.info("Building Fixed Encoder model...")
                # Create a functional copy of the deterministic encoder for fixed use
                # We can't use clone_model with QKeras layers, so we'll create a new model
                # that uses the same layers but freezes them
                fixed_input = tf.keras.Input(
                    shape=original_input_shape, name="fixed_input"
                )
                fixed_encoded = pretrained_deterministic_encoder(fixed_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fixed_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32), name="dtype_cast_fixed"
                )(fixed_encoded)

                fixed_encoder_part = tf.keras.Model(
                    inputs=fixed_input,
                    outputs=fixed_encoded_cast,
                    name="fixed_pretrained_encoder",
                )
                fixed_encoder_part.trainable = False

                fixed_regressor_dnn = build_regressor_head("fixed_encoder")

                model_inputs_fx = tf.keras.Input(
                    shape=original_input_shape, name="input_features_fx"
                )
                encoded_fx = fixed_encoder_part(model_inputs_fx)
                predictions_fx = fixed_regressor_dnn(encoded_fx)
                model_fixed = tf.keras.Model(
                    inputs=model_inputs_fx,
                    outputs=predictions_fx,
                    name="Regressor_Fixed_Encoder",
                )

                fixed_loss = train_and_evaluate_for_size(
                    f"Fixed_Encoder_{data_size_label}",
                    model_fixed,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=label_distributions_dir,
                    label_variable_names_param=label_variable_names,
                )
                results["Fixed_Encoder"].append(fixed_loss)

            # 5. Save results and create plots
            self.logger.info("Creating data efficiency plot...")

            # Save results to JSON
            results_file = regression_dir / "regression_data_efficiency_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {results_file}")

            # Create combined training history plot if we saved training histories
            training_histories_dir = regression_dir / "training_histories"
            combined_plot_path = regression_dir / "regression_training_comparison.png"
            self.foundation_plot_manager.create_training_history_comparison_plot_from_directory(
                training_histories_dir,
                combined_plot_path,
                title_prefix="Regression Model Training Comparison",
                validation_only=True,
            )

            # Create the data efficiency plot
            plot_file = regression_dir / "regression_data_efficiency_plot.png"
            self.foundation_plot_manager.create_data_efficiency_plot(
                results,
                plot_file,
                plot_type="regression",
                metric_name="Test Loss (MSE)",
                total_test_events=total_test_events,
            )

            # Create label distribution comparison plot
            if label_variable_names:
                # Load physlite plot labels for proper titles
                physlite_plot_labels = None
                try:
                    physlite_labels_path = Path(
                        "src/hep_foundation/data/physlite_plot_labels.json"
                    )
                    if physlite_labels_path.exists():
                        with open(physlite_labels_path) as f:
                            physlite_plot_labels = json.load(f)
                except Exception as e:
                    self.logger.warning(
                        f"Could not load physlite plot labels: {str(e)}"
                    )

                # Use the subplot-based comparison plot
                self.foundation_plot_manager.create_label_distribution_comparison_plot_with_subplots(
                    regression_dir,
                    data_sizes,
                    label_variable_names,
                    physlite_plot_labels,
                )
            else:
                self.logger.warning(
                    "No label variable names found, skipping label distribution comparison plot"
                )

            # 5. Display summary
            log_evaluation_summary(results, evaluation_type="regression")

            return True

        except Exception as e:
            self.logger.error(
                f"Regression evaluation failed: {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Detailed traceback:")
            return False
