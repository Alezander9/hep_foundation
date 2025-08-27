import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
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
from hep_foundation.plots.histogram_manager import HistogramManager


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
        self.downstream_manager = DownstreamModelManager(logger=self.logger)

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
        dataset: str = "atlas",
        seed: Optional[int] = None,
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
            dataset: Dataset to use for regression, either "atlas" or a signal key (e.g., "wprime_taunu")
            seed: Random seed for reproducible weight initialization (optional)
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
            if dataset == "atlas":
                self.logger.info("Loading ATLAS dataset with regression labels...")
                train_dataset, val_dataset, test_dataset = (
                    data_manager.load_atlas_datasets(
                        dataset_config=dataset_config,
                        validation_fraction=dataset_config.validation_fraction,
                        test_fraction=dataset_config.test_fraction,
                        batch_size=dnn_training_config.batch_size,
                        shuffle_buffer=dataset_config.shuffle_buffer,
                        include_labels=True,
                        delete_catalogs=delete_catalogs,
                    )
                )
            else:
                # Load signal dataset
                self.logger.info(
                    f"Loading signal dataset '{dataset}' with regression labels..."
                )
                signal_datasets_splits = data_manager.load_signal_datasets(
                    dataset_config=dataset_config,
                    validation_fraction=dataset_config.validation_fraction,
                    test_fraction=dataset_config.test_fraction,
                    batch_size=dnn_training_config.batch_size,
                    shuffle_buffer=dataset_config.shuffle_buffer,
                    include_labels=True,
                    split=True,
                )

                if dataset not in signal_datasets_splits:
                    self.logger.error(
                        f"Signal dataset '{dataset}' not found in available datasets: {list(signal_datasets_splits.keys())}"
                    )
                    return False

                train_dataset, val_dataset, test_dataset = signal_datasets_splits[
                    dataset
                ]

            # Access normalization parameters for denormalizing labels

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
            self.logger.info("Deterministic encoder layers:")
            for layer in pretrained_deterministic_encoder.layers:
                self.logger.info(
                    f"  {layer.name}: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}"
                )

            original_input_shape = (task_config.input.get_total_feature_size(),)
            regression_output_shape = (task_config.labels[0].get_total_feature_size(),)

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

                # Check if we actually collected any label data
                if len(actual_labels_list) == 0:
                    self.logger.error(
                        "No label data was collected from the test dataset. This might indicate:"
                    )
                    self.logger.error(
                        "1. The signal dataset doesn't contain the required label branches"
                    )
                    self.logger.error("2. No events passed the selection criteria")
                    self.logger.error(
                        f"Expected label branches: {label_variable_names}"
                    )
                    return False

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
                    if (
                        len(actual_labels_array.shape) > 1
                        and i < actual_labels_array.shape[1]
                    ):
                        actual_data[var_name] = actual_labels_array[:, i].tolist()
                    elif len(actual_labels_array.shape) == 1 and i == 0:
                        # Handle case where we have a 1D array (single label variable)
                        actual_data[var_name] = actual_labels_array.tolist()
                    # Skip variables that don't have corresponding data

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

            # 4. Run data efficiency study for regression
            from hep_foundation.pipeline.downstream_model_manager import (
                DownstreamModelType,
            )

            # Initialize results dictionary
            results = {
                "data_sizes": data_sizes,
                "From_Scratch": [],
                "Fine_Tuned": [],
                "Fixed_Encoder": [],
            }

            # Run experiments for each data size
            for data_size_index, data_size in enumerate(data_sizes):
                self.logger.info(f"{'=' * 50}")
                self.logger.info(f"Training with {data_size} events")
                self.logger.info(f"{'=' * 50}")

                # Create subset of training data
                train_subset = self.downstream_manager.create_subset_dataset(
                    train_dataset,
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
                        output_shape=regression_output_shape,
                        pretrained_encoder=pretrained_deterministic_encoder,
                        encoder_hidden_layers=encoder_hidden_layers,
                        latent_dim=latent_dim,
                        dnn_model_config=dnn_model_config,
                        encoder_activation=encoder_activation,
                        output_activation=None,
                    )

                    # Train and evaluate
                    model_name = f"{model_type.value.replace('_', ' ').title().replace(' ', '_')}_{data_size_label}"

                    training_config = {
                        "batch_size": dnn_training_config.batch_size,
                        "learning_rate": dnn_training_config.learning_rate,
                    }

                    evaluation_results = (
                        self.downstream_manager.train_and_evaluate_model(
                            model=model,
                            model_name=model_name,
                            train_dataset=train_subset,
                            val_dataset=val_dataset,
                            test_dataset=test_dataset,
                            training_config=training_config,
                            fixed_epochs=fixed_epochs,
                            data_size=data_size,
                            output_dir=regression_dir,
                            save_training_history=True,
                            return_accuracy=False,
                            seed=seed,
                        )
                    )

                    # Store results
                    test_loss = evaluation_results[0]
                    model_key = (
                        model_type.value.replace("_", " ").title().replace(" ", "_")
                    )
                    results[model_key].append(test_loss)

                    # Generate predictions and save histogram data for regression
                    if label_variable_names:
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

                                    predictions_batch = model.predict(
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
                                        actual_labels_batch = (
                                            actual_labels_batch.squeeze(axis=0)
                                        )

                                    # Convert to numpy and collect samples
                                    predictions_np = np.array(predictions_batch)
                                    actual_labels_np = np.array(actual_labels_batch)

                                    batch_size = predictions_np.shape[0]
                                    samples_to_take = min(
                                        batch_size,
                                        max_prediction_samples - samples_collected,
                                    )

                                    predictions_list.extend(
                                        predictions_np[:samples_to_take]
                                    )
                                    actual_labels_list.extend(
                                        actual_labels_np[:samples_to_take]
                                    )
                                    samples_collected += samples_to_take

                            # Convert to numpy arrays
                            predictions_array = np.array(predictions_list)
                            actual_labels_array = np.array(actual_labels_list)

                            if (
                                len(predictions_array) > 0
                                and len(actual_labels_array) > 0
                            ):
                                # Denormalize labels and predictions to get original scale/units
                                if norm_params is not None:
                                    try:
                                        self.logger.info(
                                            f"Denormalizing labels and predictions for {model_name}..."
                                        )
                                        predictions_array = self._denormalize_labels(
                                            predictions_array,
                                            norm_params,
                                            label_config_index=0,
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

                                # Create data dictionaries for histogram manager
                                predictions_data = {}
                                differences_data = {}

                                for i, var_name in enumerate(label_variable_names):
                                    if i < predictions_array.shape[1]:
                                        predictions_data[var_name] = predictions_array[
                                            :, i
                                        ].tolist()
                                        differences_data[var_name] = (
                                            predictions_array[:, i]
                                            - actual_labels_array[:, i]
                                        ).tolist()

                                # Extract base model name to avoid duplication
                                base_model_name = (
                                    model_name.rsplit("_", 1)[0]
                                    if "_" in model_name
                                    else model_name
                                )

                                # Save predictions histogram
                                pred_file_path = (
                                    label_distributions_dir
                                    / f"{base_model_name}_{data_size_label}_predictions_hist.json"
                                )
                                self.histogram_manager.save_to_hist_file(
                                    data=predictions_data,
                                    file_path=pred_file_path,
                                    nbins=50,
                                    use_percentile_file=False,
                                    update_percentile_file=False,
                                    use_percentile_cache=True,
                                )

                                # Save differences histogram
                                diff_file_path = (
                                    label_distributions_dir
                                    / f"{base_model_name}_{data_size_label}_diff_predictions_hist.json"
                                )
                                self.histogram_manager.save_to_hist_file(
                                    data=differences_data,
                                    file_path=diff_file_path,
                                    nbins=50,
                                    use_percentile_file=False,
                                    update_percentile_file=False,
                                    use_percentile_cache=False,
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

            # 5. Save results and create plots
            self.logger.info("Creating data efficiency plot...")

            # Save results to JSON
            results_file = regression_dir / "regression_data_efficiency_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {results_file}")

            # Create combined training history and data efficiency plot
            training_histories_dir = regression_dir / "training_histories"
            combined_plot_path = regression_dir / "regression_evaluation_combined.png"
            self.foundation_plot_manager.create_combined_downstream_evaluation_plot(
                training_histories_dir,
                results,
                combined_plot_path,
                plot_type="regression",
                metric_name="Test Loss (MSE)",
                total_test_events=total_test_events,
                title_prefix="Regression Evaluation: Training History & Data Efficiency",
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
