"""
Standalone Regression Pipeline for HEP Foundation Project.

This module provides a complete pipeline for training and evaluating standalone DNN models
for regression tasks without requiring foundation model pretraining.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.standalone_config import (
    StandaloneEvaluationConfig,
    StandaloneTrainingConfig,
    load_standalone_config,
)
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.models.standalone_dnn_regressor import (
    StandaloneDNNConfig,
    StandaloneDNNRegressor,
)
from hep_foundation.plots.standalone_plot_manager import StandalonePlotManager
from hep_foundation.training.standalone_trainer import StandaloneTrainer


class StandaloneRegressionPipeline:
    """
    Complete pipeline for standalone DNN regression experiments.

    Handles data loading, model creation, training, evaluation, and visualization
    for standalone regression tasks without foundation model dependencies.
    """

    def __init__(
        self,
        processed_datasets_dir: Path = Path("_processed_datasets"),
        experiments_output_dir: Path = Path("_standalone_experiments"),
    ):
        """
        Initialize StandaloneRegressionPipeline.

        Args:
            processed_datasets_dir: Directory for processed datasets
            experiments_output_dir: Directory for experiment outputs
        """
        self.logger = get_logger(__name__)
        self.processed_datasets_dir = processed_datasets_dir
        self.experiments_output_dir = experiments_output_dir
        self.plot_manager = StandalonePlotManager()

        # Create output directory
        self.experiments_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("StandaloneRegressionPipeline initialized")
        self.logger.info(f"Processed datasets directory: {self.processed_datasets_dir}")
        self.logger.info(f"Experiments output directory: {self.experiments_output_dir}")

    def run_complete_pipeline(
        self,
        config_dict: dict[str, Any],
        delete_catalogs: bool = False,
    ) -> bool:
        """
        Run the complete standalone regression pipeline.

        Args:
            config_dict: Configuration dictionary from YAML
            delete_catalogs: Whether to delete data catalogs after processing

        Returns:
            True if pipeline succeeded, False otherwise
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING STANDALONE REGRESSION PIPELINE")
        self.logger.info("=" * 80)

        try:
            # Load configuration
            config = load_standalone_config(config_dict)
            training_config = config["training_config"]
            evaluation_config = config["evaluation_config"]
            model_config_dict = config["model_config_dict"]

            # Create experiment directory
            experiment_id = self._generate_experiment_id(config["metadata"]["name"])
            experiment_dir = self.experiments_output_dir / experiment_id
            experiment_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Experiment ID: {experiment_id}")
            self.logger.info(f"Experiment directory: {experiment_dir}")

            # Save configuration for reproducibility
            self._save_experiment_config(experiment_dir, config_dict, config)

            # Step 1: Load or create dataset
            dataset_manager = DatasetManager(base_dir=self.processed_datasets_dir)
            dataset_config, task_config = self._extract_dataset_configs(config)

            train_dataset, val_dataset, test_dataset = self._load_datasets(
                dataset_manager, dataset_config, task_config, delete_catalogs
            )

            # Step 2: Create and configure model
            model = self._create_model(model_config_dict, train_dataset)

            # Step 3: Run regression evaluation across different data sizes
            success = self._run_regression_evaluation(
                model,
                training_config,
                evaluation_config,
                train_dataset,
                val_dataset,
                test_dataset,
                experiment_dir,
            )

            if success:
                self.logger.info("=" * 80)
                self.logger.info(
                    "STANDALONE REGRESSION PIPELINE COMPLETED SUCCESSFULLY"
                )
                self.logger.info("=" * 80)
                self.logger.info(f"Results saved to: {experiment_dir}")
            else:
                self.logger.error("STANDALONE REGRESSION PIPELINE FAILED")

            return success

        except Exception as e:
            self.logger.error(f"Pipeline failed: {type(e).__name__}: {str(e)}")
            self.logger.exception("Detailed traceback:")
            return False

    def _generate_experiment_id(self, experiment_name: str) -> str:
        """Generate unique experiment ID."""
        # Find next available number
        existing_dirs = [d for d in self.experiments_output_dir.iterdir() if d.is_dir()]
        existing_numbers = []

        for d in existing_dirs:
            parts = d.name.split("_")
            if parts and parts[0].isdigit():
                existing_numbers.append(int(parts[0]))

        next_number = max(existing_numbers, default=0) + 1
        return f"{next_number:03d}_Standalone_{experiment_name.replace(' ', '_')}"

    def _save_experiment_config(
        self,
        experiment_dir: Path,
        original_config: dict[str, Any],
        processed_config: dict[str, Any],
    ) -> None:
        """Save experiment configuration for reproducibility."""
        try:
            # Save original config
            config_path = experiment_dir / "_experiment_config.yaml"
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(original_config, f, default_flow_style=False, indent=2)

            # Save experiment info
            info_path = experiment_dir / "_experiment_info.json"
            experiment_info = {
                "experiment_type": "standalone_regression",
                "timestamp": str(datetime.now()),
                "metadata": processed_config["metadata"],
                "training_config": processed_config["training_config"].to_dict(),
                "evaluation_config": processed_config["evaluation_config"].to_dict(),
                "model_config": processed_config["model_config_dict"],
            }

            with open(info_path, "w") as f:
                json.dump(experiment_info, f, indent=2)

            self.logger.info(f"Experiment configuration saved to: {config_path}")
            self.logger.info(f"Experiment info saved to: {info_path}")

        except Exception as e:
            self.logger.error(f"Failed to save experiment configuration: {e}")

    def _extract_dataset_configs(self, config: dict[str, Any]) -> tuple[Any, Any]:
        """Extract dataset and task configurations."""
        # Import here to avoid circular imports
        from hep_foundation.config.config_loader import load_pipeline_config

        # Create temporary config for existing pipeline components
        temp_config = {
            "dataset": config["dataset_settings"],
            "task": config["task_settings"],
        }

        # Use existing config loader functionality
        pipeline_config = load_pipeline_config(temp_config)
        return pipeline_config["dataset_config"], pipeline_config["task_config"]

    def _load_datasets(
        self,
        dataset_manager: DatasetManager,
        dataset_config: Any,
        task_config: Any,
        delete_catalogs: bool,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load or create datasets."""
        self.logger.info("Loading datasets...")

        train_dataset, val_dataset, test_dataset = dataset_manager.load_atlas_datasets(
            dataset_config=dataset_config,
            validation_fraction=dataset_config.validation_fraction,
            test_fraction=dataset_config.test_fraction,
            batch_size=1024,  # Will be rebatched during training
            shuffle_buffer=dataset_config.shuffle_buffer,
            include_labels=True,
            delete_catalogs=delete_catalogs,
        )

        self.logger.info("Datasets loaded successfully")
        return train_dataset, val_dataset, test_dataset

    def _create_model(
        self,
        model_config_dict: dict[str, Any],
        train_dataset: tf.data.Dataset,
    ) -> StandaloneDNNRegressor:
        """Create and build standalone DNN model."""
        self.logger.info("Creating standalone DNN model...")

        # Determine input and output shapes from dataset
        for batch in train_dataset:
            if isinstance(batch, tuple):
                features, labels = batch
                input_shape = features.shape[1:]
                if isinstance(labels, (list, tuple)):
                    # Multiple label sets - use the first one
                    output_shape = labels[0].shape[1:]
                else:
                    output_shape = labels.shape[1:]
                break
        else:
            raise ValueError("Unable to determine input/output shapes from dataset")

        self.logger.info(f"Determined input shape: {input_shape}")
        self.logger.info(f"Determined output shape: {output_shape}")

        # Create model config
        model_config = StandaloneDNNConfig(
            model_type="standalone_dnn_regressor",
            architecture={
                "input_shape": input_shape,
                "output_shape": output_shape,
                "hidden_layers": model_config_dict["architecture"]["hidden_layers"],
                "activation": model_config_dict["architecture"].get(
                    "activation", "relu"
                ),
                "output_activation": model_config_dict["architecture"].get(
                    "output_activation", "linear"
                ),
                "name": model_config_dict["architecture"].get("name", "standalone_dnn"),
            },
            hyperparameters={
                "dropout_rate": model_config_dict["hyperparameters"].get(
                    "dropout_rate", 0.0
                ),
                "l2_regularization": model_config_dict["hyperparameters"].get(
                    "l2_regularization", 0.0
                ),
                "batch_normalization": model_config_dict["hyperparameters"].get(
                    "batch_normalization", False
                ),
            },
        )

        # Create and build model
        model = StandaloneDNNRegressor(config=model_config)
        model.build(input_shape)

        self.logger.info("Standalone DNN model created successfully")
        self.logger.info(f"Total parameters: {model.model.count_params():,}")

        return model

    def _run_regression_evaluation(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        evaluation_config: StandaloneEvaluationConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        experiment_dir: Path,
    ) -> bool:
        """Run two-stage regression evaluation: main model + data efficiency study."""
        self.logger.info("Starting two-stage regression evaluation...")

        # Create directories
        eval_dir = experiment_dir / "testing" / "regression_evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        main_model_dir = experiment_dir / "models" / "main_model"
        main_model_dir.mkdir(parents=True, exist_ok=True)

        # Count total training events
        total_train_events = sum(1 for _ in train_dataset.unbatch())
        self.logger.info(f"Total training events available: {total_train_events}")

        # Stage 1: Train main model with full data
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: TRAINING MAIN MODEL WITH FULL DATA")
        self.logger.info("=" * 60)

        main_model_success, main_results, main_predictions = self._train_main_model(
            model,
            training_config,
            train_dataset,
            val_dataset,
            test_dataset,
            total_train_events,
            main_model_dir,
            eval_dir,
        )

        if not main_model_success:
            self.logger.error("Main model training failed")
            return False

        # Stage 2: Data efficiency study
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: DATA EFFICIENCY STUDY")
        self.logger.info("=" * 60)

        efficiency_success, all_results, all_predictions = (
            self._run_data_efficiency_study(
                model,
                training_config,
                evaluation_config,
                train_dataset,
                val_dataset,
                test_dataset,
                total_train_events,
                eval_dir,
            )
        )

        # Create comprehensive plots (including main model)
        self._create_comprehensive_plots(
            main_results,
            main_predictions,
            all_results,
            all_predictions,
            eval_dir,
            evaluation_config,
            total_train_events,
        )

        # Save all results
        self._save_comprehensive_results(main_results, all_results, eval_dir)

        self.logger.info("Two-stage regression evaluation completed successfully")
        return True

    def _train_and_evaluate_single_size(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        evaluation_config: StandaloneEvaluationConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        data_size: int,
        eval_dir: Path,
    ) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
        """Train and evaluate model for a single data size."""
        try:
            # Create fresh model instance (reset weights)
            fresh_model = StandaloneDNNRegressor(model.get_config())
            fresh_model.build(model.input_shape)

            # Create trainer with fixed epochs
            eval_training_config = StandaloneTrainingConfig(
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
                epochs=evaluation_config.fixed_epochs,
                early_stopping_patience=evaluation_config.fixed_epochs
                + 1,  # Disable early stopping
                early_stopping_min_delta=0,
                plot_training=training_config.plot_training,
                gradient_clip_norm=training_config.gradient_clip_norm,
                lr_scheduler=training_config.lr_scheduler,
            )

            trainer = StandaloneTrainer(fresh_model.model, eval_training_config)

            # Create data subset
            train_subset = (
                train_dataset.unbatch()
                .take(data_size)
                .batch(training_config.batch_size)
            )

            # Train model
            model_name = f"standalone_dnn_{data_size}"
            success = trainer.train(
                dataset=train_subset,
                validation_data=val_dataset,
                training_history_dir=eval_dir / "training_histories",
                model_name=model_name,
                dataset_id=f"{data_size}_events",
                experiment_id="standalone_regression",
                save_individual_history=True,
            )

            if not success:
                return False, {}, {}

            # Evaluate on test set
            test_results = trainer.evaluate(test_dataset)

            # Generate predictions for analysis
            predictions = trainer.predict(test_dataset)

            # Extract true labels from test dataset
            true_labels = []
            for batch in test_dataset:
                if isinstance(batch, tuple):
                    _, labels = batch
                    if isinstance(labels, (list, tuple)):
                        true_labels.append(labels[0].numpy())  # Use first label set
                    else:
                        true_labels.append(labels.numpy())

            true_labels = np.concatenate(true_labels, axis=0)

            # Prepare results
            results = {
                "data_size": data_size,
                "test_metrics": test_results,
                "training_history": trainer.get_training_history(),
            }

            predictions_dict = {
                "predictions": predictions,
                "targets": true_labels,
            }

            return True, results, predictions_dict

        except Exception as e:
            self.logger.error(f"Training failed for data size {data_size}: {e}")
            return False, {}, {}

    def _create_comprehensive_plots(
        self,
        main_results: dict[str, Any],
        main_predictions: dict[str, np.ndarray],
        all_results: dict[int, dict[str, Any]],
        all_predictions: dict[int, dict[str, np.ndarray]],
        eval_dir: Path,
        evaluation_config: StandaloneEvaluationConfig,
        total_train_events: int,
    ) -> None:
        """Create comprehensive plots including main model and efficiency study."""
        self.logger.info("Creating comprehensive evaluation plots...")

        plots_dir = eval_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Main model detailed analysis
            if main_predictions:
                self.plot_manager.create_prediction_quality_plot(
                    main_predictions["predictions"],
                    main_predictions["targets"],
                    plots_dir / "main_model_prediction_quality.png",
                    f"Main Model Prediction Quality ({total_train_events} events)",
                    evaluation_config.prediction_sample_size,
                )

                self.plot_manager.create_error_analysis_plot(
                    main_predictions["predictions"],
                    main_predictions["targets"],
                    plots_dir / "main_model_error_analysis.png",
                    f"Main Model Error Analysis ({total_train_events} events)",
                    evaluation_config.error_analysis_bins,
                )

            # Main model training history
            if "training_history" in main_results:
                self.plot_manager.create_training_history_plot(
                    main_results["training_history"],
                    plots_dir / "main_model_training_history.png",
                    f"Main Model Training History ({total_train_events} events)",
                )

            # Data efficiency analysis (if available)
            if all_results:
                # Add main model to efficiency comparison
                efficiency_data = {
                    str(data_size): results["test_metrics"]
                    for data_size, results in all_results.items()
                }
                # Add main model as the largest data point
                efficiency_data[str(total_train_events)] = main_results["test_metrics"]

                self.plot_manager.create_data_efficiency_plot(
                    efficiency_data,
                    plots_dir / "data_efficiency_with_main_model.png",
                    "Data Efficiency (Including Main Model)",
                )

                # Multi-size prediction comparison (efficiency models only)
                if evaluation_config.create_detailed_plots and all_predictions:
                    self.plot_manager.create_multi_size_comparison_plot(
                        all_predictions,
                        plots_dir / "efficiency_models_comparison.png",
                        "Data Efficiency Models Comparison",
                    )

            # Summary comparison plot
            self._create_summary_comparison_plot(
                main_results, all_results, plots_dir, total_train_events
            )

        except Exception as e:
            self.logger.error(f"Failed to create some comprehensive plots: {e}")

    def _create_summary_comparison_plot(
        self,
        main_results: dict[str, Any],
        all_results: dict[int, dict[str, Any]],
        plots_dir: Path,
        total_train_events: int,
    ) -> None:
        """Create a summary comparison plot showing main model vs efficiency models."""
        try:
            import matplotlib.pyplot as plt

            from hep_foundation.plots.standalone_plot_manager import (
                FONT_SIZES,
                LINE_WIDTHS,
                MARKER_SIZES,
                get_color_cycle,
                get_figure_size,
            )

            fig, ax = plt.subplots(figsize=get_figure_size("single", ratio=0.8))
            colors = get_color_cycle("high_contrast", 2)

            # Plot efficiency study points
            if all_results:
                data_sizes = sorted(all_results.keys())
                test_losses = [
                    all_results[size]["test_metrics"].get(
                        "test_loss",
                        all_results[size]["test_metrics"].get("test_mse", 0.0),
                    )
                    for size in data_sizes
                ]

                ax.plot(
                    data_sizes,
                    test_losses,
                    color=colors[0],
                    linewidth=LINE_WIDTHS["thick"],
                    marker="o",
                    markersize=MARKER_SIZES["normal"],
                    label="Efficiency Study Models",
                    alpha=0.7,
                )

            # Plot main model point
            main_loss = main_results["test_metrics"].get(
                "test_loss", main_results["test_metrics"].get("test_mse", 0.0)
            )
            ax.plot(
                [total_train_events],
                [main_loss],
                color=colors[1],
                marker="*",
                markersize=MARKER_SIZES["large"],
                label="Main Model (Full Data)",
                linewidth=0,
            )

            # Formatting
            ax.set_xlabel("Training Data Size", fontsize=FONT_SIZES["normal"])
            ax.set_ylabel("Test Loss (MSE)", fontsize=FONT_SIZES["normal"])
            ax.set_title(
                "Performance Summary: Main Model vs Data Efficiency",
                fontsize=FONT_SIZES["large"],
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=FONT_SIZES["small"])
            ax.set_xscale("log")
            ax.set_yscale("log")

            plt.tight_layout()
            plt.savefig(
                plots_dir / "performance_summary.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            self.logger.info("Created performance summary plot")

        except Exception as e:
            self.logger.error(f"Failed to create summary comparison plot: {e}")

    def _save_comprehensive_results(
        self,
        main_results: dict[str, Any],
        all_results: dict[int, dict[str, Any]],
        eval_dir: Path,
    ) -> None:
        """Save comprehensive results including main model and efficiency study."""
        try:
            # Save main model results
            main_results_file = eval_dir / "main_model_results.json"
            main_serializable = {
                "model_type": main_results.get("model_type", "main_model"),
                "data_size": main_results.get("data_size", "unknown"),
                "test_metrics": main_results.get("test_metrics", {}),
                "total_parameters": main_results.get("total_parameters", 0),
                "final_training_metrics": {
                    metric: values[-1] if values else None
                    for metric, values in main_results.get(
                        "training_history", {}
                    ).items()
                },
            }

            with open(main_results_file, "w") as f:
                json.dump(main_serializable, f, indent=2)

            # Save efficiency study results
            if all_results:
                efficiency_results_file = eval_dir / "data_efficiency_results.json"
                efficiency_serializable = {}
                for data_size, results in all_results.items():
                    efficiency_serializable[str(data_size)] = {
                        "data_size": results["data_size"],
                        "test_metrics": results["test_metrics"],
                        "final_training_metrics": {
                            metric: values[-1] if values else None
                            for metric, values in results["training_history"].items()
                        },
                    }

                with open(efficiency_results_file, "w") as f:
                    json.dump(efficiency_serializable, f, indent=2)

            # Save combined summary
            summary_file = eval_dir / "evaluation_summary.json"
            summary = {
                "experiment_type": "two_stage_standalone_regression",
                "main_model": main_serializable,
                "data_efficiency_study": efficiency_serializable if all_results else {},
                "summary_statistics": {
                    "main_model_performance": main_results.get("test_metrics", {}),
                    "efficiency_study_sizes": list(all_results.keys())
                    if all_results
                    else [],
                    "total_models_trained": 1 + len(all_results),
                },
            }

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Main model results saved to: {main_results_file}")
            if all_results:
                self.logger.info(
                    f"Efficiency study results saved to: {efficiency_results_file}"
                )
            self.logger.info(f"Evaluation summary saved to: {summary_file}")

        except Exception as e:
            self.logger.error(f"Failed to save comprehensive results: {e}")

    def _train_main_model(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        total_train_events: int,
        main_model_dir: Path,
        eval_dir: Path,
    ) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
        """Train main model with full training data."""
        try:
            # Create fresh model instance
            main_model = StandaloneDNNRegressor(model.get_config())
            main_model.build(model.input_shape)

            # Create trainer for main model
            trainer = StandaloneTrainer(main_model.model, training_config)

            # Train with full dataset
            self.logger.info(f"Training main model with {total_train_events} events...")
            success = trainer.train(
                dataset=train_dataset,
                validation_data=val_dataset,
                training_history_dir=eval_dir / "training_histories",
                model_name="main_model",
                dataset_id=f"full_{total_train_events}_events",
                experiment_id="standalone_regression_main",
                save_individual_history=True,
            )

            if not success:
                return False, {}, {}

            # Save main model
            model_save_path = main_model_dir / "model_weights.h5"
            trainer.save_model(model_save_path)

            # Save model config
            config_save_path = main_model_dir / "model_config.json"
            with open(config_save_path, "w") as f:
                json.dump(main_model.get_config(), f, indent=2)

            # Evaluate on test set
            test_results = trainer.evaluate(test_dataset)

            # Generate predictions for analysis
            predictions = trainer.predict(test_dataset)

            # Extract true labels from test dataset
            true_labels = []
            for batch in test_dataset:
                if isinstance(batch, tuple):
                    _, labels = batch
                    if isinstance(labels, (list, tuple)):
                        true_labels.append(labels[0].numpy())
                    else:
                        true_labels.append(labels.numpy())

            true_labels = np.concatenate(true_labels, axis=0)

            # Prepare results
            results = {
                "model_type": "main_model",
                "data_size": total_train_events,
                "test_metrics": test_results,
                "training_history": trainer.get_training_history(),
                "total_parameters": main_model.model.count_params(),
            }

            predictions_dict = {
                "predictions": predictions,
                "targets": true_labels,
            }

            self.logger.info("Main model training completed successfully")
            self.logger.info(f"Test metrics: {test_results}")

            return True, results, predictions_dict

        except Exception as e:
            self.logger.error(f"Main model training failed: {e}")
            return False, {}, {}

    def _run_data_efficiency_study(
        self,
        model: StandaloneDNNRegressor,
        training_config: StandaloneTrainingConfig,
        evaluation_config: StandaloneEvaluationConfig,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        total_train_events: int,
        eval_dir: Path,
    ) -> tuple[bool, dict[int, dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
        """Run data efficiency study with different training data sizes."""

        # Filter data sizes to available events
        valid_data_sizes = [
            size
            for size in evaluation_config.regression_data_sizes
            if size <= total_train_events
        ]
        self.logger.info(f"Data efficiency study sizes: {valid_data_sizes}")

        if not valid_data_sizes:
            self.logger.warning("No valid data sizes for efficiency study")
            return True, {}, {}  # Not a failure, just skip efficiency study

        # Store results for all data sizes
        all_results = {}
        all_predictions = {}

        # Train and evaluate for each data size
        for data_size in valid_data_sizes:
            self.logger.info(f"Training efficiency model with {data_size} events...")

            success, results, predictions = self._train_and_evaluate_single_size(
                model,
                training_config,
                evaluation_config,
                train_dataset,
                val_dataset,
                test_dataset,
                data_size,
                eval_dir,
            )

            if success:
                all_results[data_size] = results
                all_predictions[data_size] = predictions
            else:
                self.logger.warning(
                    f"Efficiency training failed for data size {data_size}"
                )

        return True, all_results, all_predictions
