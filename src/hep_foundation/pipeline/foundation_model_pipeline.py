import shutil
from pathlib import Path
from typing import Optional

import tensorflow as tf

from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.models.dnn_predictor import DNNPredictorConfig
from hep_foundation.models.variational_autoencoder import (
    VAEConfig,
)
from hep_foundation.pipeline.anomaly_detection_evaluator import (
    AnomalyDetectionEvaluator,
)
from hep_foundation.pipeline.foundation_model_trainer import FoundationModelTrainer
from hep_foundation.pipeline.regression_evaluator import RegressionEvaluator
from hep_foundation.pipeline.signal_classification_evaluator import (
    SignalClassificationEvaluator,
)
from hep_foundation.plots.foundation_plot_manager import FoundationPlotManager
from hep_foundation.plots.histogram_manager import HistogramManager
from hep_foundation.utils.utils import print_system_usage


class FoundationModelPipeline:
    """
    Pipeline for training and evaluating foundation models.

    This class provides methods for:
    1. Training foundation models
    2. Evaluating foundation models for anomaly detection
    3. Evaluating foundation models for regression tasks
    """

    def __init__(
        self,
        experiments_output_dir: str = "_foundation_experiments",
        processed_data_parent_dir: Optional[str] = None,
    ):
        """
        Initialize the foundation model pipeline.

        Args:
            experiments_output_dir: Base directory for storing individual experiment results.
            processed_data_parent_dir: Parent directory for '_processed_datasets'.
                                       If None, '_processed_datasets' is at the workspace root.
        """
        self.logger = get_logger(__name__)

        self.experiments_output_dir = Path(experiments_output_dir)
        self.experiments_output_dir.mkdir(parents=True, exist_ok=True)

        if processed_data_parent_dir is None:
            # Default for script runs: datasets are at the root level in '_processed_datasets'
            self.processed_datasets_dir = Path("_processed_datasets")
        else:
            # For tests or if specified: datasets are relative to this given parent
            self.processed_datasets_dir = (
                Path(processed_data_parent_dir) / "_processed_datasets"
            )

        self.processed_datasets_dir.mkdir(parents=True, exist_ok=True)

        # Source config file for reproducibility
        self._source_config_file = None

        # Initialize plot managers and histogram manager
        self.foundation_plot_manager = FoundationPlotManager()
        self.histogram_manager = HistogramManager()

        self.logger.info("Foundation Model Pipeline initialized.")
        self.logger.info(
            f"  Experiment outputs will be in: {self.experiments_output_dir.absolute()}"
        )
        self.logger.info(
            f"  Processed datasets will be in: {self.processed_datasets_dir.absolute()}"
        )
        self.logger.info(
            f"TensorFlow: {tf.__version__} (Eager: {tf.executing_eagerly()})"
        )

    def set_source_config_file(self, config_file_path: str):
        """
        Set the source config file path for reproducibility.

        Args:
            config_file_path: Path to the YAML config file used for this pipeline run
        """
        self._source_config_file = config_file_path
        self.logger.info(f"Source config file set to: {config_file_path}")

    def run_full_pipeline(
        self,
        dataset_config: DatasetConfig,
        task_config: TaskConfig,
        vae_model_config: VAEConfig,
        dnn_model_config: DNNPredictorConfig,
        vae_training_config: TrainingConfig,
        dnn_training_config: TrainingConfig,
        delete_catalogs: bool = True,
        data_sizes: list = None,
        fixed_epochs: int = None,
    ) -> bool:
        """
        Run the complete foundation model pipeline: train → regression → anomaly.

        This method runs all three processes sequentially, using the trained model
        from the training phase for both evaluation tasks.

        Args:
            dataset_config: Configuration for dataset processing
            task_config: Configuration for task processing
            vae_model_config: Configuration for VAE model
            dnn_model_config: Configuration for DNN model
            vae_training_config: Configuration for VAE training
            dnn_training_config: Configuration for DNN training
            delete_catalogs: Whether to delete catalogs after processing
            data_sizes: List of training data sizes for regression evaluation (optional)
            fixed_epochs: Number of epochs for regression evaluation (optional)

        Returns:
            bool: True if all processes completed successfully, False otherwise
        """
        self.logger.info("=" * 100)
        self.logger.info("RUNNING FULL FOUNDATION MODEL PIPELINE")
        self.logger.info(
            "Process: Train → Anomaly Detection → Regression → Signal Classification"
        )
        self.logger.info("=" * 100)
        self.logger.progress("Starting full foundation model pipeline")

        try:
            # Step 1: Train the foundation model
            self.logger.info("=" * 50)
            self.logger.progress("STEP 1/4: TRAINING FOUNDATION MODEL")
            self.logger.info("=" * 50)

            # Print system usage before training
            print_system_usage("Before Training - ")

            # Create foundation model trainer
            trainer = FoundationModelTrainer(
                experiments_output_dir=str(self.experiments_output_dir),
                processed_datasets_dir=str(self.processed_datasets_dir),
                logger=self.logger,
                source_config_file=self._source_config_file,
            )

            training_result = trainer.train_foundation_model(
                dataset_config=dataset_config,
                model_config=vae_model_config,
                training_config=vae_training_config,
                task_config=task_config,
                delete_catalogs=delete_catalogs,
            )

            # Print system usage after training
            print_system_usage("After Training - ")

            if (
                not training_result
                or not isinstance(training_result, tuple)
                or len(training_result) != 2
            ):
                self.logger.error(
                    "Training failed or did not return a valid (model_path, dataset_path) tuple"
                )
                return False

            foundation_model_path, dataset_path = training_result
            self.logger.info(
                f"Training completed successfully. Model saved at: {foundation_model_path}"
            )
            self.logger.info(f"Dataset saved at: {dataset_path}")

            # Step 2: Run anomaly detection evaluation
            self.logger.info("=" * 50)
            self.logger.progress("STEP 2/4: ANOMALY DETECTION EVALUATION")
            self.logger.info("=" * 50)

            # Print system usage before anomaly detection
            print_system_usage("Before Anomaly Detection - ")

            # Create anomaly detection evaluator
            foundation_experiment_id = Path(foundation_model_path).name
            anomaly_evaluator = AnomalyDetectionEvaluator(
                experiment_id=foundation_experiment_id,
                base_path=self.experiments_output_dir,
                processed_datasets_dir=str(self.processed_datasets_dir),
            )

            anomaly_success = anomaly_evaluator.evaluate_anomaly_detection(
                dataset_config=dataset_config,
                task_config=task_config,
                delete_catalogs=delete_catalogs,
                foundation_model_path=foundation_model_path,
                vae_training_config=vae_training_config,
            )

            # Print system usage after anomaly detection
            print_system_usage("After Anomaly Detection - ")

            if not anomaly_success:
                self.logger.error("Anomaly detection evaluation failed")
                return False

            # Step 3: Run regression evaluation
            self.logger.info("=" * 50)
            self.logger.progress("STEP 3/4: REGRESSION EVALUATION")
            self.logger.info("=" * 50)

            # Print system usage before regression evaluation
            print_system_usage("Before Regression Evaluation - ")

            # Create regression evaluator
            regression_evaluator = RegressionEvaluator(
                processed_datasets_dir=str(self.processed_datasets_dir),
                logger=self.logger,
                histogram_manager=self.histogram_manager,
                foundation_plot_manager=self.foundation_plot_manager,
            )

            regression_success = regression_evaluator.evaluate_regression(
                dataset_config=dataset_config,
                dnn_model_config=dnn_model_config,
                dnn_training_config=dnn_training_config,
                task_config=task_config,
                delete_catalogs=delete_catalogs,
                foundation_model_path=foundation_model_path,
                data_sizes=data_sizes,
                fixed_epochs=fixed_epochs,
            )

            # Print system usage after regression evaluation
            print_system_usage("After Regression Evaluation - ")

            if not regression_success:
                self.logger.error("Regression evaluation failed")
                return False

            # Step 4: Run signal classification evaluation
            self.logger.info("=" * 50)
            self.logger.progress("STEP 4/4: SIGNAL CLASSIFICATION EVALUATION")
            self.logger.info("=" * 50)

            # Print system usage before signal classification
            print_system_usage("Before Signal Classification - ")

            # Create signal classification evaluator
            signal_classification_evaluator = SignalClassificationEvaluator(
                processed_datasets_dir=str(self.processed_datasets_dir),
                logger=self.logger,
                foundation_plot_manager=self.foundation_plot_manager,
            )

            signal_classification_success = (
                signal_classification_evaluator.evaluate_signal_classification(
                    dataset_config=dataset_config,
                    dnn_model_config=dnn_model_config,
                    dnn_training_config=dnn_training_config,
                    task_config=task_config,
                    delete_catalogs=delete_catalogs,
                    foundation_model_path=foundation_model_path,
                    data_sizes=data_sizes,
                    fixed_epochs=fixed_epochs,
                )
            )

            # Print system usage after signal classification
            print_system_usage("After Signal Classification - ")

            if not signal_classification_success:
                self.logger.error("Signal classification evaluation failed")
                return False

            # Copy dataset plots to foundation model directory for easy reference
            self.logger.info("Copying dataset plots to foundation model directory...")
            try:
                # Extract dataset directory from dataset_path
                dataset_dir = Path(dataset_path).parent
                dataset_plots_dir = dataset_dir / "plots"

                if dataset_plots_dir.exists() and dataset_plots_dir.is_dir():
                    # Create destination directory in foundation model folder
                    foundation_dataset_plots_dir = (
                        Path(foundation_model_path) / "dataset_plots"
                    )
                    foundation_dataset_plots_dir.mkdir(parents=True, exist_ok=True)

                    # Copy all files from dataset plots directory
                    for plot_file in dataset_plots_dir.iterdir():
                        if plot_file.is_file():
                            destination = foundation_dataset_plots_dir / plot_file.name
                            shutil.copy2(plot_file, destination)
                            self.logger.info(f"Copied dataset plot: {plot_file.name}")

                    self.logger.info(
                        f"Dataset plots copied to: {foundation_dataset_plots_dir}"
                    )
                else:
                    self.logger.info("No dataset plots directory found to copy")

            except Exception as e:
                self.logger.warning(f"Failed to copy dataset plots: {str(e)}")
                # Don't fail the entire pipeline for plot copying issues

            # Final summary
            self.logger.info("=" * 100)
            self.logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 100)
            self.logger.info(f"Foundation model: {foundation_model_path}")
            self.logger.info(
                f"Dataset plots copied to: {Path(foundation_model_path) / 'dataset_plots'}"
            )
            self.logger.info(
                "All four processes (train → anomaly → regression → signal classification) completed successfully"
            )
            self.logger.info("=" * 100)
            self.logger.progress(
                "Full foundation model pipeline completed successfully!"
            )

            # Print final system usage
            print_system_usage("Full Pipeline Complete - ")

            return True

        except Exception as e:
            self.logger.error(f"Full pipeline failed: {type(e).__name__}: {str(e)}")
            self.logger.exception("Detailed traceback:")

            # Print system usage on failure
            print_system_usage("Full Pipeline Failed - ")
            return False
