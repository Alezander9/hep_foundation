# import tensorflow as tf
# from pathlib import Path
# import logging
# from typing import Dict

# from hep_foundation.utils import ATLAS_RUN_NUMBERS
# from hep_foundation.model_registry import ModelRegistry
# from hep_foundation.dataset_manager import DatasetManager
# from hep_foundation.variational_autoencoder import VariationalAutoEncoder, AnomalyDetectionEvaluator

# def load_model_from_experiment(experiment_id: str, registry: ModelRegistry) -> VariationalAutoEncoder:
#     """Load trained model from experiment directory"""
#     # Get experiment data
#     experiment_data = registry.get_experiment_data(experiment_id)
#     model_config = experiment_data['model_config']
    
#     # Create and build model with same configuration
#     model = VariationalAutoEncoder(
#         input_shape=(30, 6),  # From your dataset config
#         latent_dim=model_config['architecture']['latent_dim'],
#         encoder_layers=model_config['architecture']['encoder_layers'],
#         decoder_layers=model_config['architecture']['decoder_layers'],
#         quant_bits=model_config['hyperparameters'].get('quant_bits'),
#         activation=model_config['hyperparameters']['activation'],
#         beta_schedule=model_config['hyperparameters'].get('beta_schedule')
#     )
#     model.build()
    
#     # Load trained weights
#     model_paths = registry.load_model(experiment_id)
#     model.encoder = tf.keras.models.load_model(model_paths['encoder'])
#     model.decoder = tf.keras.models.load_model(model_paths['decoder'])
#     model.model = tf.keras.models.load_model(model_paths['full_model'])
    
#     return model

# def main():
#     # Setup logging
#     logging.basicConfig(
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         level=logging.INFO,
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
#     # Configuration
#     EXPERIMENT_ID = "002_vae_test"
#     BASE_PATH = Path("experiments")
    
#     # Dataset configuration (matching your original config)
#     dataset_config = {
#         'run_numbers': ATLAS_RUN_NUMBERS[-2:],
#         'track_selections': {
#             'eta': (-2.5, 2.5),
#             'chi2_per_ndof': (0.0, 10.0),
#         },
#         'event_selections': {},
#         'max_tracks_per_event': 30,
#         'min_tracks_per_event': 10,
#         'catalog_limit': 3
#     }
    
#     try:
#         # Initialize registry and data manager
#         registry = ModelRegistry(str(BASE_PATH))
#         data_manager = DatasetManager()
        
#         logging.info(f"\nLoading datasets...")
#         # Load datasets
#         _, _, test_dataset = data_manager.load_atlas_datasets(
#             config=dataset_config,
#             validation_fraction=0.15,
#             test_fraction=0.15,
#             batch_size=1024,
#             shuffle_buffer=50000,
#             plot_distributions=False,  # No need to plot again
#             delete_catalogs=False  # Keep existing catalogs
#         )
        
#         # Load signal datasets
#         logging.info(f"\nLoading signal datasets...")
#         signal_datasets = data_manager.load_signal_datasets(
#             config={
#                 'signal_types': ["zprime", "wprime_qq", "zprime_bb"],
#                 **dataset_config
#             },
#             batch_size=1024,
#             plot_distributions=False
#         )
        
#         # Load trained model
#         logging.info(f"\nLoading model from experiment {EXPERIMENT_ID}...")
#         model = load_model_from_experiment(EXPERIMENT_ID, registry)
        
#         # Initialize and run model tester
#         logging.info("\nInitializing model tester...")
#         tester = AnomalyDetectionEvaluator(
#             model=model,
#             test_dataset=test_dataset,
#             signal_datasets=signal_datasets,
#             experiment_id=EXPERIMENT_ID,
#             base_path=BASE_PATH
#         )
        
#         # Run anomaly detection test
#         logging.info("\nRunning anomaly detection test...")
#         test_results = tester.run_anomaly_detection_test()
        
#         logging.info("\nTest completed successfully!")
#         logging.info(f"Results saved to: {BASE_PATH/EXPERIMENT_ID}/testing")
        
#         return 0
        
#     except Exception as e:
#         logging.error(f"\nTest failed with error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return 1

# if __name__ == "__main__":
#     exit(main()) 