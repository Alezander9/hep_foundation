from pathlib import Path
import h5py
import json
import hashlib
from typing import Dict, Optional, Tuple, List
import tensorflow as tf
import numpy as np
from datetime import datetime
import yaml
import platform
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
import os
import logging
from hep_foundation.atlas_file_manager import ATLASFileManager
from hep_foundation.task_config import PhysliteFeatureFilter, PhysliteFeatureArrayFilter, PhysliteFeatureArrayAggregator, PhysliteSelectionConfig, PhysliteFeatureSelector, TaskConfig
from hep_foundation.utils import TypeConverter, ConfigSerializer

class DatasetManager:
    """Manages pre-processed ATLAS datasets with integrated processing capabilities"""
    
    def __init__(self, 
                 base_dir: str = "processed_datasets",
                 atlas_manager: Optional[ATLASFileManager] = None):
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.configs_dir = self.base_dir / "configs"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.atlas_manager = atlas_manager or ATLASFileManager()
        
        # Add state tracking
        self.current_dataset_id = None
        self.current_dataset_path = None
        self.current_dataset_info = None

    def get_current_dataset_id(self) -> str:
        """Get ID of currently loaded dataset"""
        if self.current_dataset_id is None:
            raise ValueError("No dataset currently loaded")
        return self.current_dataset_id

    def get_current_dataset_info(self) -> Dict:
        """Get information about the currently loaded dataset"""
        if self.current_dataset_info is None:
            raise ValueError("No dataset currently loaded")
        return self.current_dataset_info 
    
    def get_current_dataset_path(self) -> Path:
        """Get path of currently loaded dataset"""
        if self.current_dataset_path is None:
            raise ValueError("No dataset currently loaded")
        return self.current_dataset_path
        
    def generate_dataset_id(self, config: Dict) -> str:
        """Generate a human-readable dataset ID"""
        # Create a descriptive ID based on dataset type
        if 'signal_types' in config:
            # For signal datasets
            signal_str = '_'.join(sorted(config['signal_types']))
            id_components = [
                'signal',
                f'types{signal_str}',
                f'tracks{config["max_tracks_per_event"]}'
            ]
        else:
            # For regular datasets
            run_str = '_'.join(str(run) for run in sorted(config['run_numbers']))
            id_components = [
                'dataset',
                f'runs{run_str}',
                f'tracks{config["max_tracks_per_event"]}'
            ]
        
        # Add a short hash for uniqueness
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        id_components.append(config_hash)
        
        return "_".join(id_components)
    
    def save_dataset_config(self, dataset_id: str, config: Dict):
        """Save full dataset configuration"""
        config_path = self.configs_dir / f"{dataset_id}_config.yaml"
        
        # Prepare configuration
        full_config = {
            'dataset_id': dataset_id,
            'creation_date': str(datetime.now()),
            'config': config,
            'atlas_version': self.atlas_manager.get_version(),
            'software_versions': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'tensorflow': tf.__version__,
                'h5py': h5py.__version__
            }
        }
        
        # Save using our serializer
        ConfigSerializer.to_yaml(full_config, config_path)
        return config_path
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get full dataset information including recreation parameters"""
        config_path = self.configs_dir / f"{dataset_id}_config.yaml"
        logging.info(f"\nLooking for config at: {config_path}")  # Debug print
        if not config_path.exists():
            logging.info(f"Available configs: {list(self.configs_dir.glob('*.yaml'))}")  # Debug print
            raise ValueError(f"No configuration found for dataset {dataset_id}")
            
        return ConfigSerializer.from_yaml(config_path)
    
    def _create_dataset(self, task_config: TaskConfig, plot_distributions: bool = False, delete_catalogs: bool = True) -> Tuple[str, Path]:
        """
        Create new processed dataset from ATLAS data.
        
        Args:
            task_config: Configuration defining event filters, input features, and labels
            plot_distributions: Whether to generate distribution plots
            delete_catalogs: Whether to delete catalogs after processing
            
        Returns:
            Tuple containing:
            - Dataset ID
            - Path to created dataset
        """
        logging.info("Creating new dataset")
        
        # Generate dataset ID and paths
        dataset_id = self.generate_dataset_id(task_config)
        output_path = self.datasets_dir / f"{dataset_id}.h5"
        logging.info(f"\nGenerated dataset ID: {dataset_id}")
        
        try:
            # Save configuration first
            config_path = self.save_dataset_config(dataset_id, task_config)
            logging.info(f"Saved configuration to: {config_path}")
            
            # Process all runs
            all_inputs = []
            all_labels = []
            total_stats = {
                'total_events': 0,
                'processed_events': 0,
                'total_features': 0,
                'processing_time': 0
            }
            
            # Process data
            inputs, labels, stats = self._process_data(
                task_config=task_config,
                plot_distributions=plot_distributions,
                delete_catalogs=delete_catalogs
            )
            
            if not inputs:
                raise ValueError("No events passed selection criteria")
            
            # Create HDF5 dataset
            with h5py.File(output_path, 'w') as f:
                # Create input features group
                features_group = f.create_group('features')
                
                # Save scalar features if any exist
                scalar_features = {name: [] for name in inputs[0]['scalar_features'].keys()}
                for input_data in inputs:
                    for name, value in input_data['scalar_features'].items():
                        scalar_features[name].append(value)
                
                for name, values in scalar_features.items():
                    features_group.create_dataset(
                        f'scalar/{name}',
                        data=np.array(values),
                        compression='gzip'
                    )
                
                # Save aggregated features if any exist
                for agg_name, agg_data in inputs[0]['aggregated_features'].items():
                    # Stack all events for this aggregator
                    stacked_data = np.stack([
                        input_data['aggregated_features'][agg_name]
                        for input_data in inputs
                    ])
                    
                    features_group.create_dataset(
                        f'aggregated/{agg_name}',
                        data=stacked_data,
                        compression='gzip'
                    )
                
                # Create labels group if we have labels
                if labels and task_config.labels:
                    labels_group = f.create_group('labels')
                    
                    # Process each label configuration
                    for label_idx, label_config in enumerate(task_config.labels):
                        label_subgroup = labels_group.create_group(f'config_{label_idx}')
                        
                        # Get all label data for this configuration
                        label_data = [label_set[label_idx] for label_set in labels]
                        
                        # Save scalar features
                        scalar_features = {name: [] for name in label_data[0]['scalar_features'].keys()}
                        for event_labels in label_data:
                            for name, value in event_labels['scalar_features'].items():
                                scalar_features[name].append(value)
                        
                        for name, values in scalar_features.items():
                            label_subgroup.create_dataset(
                                f'scalar/{name}',
                                data=np.array(values),
                                compression='gzip'
                            )
                        
                        # Save aggregated features if any exist
                        for agg_name, agg_data in label_data[0]['aggregated_features'].items():
                            stacked_data = np.stack([
                                event_labels['aggregated_features'][agg_name]
                                for event_labels in label_data
                            ])
                            
                            label_subgroup.create_dataset(
                                f'aggregated/{agg_name}',
                                data=stacked_data,
                                compression='gzip'
                            )
                
                # Compute and store normalization parameters
                norm_params = self._compute_dataset_normalization(inputs, labels if labels else None)
                
                # Store attributes
                f.attrs.update({
                    'dataset_id': dataset_id,
                    'creation_date': str(datetime.now()),
                    'has_labels': bool(labels and task_config.labels),
                    'normalization_params': json.dumps(norm_params),
                    'processing_stats': json.dumps(stats)
                })
            
            return dataset_id, output_path
            
        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                output_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise Exception(f"Dataset creation failed: {str(e)}")

    def _compute_dataset_normalization(self, 
                                     inputs: List[Dict[str, Dict[str, np.ndarray]]], 
                                     labels: Optional[List[List[Dict[str, Dict[str, np.ndarray]]]]] = None) -> Dict:
        """
        Compute normalization parameters for all features and labels.
        
        Args:
            inputs: List of input feature dictionaries
            labels: Optional list of label feature dictionaries
            
        Returns:
            Dictionary containing normalization parameters for all features and labels
        """
        norm_params = {'features': {}}
        
        # Compute for scalar features
        if inputs[0]['scalar_features']:
            scalar_features = {name: [] for name in inputs[0]['scalar_features'].keys()}
            for input_data in inputs:
                for name, value in input_data['scalar_features'].items():
                    scalar_features[name].append(value)
            
            norm_params['features']['scalar'] = {
                name: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values) or 1.0)
                }
                for name, values in scalar_features.items()
            }
        
        # Compute for aggregated features
        if inputs[0]['aggregated_features']:
            norm_params['features']['aggregated'] = {}
            for agg_name, agg_data in inputs[0]['aggregated_features'].items():
                # Stack all events for this aggregator
                stacked_data = np.stack([
                    input_data['aggregated_features'][agg_name]
                    for input_data in inputs
                ])
                
                # Create mask for zero-padded values
                mask = np.any(stacked_data != 0, axis=-1, keepdims=True)
                mask = np.broadcast_to(mask, stacked_data.shape)
                
                # Create masked array
                masked_data = np.ma.array(stacked_data, mask=~mask)
                
                # Compute stats along event and element dimensions
                means = np.ma.mean(masked_data, axis=(0,1)).data
                stds = np.ma.std(masked_data, axis=(0,1)).data
                
                # Ensure no zero standard deviations
                stds = np.maximum(stds, 1e-6)
                
                norm_params['features']['aggregated'][agg_name] = {
                    'means': means.tolist(),
                    'stds': stds.tolist()
                }
        
        # Compute for labels if present
        if labels and labels[0]:
            norm_params['labels'] = []
            for label_idx in range(len(labels[0])):
                label_norm = {}
                
                # Get all label data for this configuration
                label_data = [label_set[label_idx] for label_set in labels]
                
                # Compute for scalar features
                if label_data[0]['scalar_features']:
                    scalar_features = {name: [] for name in label_data[0]['scalar_features'].keys()}
                    for event_labels in label_data:
                        for name, value in event_labels['scalar_features'].items():
                            scalar_features[name].append(value)
                    
                    label_norm['scalar'] = {
                        name: {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values) or 1.0)
                        }
                        for name, values in scalar_features.items()
                    }
                
                # Compute for aggregated features
                if label_data[0]['aggregated_features']:
                    label_norm['aggregated'] = {}
                    for agg_name, agg_data in label_data[0]['aggregated_features'].items():
                        stacked_data = np.stack([
                            event_labels['aggregated_features'][agg_name]
                            for event_labels in label_data
                        ])
                        
                        mask = np.any(stacked_data != 0, axis=-1, keepdims=True)
                        mask = np.broadcast_to(mask, stacked_data.shape)
                        masked_data = np.ma.array(stacked_data, mask=~mask)
                        
                        means = np.ma.mean(masked_data, axis=(0,1)).data
                        stds = np.ma.std(masked_data, axis=(0,1)).data
                        stds = np.maximum(stds, 1e-6)
                        
                        label_norm['aggregated'][agg_name] = {
                            'means': means.tolist(),
                            'stds': stds.tolist()
                        }
                
                norm_params['labels'].append(label_norm)
        
        return norm_params

    def load_datasets(self, 
                     task_config: TaskConfig,
                     validation_fraction: float = 0.15,
                     test_fraction: float = 0.15,
                     batch_size: int = 1000,
                     shuffle_buffer: int = 10000,
                     include_labels: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and split dataset into train/val/test.
        
        Args:
            task_config: Task configuration defining data selection and processing
            validation_fraction: Fraction of data to use for validation
            test_fraction: Fraction of data to use for testing
            batch_size: Size of batches in returned dataset
            shuffle_buffer: Size of shuffle buffer for training data
            include_labels: Whether to include labels in the dataset
            
        Returns:
            Tuple of (train, validation, test) tf.data.Datasets
        """
        logging.info("Attempting to load datasets")
        try:
            # Generate dataset ID and paths
            dataset_id = self.generate_dataset_id(task_config)
            dataset_path = self.datasets_dir / f"{dataset_id}.h5"
            logging.info(f"\nLooking for dataset: {dataset_id}")
            
            # Create dataset if it doesn't exist
            if not dataset_path.exists():
                logging.info(f"\nDataset not found, creating new dataset: {dataset_id}")
                dataset_id, dataset_path = self._create_dataset(task_config=task_config)
            
            # Set current dataset tracking
            self.current_dataset_id = dataset_id
            self.current_dataset_path = dataset_path
            self.current_dataset_info = self.get_dataset_info(dataset_id)
            
            # Load and process dataset
            with h5py.File(dataset_path, 'r') as f:
                # Verify file integrity
                if 'features' not in f:
                    raise ValueError("Dataset missing features group")
                
                # Load input features
                features_dict = {}
                
                # Load scalar features
                if 'scalar' in f['features']:
                    for name, dataset in f['features']['scalar'].items():
                        features_dict[f'scalar/{name}'] = dataset[:]
                
                # Load aggregated features
                if 'aggregated' in f['features']:
                    for name, dataset in f['features']['aggregated'].items():
                        features_dict[f'aggregated/{name}'] = dataset[:]
                
                # Load labels if requested
                labels_dict = {}
                if include_labels and 'labels' in f and f.attrs.get('has_labels', False):
                    for config_name, label_group in f['labels'].items():
                        config_dict = {}
                        
                        # Load scalar features
                        if 'scalar' in label_group:
                            for name, dataset in label_group['scalar'].items():
                                config_dict[f'scalar/{name}'] = dataset[:]
                        
                        # Load aggregated features
                        if 'aggregated' in label_group:
                            for name, dataset in label_group['aggregated'].items():
                                config_dict[f'aggregated/{name}'] = dataset[:]
                        
                        labels_dict[config_name] = config_dict
                
                # Get normalization parameters
                norm_params = json.loads(f.attrs['normalization_params'])
                
                # Verify we have data
                if not features_dict:
                    raise ValueError("Dataset contains no features")
                
                # Get the number of events (from first feature)
                n_events = next(iter(features_dict.values())).shape[0]
                
                # Create TensorFlow datasets
                feature_arrays = []
                for name in sorted(features_dict.keys()):  # Sort for consistent order
                    feature_arrays.append(features_dict[name])
                
                if labels_dict and include_labels:
                    label_arrays = []
                    for config_name in sorted(labels_dict.keys()):  # Sort for consistent order
                        config_arrays = []
                        for name in sorted(labels_dict[config_name].keys()):
                            config_arrays.append(labels_dict[config_name][name])
                        label_arrays.extend(config_arrays)
                    
                    # Create dataset with features and labels
                    dataset = tf.data.Dataset.from_tensor_slices((
                        tuple(feature_arrays),
                        tuple(label_arrays)
                    ))
                    
                    # Apply normalization
                    def normalize_data(features, labels):
                        normalized_features = []
                        for i, feature in enumerate(features):
                            if f'scalar' in norm_params['features']:
                                name = sorted(features_dict.keys())[i]
                                if name in norm_params['features']['scalar']:
                                    params = norm_params['features']['scalar'][name]
                                    feature = (feature - params['mean']) / params['std']
                            elif f'aggregated' in norm_params['features']:
                                name = sorted(features_dict.keys())[i]
                                if name in norm_params['features']['aggregated']:
                                    params = norm_params['features']['aggregated'][name]
                                    feature = (feature - params['means']) / params['stds']
                            normalized_features.append(feature)
                        return tuple(normalized_features), labels
                    
                    dataset = dataset.map(normalize_data)
                
                else:
                    # Create dataset with features only
                    dataset = tf.data.Dataset.from_tensor_slices(tuple(feature_arrays))
                    
                    # Apply normalization
                    def normalize_features(*features):
                        normalized_features = []
                        for i, feature in enumerate(features):
                            if f'scalar' in norm_params['features']:
                                name = sorted(features_dict.keys())[i]
                                if name in norm_params['features']['scalar']:
                                    params = norm_params['features']['scalar'][name]
                                    feature = (feature - params['mean']) / params['std']
                            elif f'aggregated' in norm_params['features']:
                                name = sorted(features_dict.keys())[i]
                                if name in norm_params['features']['aggregated']:
                                    params = norm_params['features']['aggregated'][name]
                                    feature = (feature - params['means']) / params['stds']
                            normalized_features.append(feature)
                        return tuple(normalized_features)
                    
                    dataset = dataset.map(normalize_features)
                
                # Calculate split sizes
                val_size = int(validation_fraction * n_events)
                test_size = int(test_fraction * n_events)
                train_size = n_events - val_size - test_size
                
                # Create splits
                train_dataset = dataset.take(train_size).shuffle(
                    buffer_size=shuffle_buffer,
                    reshuffle_each_iteration=True
                ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                remaining = dataset.skip(train_size)
                val_dataset = remaining.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                test_dataset = remaining.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                # Log dataset sizes
                logging.info("\nDataset sizes:")
                for name, ds in [("Training", train_dataset), 
                               ("Validation", val_dataset), 
                               ("Test", test_dataset)]:
                    n_batches = sum(1 for _ in ds)
                    n_events = n_batches * batch_size
                    logging.info(f"{name}: {n_events} events ({n_batches} batches)")
                
                return train_dataset, val_dataset, test_dataset
                
        except Exception as e:
            raise Exception(f"Failed to load dataset: {str(e)}")

    def _apply_event_filters(self, 
                            event_data: Dict[str, np.ndarray],
                            event_filters: List[PhysliteFeatureFilter]) -> bool:
        """
        Apply event-level filters to determine if an event should be processed.
        
        Args:
            event_data: Dictionary mapping branch names to their values
            event_filters: List of event-level feature filters to apply
            
        Returns:
            bool: True if event passes all filters, False otherwise
        """
        if not event_filters:
            return True
        
        for filter in event_filters:
            value = event_data.get(filter.branch.name)
            if value is None:
                return False
            
            # For event filters, we expect scalar values
            if isinstance(value, np.ndarray) and value.size > 1:
                return False
            
            # Convert to scalar if needed
            value = value.item() if isinstance(value, np.ndarray) else value
            
            # Apply min/max filters
            if filter.min_value is not None and value < filter.min_value:
                return False
            if filter.max_value is not None and value > filter.max_value:
                return False
        
        return True

    def _extract_selected_features(self,
                                 event_data: Dict[str, np.ndarray],
                                 feature_selectors: List[PhysliteFeatureSelector]) -> Dict[str, np.ndarray]:
        """
        Extract selected features from event data.
        
        Args:
            event_data: Dictionary mapping branch names to their values
            feature_selectors: List of feature selectors defining what to extract
            
        Returns:
            Dictionary mapping branch names to their selected values
        """
        selected_features = {}
        
        for selector in feature_selectors:
            value = event_data.get(selector.branch.name)
            if value is None:
                continue
            
            # For scalar features, ensure we get a single value
            if selector.branch.is_feature:
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        continue
                    value = value.item()
            
            selected_features[selector.branch.name] = value
        
        return selected_features

    def _process_selection_config(self,
                                event_data: Dict[str, np.ndarray],
                                selection_config: PhysliteSelectionConfig) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single selection configuration to extract and aggregate features.
        
        Args:
            event_data: Dictionary mapping branch names to their values
            selection_config: Configuration defining what features to select and how to aggregate arrays
            
        Returns:
            Dictionary containing:
                - 'scalar_features': Dict of selected scalar features
                - 'aggregated_features': Dict of aggregated array features
            Returns None if any required features are missing or aggregation fails
        """
        # Extract scalar features
        scalar_features = self._extract_selected_features(
            event_data,
            selection_config.feature_selectors
        )
        
        # Process aggregators
        aggregated_features = {}
        for idx, aggregator in enumerate(selection_config.feature_array_aggregators):
            # Get all needed arrays first
            array_features = {}
            
            # Get input branch arrays
            for selector in aggregator.input_branches:
                value = event_data.get(selector.branch.name)
                if value is None or len(value) == 0:
                    return None
                array_features[selector.branch.name] = value
            
            # Get sort branch array
            sort_value = event_data.get(aggregator.sort_by_branch.branch.name)
            if sort_value is None or len(sort_value) == 0:
                return None
            array_features[aggregator.sort_by_branch.branch.name] = sort_value
            
            # Apply filters to get valid mask
            valid_mask = self._apply_feature_array_filters(
                array_features,
                aggregator.filter_branches
            )
            
            # Apply aggregator
            result = self._apply_feature_array_aggregator(
                array_features,
                aggregator,
                valid_mask
            )
            
            if result is None:
                return None
            
            aggregated_features[f'aggregator_{idx}'] = result
        
        # Return None if we have no features at all
        if not scalar_features and not aggregated_features:
            return None
        
        return {
            'scalar_features': scalar_features,
            'aggregated_features': aggregated_features
        }

    def _process_event(self,
                      event_data: Dict[str, np.ndarray],
                      task_config: PhysliteSelectionConfig) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single event using the task configuration.
        
        Args:
            event_data: Dictionary mapping branch names to their raw values
            task_config: PhysliteSelectionConfig defining features and filters
            
        Returns:
            Dictionary of processed features (scalar, array, and labels) or None if event rejected
        """
        # Apply event filters first
        if not self._apply_event_filters(event_data, task_config.event_filters):
            return None
        
        # Process input selection config
        input_features = self._process_selection_config(event_data, task_config)
        if input_features is None:
            return None
        
        # Process each label config if present
        label_features = {}
        for label_config in task_config.labels:
            label_result = self._process_selection_config(event_data, label_config)
            if label_result is None:
                return None
            label_features.update(label_result)
        
        return {
            'scalar_features': input_features['scalar_features'],
            'array_features': input_features['array_features'],
            'aggregated_features': input_features['aggregated_features'],
            'label_features': label_features
        }

    def _get_catalog_paths(self, 
                          run_number: Optional[str] = None,
                          signal_key: Optional[str] = None,
                          catalog_limit: Optional[int] = None) -> List[Path]:
        """Get list of catalog paths for either ATLAS data or signal data"""
        if run_number is not None:
            # Get ATLAS data catalogs
            paths = []
            for catalog_idx in range(self.atlas_manager.get_catalog_count(run_number)):
                if catalog_limit and catalog_idx >= catalog_limit:
                    break
                catalog_path = self.atlas_manager.get_run_catalog_path(run_number, catalog_idx)
                if not catalog_path.exists():
                    catalog_path = self.atlas_manager.download_run_catalog(run_number, catalog_idx)
                if catalog_path:
                    paths.append(catalog_path)
            logging.info(f"Found {len(paths)} catalogs for run {run_number}")
            return paths
        elif signal_key is not None:
            # Get signal data catalog
            catalog_path = self.atlas_manager.get_signal_catalog_path(signal_key, 0)
            if not catalog_path.exists():
                catalog_path = self.atlas_manager.download_signal_catalog(signal_key, 0)
            logging.info(f"Found signal catalog for {signal_key}")
            return [catalog_path] if catalog_path else []
        else:
            raise ValueError("Must provide either run_number or signal_key")

    def _process_data(self,
                     task_config: TaskConfig,
                     run_number: Optional[str] = None,
                     signal_key: Optional[str] = None,
                     catalog_limit: Optional[int] = None,
                     plot_distributions: bool = False,
                     delete_catalogs: bool = True) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], Dict]:
        """
        Process either ATLAS or signal data using task configuration.
        
        Args:
            task_config: Configuration defining event filters, input features, and labels
            run_number: Optional run number for ATLAS data
            signal_key: Optional signal type for signal data
            catalog_limit: Optional limit on number of catalogs to process
            plot_distributions: Whether to generate distribution plots
            delete_catalogs: Whether to delete catalogs after processing
            
        Returns:
            Tuple containing:
            - List of processed input features (each a dict with scalar and aggregated features)
            - List of processed label features (each a list of dicts for multiple label configs)
            - Processing statistics
        """
        logging.info(f"\nProcessing {'signal' if signal_key else 'ATLAS'} data")
        
        # Initialize statistics
        stats = {
            'total_events': 0,
            'processed_events': 0,
            'total_features': 0,
            'processing_time': 0.0
        }
        
        processed_inputs = []
        processed_labels = []
        
        # Collect all required branch names
        required_branches = set()
        
        # Add event filter branches
        for filter in task_config.event_filters:
            required_branches.add(filter.branch.name)
        
        # Add input selection branches
        for selector in task_config.input.feature_selectors:
            required_branches.add(selector.branch.name)
        
        # Add input aggregator branches
        for aggregator in task_config.input.feature_array_aggregators:
            # Add input branches
            for selector in aggregator.input_branches:
                required_branches.add(selector.branch.name)
            # Add filter branches
            for filter in aggregator.filter_branches:
                required_branches.add(filter.branch.name)
            # Add sort branch
            required_branches.add(aggregator.sort_by_branch.branch.name)
        
        # Add label selection branches
        for label_config in task_config.labels:
            for selector in label_config.feature_selectors:
                required_branches.add(selector.branch.name)
            for aggregator in label_config.feature_array_aggregators:
                for selector in aggregator.input_branches:
                    required_branches.add(selector.branch.name)
                for filter in aggregator.filter_branches:
                    required_branches.add(filter.branch.name)
                required_branches.add(aggregator.sort_by_branch.branch.name)
        
        # Get catalog paths
        logging.info(f"Getting catalog paths for run {run_number} and signal {signal_key}")
        catalog_paths = self._get_catalog_paths(run_number, signal_key, catalog_limit)
        logging.info(f"Found {len(catalog_paths)} catalog paths")
        
        for catalog_idx, catalog_path in enumerate(catalog_paths):
            logging.info(f"\nProcessing catalog {catalog_idx} with path: {catalog_path}")
            
            try:
                catalog_start_time = datetime.now()
                catalog_stats = {'events': 0, 'processed': 0}
                
                with uproot.open(catalog_path) as file:
                    tree = file["CollectionTree;1"]
                    
                    # Read all required branches
                    for arrays in tree.iterate(required_branches, library="np", step_size=1000):
                        catalog_stats['events'] += len(next(iter(arrays.values())))
                        
                        for evt_idx in range(len(next(iter(arrays.values())))):
                            # Extract all branches for this event
                            event_data = {
                                branch_name: arrays[branch_name][evt_idx]
                                for branch_name in required_branches
                                if branch_name in arrays
                            }
                            
                            # Skip empty events
                            if not event_data or not any(len(v) > 0 if isinstance(v, np.ndarray) else True 
                                                       for v in event_data.values()):
                                continue
                            
                            # Process event
                            result = self._process_event(event_data, task_config)
                            if result is not None:
                                input_features, label_features = result
                                processed_inputs.append(input_features)
                                processed_labels.append(label_features)
                                catalog_stats['processed'] += 1
                                
                                # Update feature statistics using the first aggregator's result if available
                                if input_features['aggregated_features']:
                                    first_aggregator = next(iter(input_features['aggregated_features'].values()))
                                    stats['total_features'] += np.sum(np.any(first_aggregator != 0, axis=1))
            
                # Update statistics
                catalog_duration = (datetime.now() - catalog_start_time).total_seconds()
                stats['processing_time'] += catalog_duration
                stats['total_events'] += catalog_stats['events']
                stats['processed_events'] += catalog_stats['processed']
                
                # Print catalog summary
                logging.info(f"Catalog {catalog_idx} summary:")
                logging.info(f"  Events processed: {catalog_stats['events']}")
                logging.info(f"  Events passing selection: {catalog_stats['processed']}")
                logging.info(f"  Processing time: {catalog_duration:.1f}s")
                logging.info(f"  Rate: {catalog_stats['events']/catalog_duration:.1f} events/s")
                
                if catalog_limit and catalog_idx >= catalog_limit - 1:
                    break
                    
            except Exception as e:
                logging.error(f"Error processing catalog {catalog_path}: {str(e)}")
                continue
            
            finally:
                if delete_catalogs and catalog_path.exists():
                    os.remove(catalog_path)
        
        # Convert stats to native Python types
        stats = {
            'total_events': int(stats['total_events']),
            'processed_events': int(stats['processed_events']),
            'total_features': int(stats['total_features']),
            'processing_time': float(stats['processing_time'])
        }
        
        return processed_inputs, processed_labels, stats

    def _plot_distributions(self, 
                           pre_selection_stats: Dict[str, List],
                           post_selection_stats: Dict[str, List],
                           output_dir: Path):
        """Create distribution plots and print statistical summaries for track and event features"""
        logging.info(f"\nGenerating plots in: {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print track multiplicity statistics
        logging.info("\n=== Track Multiplicity Statistics ===")
        logging.info("Before Selection:")
        logging.info(f"  Total events: {len(pre_selection_stats['tracks_per_event']):,}")
        logging.info(f"  Average tracks/event: {np.mean(pre_selection_stats['tracks_per_event']):.2f}")
        logging.info(f"  Median tracks/event: {np.median(pre_selection_stats['tracks_per_event']):.2f}")
        logging.info(f"  Min tracks: {min(pre_selection_stats['tracks_per_event'])}")
        logging.info(f"  Max tracks: {max(pre_selection_stats['tracks_per_event'])}")
        
        logging.info("\nAfter Selection:")
        logging.info(f"  Total events: {len(post_selection_stats['tracks_per_event']):,}")
        logging.info(f"  Average tracks/event: {np.mean(post_selection_stats['tracks_per_event']):.2f}")
        logging.info(f"  Median tracks/event: {np.median(post_selection_stats['tracks_per_event']):.2f}")
        logging.info(f"  Min tracks: {min(post_selection_stats['tracks_per_event'])}")
        logging.info(f"  Max tracks: {max(post_selection_stats['tracks_per_event'])}")
        logging.info(f"  Selection efficiency: {100 * len(post_selection_stats['tracks_per_event']) / len(pre_selection_stats['tracks_per_event']):.1f}%")
        
        # Use matplotlib style
        plt.style.use('seaborn-v0_8')
        
        logging.info("\nCreating track multiplicity plot...")
        plt.figure(figsize=(12, 6))
        
        # Calculate integer bin edges with percentile limits
        min_tracks = max(1, int(np.percentile(pre_selection_stats['tracks_per_event'], 1)))
        max_tracks = int(np.percentile(pre_selection_stats['tracks_per_event'], 99))
        
        # Create integer bins between these limits
        bins = np.arange(min_tracks - 0.5, max_tracks + 1.5, 1)  # +/- 0.5 centers bins on integers
        
        plt.hist(pre_selection_stats['tracks_per_event'], bins=bins, alpha=0.5, 
                 label='Before Selection', density=True)
        plt.hist(post_selection_stats['tracks_per_event'], bins=bins, alpha=0.5,
                 label='After Selection', density=True)
        
        plt.xlabel('Number of Tracks per Event')
        plt.ylabel('Density')
        plt.title('Track Multiplicity Distribution')
        plt.legend()
        plt.grid(True)
        
        # Set x-axis limits to show the main distribution
        plt.xlim(min_tracks - 1, max_tracks + 1)
        
        plt.savefig(output_dir / 'track_multiplicity.pdf')
        plt.close()
        
        logging.info("\nCreating track features plot...")
        # 2. Track features distributions (6x2 subplot grid)
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Track Feature Distributions (Before vs After Selection)')
        
        features = ['pt', 'eta', 'phi', 'd0', 'z0', 'chi2_per_ndof']
        for (feature, ax) in zip(features, axes.flat):
            if feature == 'pt':
                ax.set_xlabel('pT [GeV]')
                ax.set_xscale('log')
                # Use log-spaced bins for pT
                log_bins = np.logspace(
                    np.log10(max(0.1, np.percentile(pre_selection_stats[feature], 0.1))),  # min
                    np.log10(np.percentile(pre_selection_stats[feature], 99.9)),  # max
                    50  # number of bins
                )
                ax.hist(pre_selection_stats[feature], bins=log_bins, alpha=0.5,
                        label='Before Selection', density=True)
                ax.hist(post_selection_stats[feature], bins=log_bins, alpha=0.5,
                        label='After Selection', density=True)
            else:
                # For other features, use percentile-based limits
                x_min = np.percentile(pre_selection_stats[feature], 0.1)
                x_max = np.percentile(pre_selection_stats[feature], 99.9)
                ax.set_xlim(x_min, x_max)
                ax.hist(pre_selection_stats[feature], bins=50, alpha=0.5,
                        label='Before Selection', density=True, range=(x_min, x_max))
                ax.hist(post_selection_stats[feature], bins=50, alpha=0.5,
                        label='After Selection', density=True, range=(x_min, x_max))
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True)
            
            # Add specific axis labels and ranges
            if feature == 'pt':
                ax.set_xlabel('pT [GeV]')
                ax.set_xscale('log')
            elif feature == 'eta':
                ax.set_xlabel('η')
            elif feature == 'phi':
                ax.set_xlabel('φ')
                ax.set_xlim(-3.5, 3.5)
            elif feature == 'd0':
                ax.set_xlabel('d0 [mm]')
            elif feature == 'z0':
                ax.set_xlabel('z0 [mm]')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'track_features.pdf')
        plt.close()
        
        logging.info("\n=== Track Feature Statistics ===")
        features = ['pt', 'eta', 'phi', 'd0', 'z0', 'chi2_per_ndof']
        labels = {
            'pt': 'pT [GeV]',
            'eta': 'η',
            'phi': 'φ',
            'd0': 'd0 [mm]',
            'z0': 'z0 [mm]',
            'chi2_per_ndof': 'χ²/ndof'
        }
        
        for feature in features:
            logging.info(f"\n{labels[feature]}:")
            logging.info("  Before Selection:")
            logging.info(f"    Mean: {np.mean(pre_selection_stats[feature]):.3f}")
            logging.info(f"    Std:  {np.std(pre_selection_stats[feature]):.3f}")
            logging.info(f"    Min:  {np.min(pre_selection_stats[feature]):.3f}")
            logging.info(f"    Max:  {np.max(pre_selection_stats[feature]):.3f}")
            logging.info(f"    Tracks: {len(pre_selection_stats[feature]):,}")
            
            logging.info("  After Selection:")
            logging.info(f"    Mean: {np.mean(post_selection_stats[feature]):.3f}")
            logging.info(f"    Std:  {np.std(post_selection_stats[feature]):.3f}")
            logging.info(f"    Min:  {np.min(post_selection_stats[feature]):.3f}")
            logging.info(f"    Max:  {np.max(post_selection_stats[feature]):.3f}")
            logging.info(f"    Tracks: {len(post_selection_stats[feature]):,}")
        
        # Print correlation information
        logging.info("\n=== Feature Correlations ===")
        df = pd.DataFrame({
            feature: post_selection_stats[feature] 
            for feature in features
        })
        corr_matrix = df.corr()
        
        logging.info("\nCorrelation Matrix (after selection):")
        pd.set_option('display.float_format', '{:.3f}'.format)
        logging.info(corr_matrix)
        
        logging.info("\nCreating correlation plot...")
        # 3. 2D correlation plots
        plt.figure(figsize=(12, 10))
        feature_data = {
            feature: post_selection_stats[feature] 
            for feature in features
        }
        df = pd.DataFrame(feature_data)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Track Feature Correlations (After Selection)')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_correlations.pdf')
        plt.close()
        
        logging.info("\nPlotting complete!") 

    def _create_signal_dataset(self, task_config: TaskConfig, plot_distributions: bool = False) -> Tuple[str, Path]:
        """
        Create new processed dataset from signal data.
        
        Args:
            task_config: Configuration defining event filters, input features, and labels
            plot_distributions: Whether to generate distribution plots
            
        Returns:
            Tuple containing:
            - Dataset ID
            - Path to created dataset
        """
        logging.info("Creating new signal dataset")
        
        # Generate dataset ID and paths
        dataset_id = self.generate_dataset_id(task_config)
        output_path = self.datasets_dir / "signals" / f"{dataset_id}.h5"
        
        try:
            # Create signals directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = self.save_dataset_config(dataset_id, task_config)
            
            # Process each signal type separately
            with h5py.File(output_path, 'w') as f:
                # Create group for each signal type
                for signal_key in task_config.signal_types:
                    logging.info(f"\nProcessing signal type: {signal_key}")
                    
                    # Process data for this signal type
                    inputs, labels, stats = self._process_data(
                        task_config=task_config,
                        signal_key=signal_key,
                        plot_distributions=plot_distributions,
                        delete_catalogs=False
                    )
                    
                    if not inputs:
                        logging.warning(f"No events passed selection for {signal_key}")
                        continue
                    
                    # Create signal-specific group
                    signal_group = f.create_group(signal_key)
                    
                    # Create features group
                    features_group = signal_group.create_group('features')
                    
                    # Save scalar features
                    scalar_features = {name: [] for name in inputs[0]['scalar_features'].keys()}
                    for input_data in inputs:
                        for name, value in input_data['scalar_features'].items():
                            scalar_features[name].append(value)
                    
                    for name, values in scalar_features.items():
                        features_group.create_dataset(
                            f'scalar/{name}',
                            data=np.array(values),
                            compression='gzip'
                        )
                    
                    # Save aggregated features
                    for agg_name, agg_data in inputs[0]['aggregated_features'].items():
                        stacked_data = np.stack([
                            input_data['aggregated_features'][agg_name]
                            for input_data in inputs
                        ])
                        
                        features_group.create_dataset(
                            f'aggregated/{agg_name}',
                            data=stacked_data,
                            compression='gzip'
                        )
                    
                    # Create labels group if we have labels
                    if labels and task_config.labels:
                        labels_group = signal_group.create_group('labels')
                        
                        # Process each label configuration
                        for label_idx, label_config in enumerate(task_config.labels):
                            label_subgroup = labels_group.create_group(f'config_{label_idx}')
                            
                            # Get all label data for this configuration
                            label_data = [label_set[label_idx] for label_set in labels]
                            
                            # Save scalar features
                            scalar_features = {name: [] for name in label_data[0]['scalar_features'].keys()}
                            for event_labels in label_data:
                                for name, value in event_labels['scalar_features'].items():
                                    scalar_features[name].append(value)
                            
                            for name, values in scalar_features.items():
                                label_subgroup.create_dataset(
                                    f'scalar/{name}',
                                    data=np.array(values),
                                    compression='gzip'
                                )
                            
                            # Save aggregated features if any exist
                            for agg_name, agg_data in label_data[0]['aggregated_features'].items():
                                stacked_data = np.stack([
                                    event_labels['aggregated_features'][agg_name]
                                    for event_labels in label_data
                                ])
                                
                                label_subgroup.create_dataset(
                                    f'aggregated/{agg_name}',
                                    data=stacked_data,
                                    compression='gzip'
                                )
                    
                    # Compute and store normalization parameters
                    norm_params = self._compute_dataset_normalization(inputs, labels if labels else None)
                    
                    # Store signal-specific attributes
                    signal_group.attrs.update({
                        'has_labels': bool(labels and task_config.labels),
                        'normalization_params': json.dumps(norm_params),
                        'processing_stats': json.dumps(stats)
                    })
                
                # Store global attributes
                f.attrs.update({
                    'dataset_id': dataset_id,
                    'creation_date': str(datetime.now())
                })
            
            return dataset_id, output_path
            
        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                output_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise Exception(f"Signal dataset creation failed: {str(e)}")

    def load_signal_datasets(self,
                            task_config: TaskConfig,
                            batch_size: int = 1000,
                            include_labels: bool = False) -> Dict[str, tf.data.Dataset]:
        """
        Load signal datasets for evaluation.
        
        Args:
            task_config: Task configuration defining data selection and processing
            batch_size: Size of batches in returned dataset
            include_labels: Whether to include labels in the dataset
            
        Returns:
            Dictionary mapping signal_type to its corresponding TensorFlow dataset
        """
        logging.info("Loading signal datasets")
        try:
            # Generate dataset ID and paths
            dataset_id = self.generate_dataset_id(task_config)
            dataset_path = self.datasets_dir / "signals" / f"{dataset_id}.h5"
            
            # Create if doesn't exist
            if not dataset_path.exists():
                dataset_id, dataset_path = self._create_signal_dataset(task_config=task_config)
            
            # Load datasets
            signal_datasets = {}
            with h5py.File(dataset_path, 'r') as f:
                for signal_key in f.keys():
                    signal_group = f[signal_key]
                    
                    # Load features
                    features_dict = {}
                    
                    # Load scalar features
                    if 'scalar' in signal_group['features']:
                        for name, dataset in signal_group['features']['scalar'].items():
                            features_dict[f'scalar/{name}'] = dataset[:]
                    
                    # Load aggregated features
                    if 'aggregated' in signal_group['features']:
                        for name, dataset in signal_group['features']['aggregated'].items():
                            features_dict[f'aggregated/{name}'] = dataset[:]
                    
                    # Load labels if requested
                    labels_dict = {}
                    if include_labels and 'labels' in signal_group and signal_group.attrs.get('has_labels', False):
                        for config_name, label_group in signal_group['labels'].items():
                            config_dict = {}
                            
                            # Load scalar features
                            if 'scalar' in label_group:
                                for name, dataset in label_group['scalar'].items():
                                    config_dict[f'scalar/{name}'] = dataset[:]
                            
                            # Load aggregated features
                            if 'aggregated' in label_group:
                                for name, dataset in label_group['aggregated'].items():
                                    config_dict[f'aggregated/{name}'] = dataset[:]
                            
                            labels_dict[config_name] = config_dict
                    
                    # Get normalization parameters
                    norm_params = json.loads(signal_group.attrs['normalization_params'])
                    
                    # Create TensorFlow dataset
                    feature_arrays = []
                    for name in sorted(features_dict.keys()):  # Sort for consistent order
                        feature_arrays.append(features_dict[name])
                    
                    if labels_dict and include_labels:
                        label_arrays = []
                        for config_name in sorted(labels_dict.keys()):
                            config_arrays = []
                            for name in sorted(labels_dict[config_name].keys()):
                                config_arrays.append(labels_dict[config_name][name])
                            label_arrays.extend(config_arrays)
                        
                        # Create dataset with features and labels
                        dataset = tf.data.Dataset.from_tensor_slices((
                            tuple(feature_arrays),
                            tuple(label_arrays)
                        ))
                        
                        # Apply normalization (features only)
                        def normalize_data(features, labels):
                            normalized_features = []
                            for i, feature in enumerate(features):
                                if f'scalar' in norm_params['features']:
                                    name = sorted(features_dict.keys())[i]
                                    if name in norm_params['features']['scalar']:
                                        params = norm_params['features']['scalar'][name]
                                        feature = (feature - params['mean']) / params['std']
                                elif f'aggregated' in norm_params['features']:
                                    name = sorted(features_dict.keys())[i]
                                    if name in norm_params['features']['aggregated']:
                                        params = norm_params['features']['aggregated'][name]
                                        feature = (feature - params['means']) / params['stds']
                                normalized_features.append(feature)
                            return tuple(normalized_features), labels
                        
                        dataset = dataset.map(normalize_data)
                    
                    else:
                        # Create dataset with features only
                        dataset = tf.data.Dataset.from_tensor_slices(tuple(feature_arrays))
                        
                        # Apply normalization
                        def normalize_features(*features):
                            normalized_features = []
                            for i, feature in enumerate(features):
                                if f'scalar' in norm_params['features']:
                                    name = sorted(features_dict.keys())[i]
                                    if name in norm_params['features']['scalar']:
                                        params = norm_params['features']['scalar'][name]
                                        feature = (feature - params['mean']) / params['std']
                                elif f'aggregated' in norm_params['features']:
                                    name = sorted(features_dict.keys())[i]
                                    if name in norm_params['features']['aggregated']:
                                        params = norm_params['features']['aggregated'][name]
                                        feature = (feature - params['means']) / params['stds']
                                normalized_features.append(feature)
                            return tuple(normalized_features)
                        
                        dataset = dataset.map(normalize_features)
                    
                    # Batch and prefetch
                    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                    signal_datasets[signal_key] = dataset
            
            # Store current dataset info
            self.current_dataset_id = dataset_id
            self.current_dataset_path = dataset_path
            self.current_dataset_info = self.get_dataset_info(dataset_id)
            
            logging.info(f"\nSuccessfully loaded {len(signal_datasets)} signal datasets")
            return signal_datasets
            
        except Exception as e:
            raise Exception(f"Failed to load signal datasets: {str(e)}")

    def _apply_feature_filters(self, 
                             feature_values: Dict[str, np.ndarray],
                             filters: List[PhysliteFeatureFilter]) -> bool:
        """
        Apply scalar feature filters to event-level features.
        
        Args:
            feature_values: Dictionary mapping feature names to their values
            filters: List of PhysliteFeatureFilter objects to apply
            
        Returns:
            bool: True if all filters pass, False otherwise
        """
        if not filters:
            return True
        
        for filter in filters:
            # Get the feature value using the branch name
            value = feature_values.get(filter.branch.name)
            if value is None:
                return False
            
            # Apply min/max filters if they exist
            if filter.min_value is not None and value < filter.min_value:
                return False
            if filter.max_value is not None and value > filter.max_value:
                return False
        
        return True

    def _apply_feature_array_filters(self,
                                   feature_arrays: Dict[str, np.ndarray],
                                   filters: List[PhysliteFeatureArrayFilter]) -> np.ndarray:
        """
        Apply array feature filters to track-level features.
        
        Args:
            feature_arrays: Dictionary mapping feature names to their array values
            filters: List of PhysliteFeatureArrayFilter objects to apply
            
        Returns:
            np.ndarray: Boolean mask indicating which array elements pass all filters
        """
        if not filters:
            # If no filters, accept all elements
            sample_array = next(iter(feature_arrays.values()))
            return np.ones(len(sample_array), dtype=bool)
        
        # Start with all True mask
        mask = np.ones(len(next(iter(feature_arrays.values()))), dtype=bool)
        
        for filter in filters:
            # Get the feature array using the branch name
            values = feature_arrays.get(filter.branch.name)
            if values is None:
                # If feature is missing, reject all elements
                mask[:] = False
                break
            
            # Apply min filter if it exists
            if filter.min_value is not None:
                mask &= (values >= filter.min_value)
            
            # Apply max filter if it exists
            if filter.max_value is not None:
                mask &= (values <= filter.max_value)
        
        return mask

    def _apply_feature_array_aggregator(self,
                                      feature_arrays: Dict[str, np.ndarray],
                                      aggregator: PhysliteFeatureArrayAggregator,
                                      valid_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply feature array aggregator to combine and sort multiple array features.
        
        Args:
            feature_arrays: Dictionary mapping feature names to their array values
            aggregator: PhysliteFeatureArrayAggregator configuration
            valid_mask: Boolean mask indicating which array elements passed filters
            
        Returns:
            np.ndarray: Aggregated and sorted array of shape (max_length, n_features) 
                       or None if requirements not met
        """
        # Get the number of valid elements after filtering
        n_valid = np.sum(valid_mask)
        
        # Check if we meet minimum length requirement
        if n_valid < aggregator.min_length:
            return None
        
        # Extract arrays for each input branch
        feature_list = []
        for filter in aggregator.input_branches:
            values = feature_arrays.get(filter.branch.name)
            if values is None:
                return None
            # Apply mask and reshape to column
            filtered_values = values[valid_mask].reshape(-1, 1)
            feature_list.append(filtered_values)
        
        # Get sorting values
        sort_values = feature_arrays.get(aggregator.sort_by_branch.branch.name)
        if sort_values is None:
            return None
        sort_values = sort_values[valid_mask]
        
        # Stack features horizontally
        features = np.hstack(feature_list)  # Shape: (n_valid, n_features)
        
        # Sort by specified branch
        sort_indices = np.argsort(sort_values)[::-1]  # Descending order
        sorted_features = features[sort_indices]
        
        # Handle length requirements
        if len(sorted_features) > aggregator.max_length:
            # Truncate
            final_features = sorted_features[:aggregator.max_length]
        else:
            # Pad with zeros
            padding = np.zeros((aggregator.max_length - len(sorted_features), features.shape[1]))
            final_features = np.vstack([sorted_features, padding])
        
        return final_features