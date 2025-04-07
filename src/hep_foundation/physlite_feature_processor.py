from pathlib import Path
import h5py
import json
from typing import Dict, Optional, Tuple, List, Union
import tensorflow as tf
import numpy as np
from datetime import datetime
import uproot
import logging
import os

from hep_foundation.task_config import (
    PhysliteFeatureFilter, 
    PhysliteFeatureArrayFilter, 
    PhysliteFeatureArrayAggregator, 
    PhysliteSelectionConfig, 
    PhysliteFeatureSelector, 
    TaskConfig
)
from hep_foundation.logging_config import setup_logging

class PhysliteFeatureProcessor:
    """
    Handles processing, filtering, and aggregation of ATLAS PhysLite features.
    
    This class is responsible for:
    - Applying event and feature filters
    - Processing and aggregating feature arrays
    - Computing normalization parameters
    - Creating normalized datasets
    - Loading features and labels from HDF5 files
    """
    
    def __init__(self):
        """Initialize the PhysliteFeatureProcessor."""
        # Setup logging at INFO level
        setup_logging()
        
        # Could add configuration parameters here if needed in the future
        # For now, keeping it simple as the class is primarily stateless
        pass

    def get_required_branches(self, task_config: TaskConfig) -> set:
        """
        Get set of all required branch names for a given task configuration.
        
        Args:
            task_config: TaskConfig object containing event filters, input features, and labels
            
        Returns:
            set: Set of branch names required for processing
        """
        required_branches = set()
        
        # Add event filter branches
        for filter in task_config.event_filters:
            required_branches.add(filter.branch.name)

        # Process all selection configs (input and labels)
        for selection_config in [task_config.input, *task_config.labels]:
            # Add feature selector branches
            for selector in selection_config.feature_selectors:
                required_branches.add(selector.branch.name)
            
            # Add aggregator branches
            for aggregator in selection_config.feature_array_aggregators:
                # Add input branches
                for selector in aggregator.input_branches:
                    required_branches.add(selector.branch.name)
                # Add filter branches
                for filter in aggregator.filter_branches:
                    required_branches.add(filter.branch.name)
                # Add sort branch if present
                if aggregator.sort_by_branch:
                    required_branches.add(aggregator.sort_by_branch.branch.name)
                    
        return required_branches

    # functions brought over from dataset_manager.py

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
                                      valid_mask: np.ndarray,
                                      sort_indices: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply feature array aggregator to combine and sort multiple array features.
        
        Args:
            feature_arrays: Dictionary mapping feature names to their array values
            aggregator: PhysliteFeatureArrayAggregator configuration
            valid_mask: Boolean mask indicating which array elements passed filters
            sort_indices: Indices to use for sorting the filtered arrays
            
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
        for selector in aggregator.input_branches:
            values = feature_arrays.get(selector.branch.name)
            if values is None:
                return None
            # Apply mask and reshape to column
            filtered_values = values[valid_mask].reshape(-1, 1)
            feature_list.append(filtered_values)
        
        # Stack features horizontally
        features = np.hstack(feature_list)  # Shape: (n_valid, n_features)
        
        # Apply sorting using provided indices
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
        if selection_config.feature_selectors:
            scalar_features = self._extract_selected_features(
                event_data,
                selection_config.feature_selectors
            )
        else:
            scalar_features = {}
        
        # Process aggregators
        aggregated_features = {}
        if selection_config.feature_array_aggregators:
            for idx, aggregator in enumerate(selection_config.feature_array_aggregators):
                # Get all needed arrays first
                array_features = {}
                
                # Get input branch arrays
                for selector in aggregator.input_branches:
                    value = event_data.get(selector.branch.name)
                    if value is None or len(value) == 0:
                        return None
                    array_features[selector.branch.name] = value
                
                # Apply filters to get valid mask
                valid_mask = self._apply_feature_array_filters(
                    array_features,
                    aggregator.filter_branches
                )
                
                # Get sort values and indices if sort_by_branch is specified
                if aggregator.sort_by_branch:
                    sort_value = event_data.get(aggregator.sort_by_branch.branch.name)
                    if sort_value is None or len(sort_value) == 0:
                        return None
                    array_features[aggregator.sort_by_branch.branch.name] = sort_value
                    # Create sort indices in descending order
                    sort_indices = np.argsort(sort_value[valid_mask])[::-1]
                else:
                    # If no sorting specified, keep original order
                    sort_indices = np.arange(np.sum(valid_mask))
                
                # Apply aggregator with optional sorting
                result = self._apply_feature_array_aggregator(
                    array_features,
                    aggregator,
                    valid_mask,
                    sort_indices
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
                      task_config: TaskConfig) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single event using the task configuration.
        
        Args:
            event_data: Dictionary mapping branch names to their raw values
            task_config: TaskConfig object containing event filters, input features, and labels
            
        Returns:
            Dictionary of processed features (scalar, array, and labels) or None if event rejected
        """
        # Apply event filters first
        if not self._apply_event_filters(event_data, task_config.event_filters):
            return None
        
        # Process input selection config
        input_features = self._process_selection_config(event_data, task_config.input)
        if input_features is None:
            return None
        
        # Process each label config if present
        label_features = []  # Change to list since we can have multiple label configs
        for label_config in task_config.labels:
            label_result = self._process_selection_config(event_data, label_config)
            if label_result is None:
                return None
            label_features.append(label_result)  # Append the whole result
        
        return {
            'scalar_features': input_features['scalar_features'],
            'aggregated_features': input_features['aggregated_features'],
            'label_features': label_features  # Now a list of dicts, each with scalar_features and aggregated_features
        }

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

        if signal_key:
            logging.info(f"\nProcessing signal data for {signal_key}")
        elif run_number:
            logging.info(f"\nProcessing ATLAS data for run {run_number}")
        else:
            raise ValueError("Must provide either run_number or signal_key")
        
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
        required_branches = self.get_required_branches(task_config)
        
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
                    logging.info(f"DEBUG: Starting to process tree with branches: {required_branches}")
                    
                    # Read all required branches
                    for arrays in tree.iterate(required_branches, library="np", step_size=1000):
                        logging.info(f"DEBUG: Processing batch with {len(next(iter(arrays.values())))} events")
                        try:
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
                                    # Extract the components from the result dictionary
                                    input_features = {
                                        'scalar_features': result['scalar_features'],
                                        'aggregated_features': result['aggregated_features']
                                    }
                                    processed_inputs.append(input_features)
                                    processed_labels.append(result['label_features'])
                                    
                                    catalog_stats['processed'] += 1
                                    
                                    # Update feature statistics
                                    if input_features['aggregated_features']:
                                        first_aggregator = next(iter(input_features['aggregated_features'].values()))
                                        stats['total_features'] += np.sum(np.any(first_aggregator != 0, axis=1))
                        except Exception as e:
                            logging.error(f"DEBUG: Error in batch processing: {str(e)}")
                            raise
                
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

    def _create_normalized_dataset(self,
                                features_dict: Dict[str, np.ndarray],
                                norm_params: Dict,
                                labels_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None) -> tf.data.Dataset:
        """
        Create a normalized TensorFlow dataset from features and optional labels.
        
        Args:
            features_dict: Dictionary mapping feature names to their array values
            norm_params: Dictionary containing normalization parameters
            labels_dict: Optional dictionary mapping label config names to their feature dictionaries
            
        Returns:
            tf.data.Dataset containing normalized features (and labels if provided)
        """
        # Get number of events from first feature
        n_events = next(iter(features_dict.values())).shape[0]
        
        # Normalize and reshape features
        normalized_features = []
        for name in sorted(features_dict.keys()):
            feature_array = features_dict[name]
            if name.startswith('scalar/'):
                params = norm_params['features']['scalar'][name.split('/', 1)[1]]
                normalized = (feature_array - params['mean']) / params['std']
            else:  # aggregated features
                params = norm_params['features']['aggregated'][name.split('/', 1)[1]]
                normalized = (feature_array - np.array(params['means'])) / np.array(params['stds'])
            normalized_features.append(normalized.reshape(n_events, -1))
        
        # Concatenate normalized features
        all_features = np.concatenate(normalized_features, axis=1)
        
        if labels_dict:
            # Normalize and reshape labels
            normalized_labels = []
            for config_name in sorted(labels_dict.keys()):
                config_features = labels_dict[config_name]
                for name in sorted(config_features.keys()):
                    label_array = config_features[name]
                    if name.startswith('scalar/'):
                        params = norm_params['labels'][int(config_name.split('_')[1])]['scalar'][name.split('/', 1)[1]]
                        normalized = (label_array - params['mean']) / params['std']
                    else:  # aggregated features
                        params = norm_params['labels'][int(config_name.split('_')[1])]['aggregated'][name.split('/', 1)[1]]
                        normalized = (label_array - np.array(params['means'])) / np.array(params['stds'])
                    normalized_labels.append(normalized.reshape(n_events, -1))
            
            # Concatenate normalized labels
            all_labels = np.concatenate(normalized_labels, axis=1)
            return tf.data.Dataset.from_tensor_slices((all_features, all_labels))
        else:
            return tf.data.Dataset.from_tensor_slices(all_features)

    def _load_features_from_group(self, features_group: h5py.Group) -> Dict[str, np.ndarray]:
        """Load features from an HDF5 group."""
        features_dict = {}
        
        # Load scalar features
        if 'scalar' in features_group:
            for name, dataset in features_group['scalar'].items():
                features_dict[f'scalar/{name}'] = dataset[:]
        
        # Load aggregated features
        if 'aggregated' in features_group:
            for name, dataset in features_group['aggregated'].items():
                features_dict[f'aggregated/{name}'] = dataset[:]
            
        return features_dict

    def _load_labels_from_group(self, labels_group: h5py.Group) -> Dict[str, Dict[str, np.ndarray]]:
        """Load labels from an HDF5 group."""
        labels_dict = {}
        
        for config_name, label_group in labels_group.items():
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
        
        return labels_dict


