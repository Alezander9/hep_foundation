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
from hep_foundation.atlas_data_manager import ATLASDataManager
from hep_foundation.selection_config import SelectionConfig
from hep_foundation.utils import TypeConverter, ConfigSerializer

class ProcessedDatasetManager:
    """Manages pre-processed ATLAS datasets with integrated processing capabilities"""
    
    def __init__(self, 
                 base_dir: str = "processed_datasets",
                 atlas_manager: Optional[ATLASDataManager] = None):
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
        self.atlas_manager = atlas_manager or ATLASDataManager()
        
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
    
    def _create_dataset(self, config: Dict, plot_distributions: bool = False, delete_catalogs: bool = True) -> Tuple[str, Path]:
        """Create new processed dataset from ATLAS data"""
        logging.info("Creating new dataset")

        dataset_id = self.generate_dataset_id(config)
        output_path = self.datasets_dir / f"{dataset_id}.h5"
        logging.info(f"\nGenerated dataset ID: {dataset_id}")
        logging.info(f"From config: {json.dumps(config, indent=2)}")
        
        try:
            # Save full configuration first
            config_path = self.save_dataset_config(dataset_id, config)
            logging.info(f"Saved configuration to: {config_path}")
            
            # Create selection config
            selection_config = SelectionConfig(
                max_tracks_per_event=config['max_tracks_per_event'],
                min_tracks_per_event=config['min_tracks_per_event'],
                track_selections=config.get('track_selections'),
                event_selections=config.get('event_selections')
            )
            
            # Process all runs
            all_events = []
            total_stats = {
                'total_events': 0,
                'processed_events': 0,
                'total_tracks': 0,
                'processing_time': 0
            }
            
            for run_number in config['run_numbers']:
                events, stats = self._process_data(
                    selection_config=selection_config,
                    run_number=run_number,
                    catalog_limit=config.get('catalog_limit'),
                    plot_distributions=plot_distributions,
                    delete_catalogs=delete_catalogs
                )
                all_events.extend(events)
                
                # Update total statistics
                for key in total_stats:
                    total_stats[key] += stats[key]
            
            if not all_events:
                raise ValueError("No events passed selection criteria")
            
            # Create HDF5 dataset
            with h5py.File(output_path, 'w') as f:
                features = f.create_dataset(
                    'features', 
                    data=np.stack(all_events),
                    chunks=True,
                    compression='gzip'
                )
                
                # Store all attributes
                attrs_dict = {
                    'config': json.dumps(config),
                    'creation_date': str(datetime.now()),
                    'dataset_id': dataset_id,
                    'normalization_params': json.dumps(self._compute_normalization(features[:])),
                    'processing_stats': json.dumps(total_stats)
                }
                
                for key, value in attrs_dict.items():
                    f.attrs[key] = value
            
            return dataset_id, output_path
            
        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                output_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise Exception(f"Dataset creation failed: {str(e)}")
    
    def _compute_normalization(self, data: np.ndarray) -> Dict:
        """Compute normalization parameters for dataset"""
        # Create mask for zero-padded values
        # Expand mask to match data shape
        mask = np.any(data != 0, axis=-1, keepdims=True)  # Shape: (N_events, max_tracks, 1)
        mask = np.broadcast_to(mask, data.shape)  # Shape: (N_events, max_tracks, 6)
        
        # Create masked array
        masked_data = np.ma.array(data, mask=~mask)
        
        # Compute stats along event and track dimensions
        means = np.ma.mean(masked_data, axis=(0,1)).data
        stds = np.ma.std(masked_data, axis=(0,1)).data
        
        # Ensure no zero standard deviations
        stds = np.maximum(stds, 1e-6)
        
        # Convert to Python types for JSON serialization
        return TypeConverter.to_python({
            'means': means,
            'stds': stds
        })
    
    def load_datasets(self, 
                     config: Dict,
                     validation_fraction: float = 0.15,
                     test_fraction: float = 0.15,
                     batch_size: int = 1000,
                     shuffle_buffer: int = 10000,
                     plot_distributions: bool = False,
                     delete_catalogs: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and split dataset into train/val/test"""
        logging.info("Attempting to load datasets")
        try:
            # Ensure config is properly formatted
            config = {
                'run_numbers': config['run_numbers'],
                'track_selections': config['track_selections'],
                'event_selections': config.get('event_selections', {}),
                'max_tracks_per_event': config['max_tracks_per_event'],
                'min_tracks_per_event': config.get('min_tracks_per_event', 1),
                'catalog_limit': config.get('catalog_limit', None)
            }
            
            # Generate dataset ID
            dataset_id = self.generate_dataset_id(config)
            dataset_path = self.datasets_dir / f"{dataset_id}.h5"
            logging.info(f"\nGenerated dataset ID: {dataset_id}")
            logging.info(f"From config: {json.dumps(config, indent=2)}")
            
            # Check if dataset exists, if not create it
            if not dataset_path.exists():
                logging.info(f"\nDataset not found, creating new dataset: {dataset_id}")
                dataset_id, dataset_path = self._create_dataset(
                    config=config,
                    plot_distributions=plot_distributions,
                    delete_catalogs=delete_catalogs
                )
            
            # Set current dataset tracking
            self.current_dataset_id = dataset_id
            self.current_dataset_path = dataset_path
            self.current_dataset_info = self.get_dataset_info(dataset_id)
            
            # Load and process dataset
            with h5py.File(dataset_path, 'r') as f:
                # Verify file integrity
                required_attrs = ['config', 'normalization_params', 'dataset_id']
                missing_attrs = [attr for attr in required_attrs if attr not in f.attrs]
                if missing_attrs:
                    raise ValueError(f"Dataset missing required attributes: {missing_attrs}")
                    
                # Convert data to float32 immediately when loading
                data = TypeConverter.to_numpy(f['features'][:], dtype=np.float32)
                stored_config = json.loads(f.attrs['config'])
                norm_params = json.loads(f.attrs['normalization_params'])
                
                if len(data) == 0:
                    raise ValueError("Dataset is empty")
            
            # Verify config matches
            if self.generate_dataset_id(stored_config) != dataset_id:
                raise ValueError("Dataset was created with different configuration")
            logging.info("Verified that saved config of loaded dataset matches desired config")

            # Convert normalization parameters to tensorflow tensors
            means = TypeConverter.to_tensorflow(norm_params['means'])
            stds = TypeConverter.to_tensorflow(norm_params['stds'])
            
            # Create TF dataset
            logging.info("Creating tf dataset...")
            dataset = tf.data.Dataset.from_tensor_slices(data)
            logging.info("Created tf dataset")

            # Normalize data
            logging.info("Normalizing data...")
            dataset = dataset.map(lambda x: (x - means) / stds)
            logging.info("Normalized data")
            
            # Calculate split sizes
            total_size = len(data)
            val_size = int(validation_fraction * total_size)
            test_size = int(test_fraction * total_size)
            train_size = total_size - val_size - test_size
            
            # Create splits
            logging.info("Creating dataset splits...")
            train_dataset = dataset.take(train_size).shuffle(
                buffer_size=shuffle_buffer,
                reshuffle_each_iteration=True
            ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            remaining = dataset.skip(train_size)
            val_dataset = remaining.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_dataset = remaining.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            logging.info("Created dataset splits")
            
            # After creating splits
            logging.info("\nDataset sizes:")
            total_events = 0
            for name, dataset in [("Training", train_dataset), 
                                 ("Validation", val_dataset), 
                                 ("Test", test_dataset)]:
                n_batches = sum(1 for _ in dataset)
                n_events = n_batches * batch_size
                total_events += n_events
                logging.info(f"{name}: {n_events} events ({n_batches} batches)")
            
            logging.info(f"Total events in datasets: {total_events}")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            raise Exception(f"Failed to load dataset: {str(e)}")

    def _process_event(self, 
                      event_tracks: Dict[str, np.ndarray], 
                      selection_config: SelectionConfig) -> Optional[np.ndarray]:
        """Process a single event's tracks with selections"""
        n_initial_tracks = len(event_tracks['d0'])
        
        # Calculate derived quantities
        track_features = {
            'pt': np.abs(1.0 / (event_tracks['qOverP'] * 1000)) * np.sin(event_tracks['theta']),
            'eta': -np.log(np.tan(event_tracks['theta'] / 2)),
            'phi': event_tracks['phi'],
            'd0': event_tracks['d0'],
            'z0': event_tracks['z0'],
            'chi2_per_ndof': event_tracks['chiSquared'] / event_tracks['numberDoF']
        }
        
        # Add event-level features
        event_features = {
            'n_total_tracks': len(track_features['pt']),
            'mean_pt': np.mean(track_features['pt']),
            'max_pt': np.max(track_features['pt'])
        }
        
        # Apply event-level selections
        if not selection_config.apply_event_selections(event_features):
            return None
        
        # Apply track-level selections
        good_tracks_mask = selection_config.apply_track_selections(track_features)
        good_tracks = np.where(good_tracks_mask)[0]
        
        # Check if we have enough tracks
        if len(good_tracks) < selection_config.min_tracks_per_event:
            return None
        
        # Sort by pT and take top N tracks
        track_pts = track_features['pt'][good_tracks]
        sorted_indices = np.argsort(track_pts)[::-1]
        top_tracks = good_tracks[sorted_indices[:selection_config.max_tracks_per_event]]
        
        # Create feature array
        features = np.column_stack([
            track_features['pt'][top_tracks],
            track_features['eta'][top_tracks],
            track_features['phi'][top_tracks],
            track_features['d0'][top_tracks],
            track_features['z0'][top_tracks],
            track_features['chi2_per_ndof'][top_tracks]
        ])
        
        # Pad if necessary
        if len(features) < selection_config.max_tracks_per_event:
            padding = np.zeros((selection_config.max_tracks_per_event - len(features), 6))
            features = np.vstack([features, padding])
            
        return features

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
                     selection_config: SelectionConfig,
                     run_number: Optional[str] = None,
                     signal_key: Optional[str] = None,
                     catalog_limit: Optional[int] = None,
                     plot_distributions: bool = False,
                     delete_catalogs: bool = True) -> Tuple[List[np.ndarray], Dict]:
        """Process either ATLAS or signal data using common code"""
        logging.info(f"\nProcessing {'signal' if signal_key else 'ATLAS'} data")
        
        # Initialize statistics
        stats = {
            'total_events': 0,
            'processed_events': 0,
            'total_tracks': 0,
            'processing_time': 0.0  # Explicitly use float
        }
        
        processed_events = []
        
        # Add statistics collection
        pre_selection_stats = {
            'tracks_per_event': [],
            'pt': [], 'eta': [], 'phi': [], 'd0': [], 'z0': [], 'chi2_per_ndof': []
        }
        post_selection_stats = {
            'tracks_per_event': [],
            'pt': [], 'eta': [], 'phi': [], 'd0': [], 'z0': [], 'chi2_per_ndof': []
        }
        # Get catalog paths (this will download the catalogs if they don't yet exist)
        logging.info(f"Getting catalog paths for run {run_number} and signal {signal_key}")
        catalog_paths = self._get_catalog_paths(run_number, signal_key, catalog_limit)
        logging.info(f"Found {len(catalog_paths)} catalog paths")
        
        for catalog_idx, catalog_path in enumerate(catalog_paths):
            logging.info(f"\nProcessing catalog {catalog_idx} with path: {catalog_path}")
            
            try:
                logging.info(f"Processing catalog {catalog_idx}")
                catalog_start_time = datetime.now()
                catalog_stats = {'events': 0, 'processed': 0}
                
                with uproot.open(catalog_path) as file:
                    tree = file["CollectionTree;1"]
                    branches = [
                        "InDetTrackParticlesAuxDyn.d0",
                        "InDetTrackParticlesAuxDyn.z0",
                        "InDetTrackParticlesAuxDyn.phi",
                        "InDetTrackParticlesAuxDyn.theta",
                        "InDetTrackParticlesAuxDyn.qOverP",
                        "InDetTrackParticlesAuxDyn.chiSquared",
                        "InDetTrackParticlesAuxDyn.numberDoF"
                    ]
                    
                    for arrays in tree.iterate(branches, library="np", step_size=1000):
                        catalog_stats['events'] += len(arrays["InDetTrackParticlesAuxDyn.d0"])
                        
                        for evt_idx in range(len(arrays["InDetTrackParticlesAuxDyn.d0"])):
                            raw_event = {
                                'd0': arrays["InDetTrackParticlesAuxDyn.d0"][evt_idx],
                                'z0': arrays["InDetTrackParticlesAuxDyn.z0"][evt_idx],
                                'phi': arrays["InDetTrackParticlesAuxDyn.phi"][evt_idx],
                                'theta': arrays["InDetTrackParticlesAuxDyn.theta"][evt_idx],
                                'qOverP': arrays["InDetTrackParticlesAuxDyn.qOverP"][evt_idx],
                                'chiSquared': arrays["InDetTrackParticlesAuxDyn.chiSquared"][evt_idx],
                                'numberDoF': arrays["InDetTrackParticlesAuxDyn.numberDoF"][evt_idx]
                            }
                            
                            if len(raw_event['d0']) == 0:
                                continue
                                
                            # Collect pre-selection statistics
                            track_features = {
                                'pt': np.abs(1.0 / (raw_event['qOverP'] * 1000)) * np.sin(raw_event['theta']),
                                'eta': -np.log(np.tan(raw_event['theta'] / 2)),
                                'phi': raw_event['phi'],
                                'd0': raw_event['d0'],
                                'z0': raw_event['z0'],
                                'chi2_per_ndof': raw_event['chiSquared'] / raw_event['numberDoF']
                            }
                            
                            pre_selection_stats['tracks_per_event'].append(len(track_features['pt']))
                            for feature, values in track_features.items():
                                pre_selection_stats[feature].extend(values)
                            
                            # Process event
                            processed_event = self._process_event(raw_event, selection_config)
                            if processed_event is not None:
                                processed_events.append(processed_event)
                                catalog_stats['processed'] += 1
                                stats['total_tracks'] += np.sum(np.any(processed_event != 0, axis=1))
                                
                                # Collect post-selection statistics
                                post_selection_stats['tracks_per_event'].append(
                                    np.sum(np.any(processed_event != 0, axis=1))
                                )
                                for i, feature in enumerate(['pt', 'eta', 'phi', 'd0', 'z0', 'chi2_per_ndof']):
                                    valid_tracks = processed_event[np.any(processed_event != 0, axis=1)]
                                    post_selection_stats[feature].extend(valid_tracks[:, i])
                
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
                
                catalog_idx += 1
                if catalog_limit and catalog_idx >= catalog_limit:
                    break
                    
            except Exception as e:
                logging.error(f"Error processing catalog {catalog_path}: {str(e)}")
                continue

            finally:
                # Delete catalog after processing
                if delete_catalogs:
                    if catalog_path.exists():
                        os.remove(catalog_path)

        if plot_distributions:
            logging.info(f"\nCollecting statistics for plotting...")
            logging.info(f"Pre-selection events: {len(pre_selection_stats['tracks_per_event'])}")
            logging.info(f"Post-selection events: {len(post_selection_stats['tracks_per_event'])}")
            
            plots_dir = self.base_dir / "plots" / (f"run_{run_number}" if run_number else f'signal_{signal_key}')
            self._plot_distributions(pre_selection_stats, post_selection_stats, plots_dir)
        
        # Convert stats to native Python types before returning
        return processed_events, {
            'total_events': int(stats['total_events']),
            'processed_events': int(stats['processed_events']),
            'total_tracks': int(stats['total_tracks']),
            'processing_time': float(stats['processing_time'])
        }

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

    def _create_signal_dataset(self, config: Dict, plot_distributions: bool = False) -> Tuple[str, Path]:
        """Create new processed dataset from ATLAS signal data"""
        logging.info(f"Creating new signal dataset")

        dataset_id = self.generate_dataset_id(config)
        output_path = self.datasets_dir / "signals" / f"{dataset_id}.h5"
        
        try:
            # Create signals directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = self.save_dataset_config(dataset_id, config)
            
            # Create selection config
            selection_config = SelectionConfig(
                max_tracks_per_event=config['max_tracks_per_event'],
                min_tracks_per_event=config['min_tracks_per_event'],
                track_selections=config.get('track_selections'),
                event_selections=config.get('event_selections')
            )
            
            # Process each signal type separately
            with h5py.File(output_path, 'w') as f:
                # Create group for each signal type
                for signal_key in config['signal_types']:
                    logging.info(f"\nProcessing signal type: {signal_key}")
                    events, stats = self._process_data(
                        selection_config=selection_config,
                        signal_key=signal_key,
                        catalog_limit=config.get('catalog_limit'),
                        plot_distributions=plot_distributions,
                        delete_catalogs=False
                    )
                    
                    if not events:
                        logging.info(f"Warning: No events passed selection for {signal_key}")
                        continue
                    
                    # Create dataset for this signal type
                    signal_group = f.create_group(signal_key)
                    features = signal_group.create_dataset(
                        'features',
                        data=np.stack(events),
                        chunks=True,
                        compression='gzip'
                    )
                    
                    # Store signal-specific stats
                    signal_group.attrs['processing_stats'] = json.dumps(stats)
                    signal_group.attrs['normalization_params'] = json.dumps(
                        self._compute_normalization(features[:])
                    )
                
                # Store global attributes
                f.attrs.update({
                    'config': json.dumps(config),
                    'creation_date': str(datetime.now()),
                    'dataset_id': dataset_id
                })
            
            return dataset_id, output_path
            
        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                output_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise Exception(f"Signal dataset creation failed: {str(e)}")

    def load_signal_datasets(
        self,
        config: Dict,
        batch_size: int = 1000,
        plot_distributions: bool = False
    ) -> Dict[str, tf.data.Dataset]:
        """
        Load signal datasets for evaluation
        
        Args:
            config: Configuration dictionary containing:
                - signal_types: List of signal keys to process
                - track_selections: Dict of track selection criteria
                - event_selections: Dict of event selection criteria
                - max_tracks_per_event: Maximum tracks to keep
                - min_tracks_per_event: Minimum tracks required
                - catalog_limit: Optional limit on catalogs to process
            batch_size: Size of batches in returned dataset
            plot_distributions: Whether to generate distribution plots
            
        Returns:
            Dictionary mapping signal_type to its corresponding TensorFlow dataset
        """
        logging.info(f"Loading signal datasets")
        try:
            # Format config and get paths
            config = {
                'signal_types': config['signal_types'],
                'track_selections': config['track_selections'],
                'event_selections': config.get('event_selections', {}),
                'max_tracks_per_event': config['max_tracks_per_event'],
                'min_tracks_per_event': config.get('min_tracks_per_event', 1),
                'catalog_limit': config.get('catalog_limit', None)
            }
            
            dataset_id = self.generate_dataset_id(config)
            dataset_path = self.datasets_dir / "signals" / f"{dataset_id}.h5"
            
            # Create if doesn't exist
            if not dataset_path.exists():
                dataset_id, dataset_path = self._create_signal_dataset(
                    config=config,
                    plot_distributions=plot_distributions
                )
            
            # Load datasets
            signal_datasets = {}
            with h5py.File(dataset_path, 'r') as f:
                for signal_key in config['signal_types']:
                    if signal_key not in f:
                        logging.warning(f"Warning: No data found for {signal_key}")
                        continue
                        
                    signal_group = f[signal_key]
                    # Explicitly cast features to float32
                    features = signal_group['features'][:].astype(np.float32)
                    normalization = json.loads(signal_group.attrs['normalization_params'])
                    
                    # Create dataset with explicit type
                    dataset = tf.data.Dataset.from_tensor_slices(
                        features
                    )
                    
                    # Ensure normalization parameters are float32
                    means = tf.constant(normalization['means'], dtype=tf.float32)
                    stds = tf.constant(normalization['stds'], dtype=tf.float32)
                    
                    # Apply normalization with explicit casting
                    dataset = dataset.map(
                        lambda x: (tf.cast(x, tf.float32) - means) / stds,
                        num_parallel_calls=tf.data.AUTOTUNE
                    )
                    
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
            logging.error(f"Error loading signal datasets: {str(e)}")
            raise