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

from hep_foundation.atlas_data_manager import ATLASDataManager
from hep_foundation.selection_config import SelectionConfig
from hep_foundation.utils import TypeConverter, ConfigSerializer

class ProcessedDatasetManager:
    """Manages pre-processed ATLAS datasets with integrated processing capabilities"""
    
    def __init__(self, 
                 base_dir: str = "processed_datasets",
                 atlas_manager: Optional[ATLASDataManager] = None):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.configs_dir = self.base_dir / "configs"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.atlas_manager = atlas_manager or ATLASDataManager()
        
    def generate_dataset_id(self, config: Dict) -> str:
        """Generate a human-readable dataset ID"""
        # Create a descriptive ID using key parameters
        run_str = '_'.join(str(run) for run in sorted(config['run_numbers']))
        # Add a short hash for uniqueness
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        return f"dataset_runs{run_str}_tracks{config['max_tracks_per_event']}_{config_hash}"
    
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
        if not config_path.exists():
            raise ValueError(f"No configuration found for dataset {dataset_id}")
            
        return ConfigSerializer.from_yaml(config_path)
    
    def create_dataset(self, config: Dict) -> Tuple[str, Path]:
        """Create new processed dataset from ATLAS data"""
        dataset_id = self.generate_dataset_id(config)
        output_path = self.datasets_dir / f"{dataset_id}.h5"
        
        if output_path.exists():
            # Verify the existing file is valid
            try:
                with h5py.File(output_path, 'r') as f:
                    required_attrs = ['config', 'creation_date', 'normalization_params', 'dataset_id']
                    if not all(attr in f.attrs for attr in required_attrs):
                        print(f"Existing dataset file is incomplete. Recreating...")
                        output_path.unlink()
                    else:
                        print(f"Dataset already exists: {output_path}")
                        return dataset_id, output_path
            except Exception as e:
                print(f"Error reading existing dataset: {e}")
                print("Recreating dataset...")
                output_path.unlink()
        
        print(f"Creating new dataset: {dataset_id}")
        
        try:
            # Save full configuration first
            config_path = self.save_dataset_config(dataset_id, config)
            print(f"Saved configuration to: {config_path}")
            
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
                events, stats = self._process_run_data(
                    run_number=run_number,
                    selection_config=selection_config,
                    catalog_limit=config.get('catalog_limit')
                )
                all_events.extend(events)
                
                # Update total statistics
                for key in total_stats:
                    total_stats[key] += stats[key]
            
            if not all_events:
                raise ValueError("No events passed selection criteria")
            
            # Create HDF5 dataset
            with h5py.File(output_path, 'w') as f:
                # Create and fill features dataset
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
                
            # Print final statistics
            print("\nDataset Creation Summary:")
            print(f"Total events processed: {total_stats['total_events']}")
            print(f"Events passing selection: {total_stats['processed_events']}")
            print(f"Total processing time: {total_stats['processing_time']:.1f}s")
            print(f"Average rate: {total_stats['total_events']/total_stats['processing_time']:.1f} events/s")
            print(f"Selection efficiency: {100 * total_stats['processed_events']/total_stats['total_events']:.1f}%")
            print(f"Average tracks per event: {total_stats['total_tracks']/total_stats['processed_events']:.2f}")
            
            return dataset_id, output_path
            
        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                output_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise Exception(f"Dataset creation failed: {str(e)}")
    
    def _save_buffer(self, dataset: h5py.Dataset, buffer: List[np.ndarray]):
        """Save batch of events to HDF5 dataset"""
        current_size = dataset.shape[0]
        new_size = current_size + len(buffer)
        dataset.resize(new_size, axis=0)
        dataset[current_size:new_size] = np.stack(buffer)
    
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
    
    def verify_dataset(self, dataset_id: str, config: Dict) -> bool:
        """Verify if existing dataset matches configuration"""
        try:
            stored_info = self.get_dataset_info(dataset_id)
            stored_config = stored_info['config']
            
            # Compare key parameters
            for key in ['run_numbers', 'track_selections', 'event_selections',
                       'max_tracks_per_event', 'min_tracks_per_event']:
                if stored_config[key] != config[key]:
                    return False
            return True
        except Exception:
            return False
    
    def recreate_dataset(self, dataset_id: str) -> Path:
        """Recreate dataset from stored configuration"""
        info = self.get_dataset_info(dataset_id)
        config = info['config']
        
        print(f"Recreating dataset {dataset_id} from stored configuration")
        print(f"Original creation date: {info['creation_date']}")
        print(f"Software versions used: {info['software_versions']}")
        
        _, new_path = self.create_dataset(config)
        return new_path
    
    def load_datasets(self, 
                     config: Dict,
                     validation_fraction: float = 0.15,
                     test_fraction: float = 0.15,
                     batch_size: int = 1000,
                     shuffle_buffer: int = 10000) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and split dataset into train/val/test"""
        try:
            dataset_id, dataset_path = self.create_dataset(config)
            
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
            
            # Convert normalization parameters to tensorflow tensors
            means = TypeConverter.to_tensorflow(norm_params['means'])
            stds = TypeConverter.to_tensorflow(norm_params['stds'])
            
            # Create TF dataset
            dataset = tf.data.Dataset.from_tensor_slices(data)
            
            # Normalize data
            dataset = dataset.map(lambda x: (x - means) / stds)
            
            # Calculate split sizes
            total_size = len(data)
            val_size = int(validation_fraction * total_size)
            test_size = int(test_fraction * total_size)
            train_size = total_size - val_size - test_size
            
            # Create splits
            train_dataset = dataset.take(train_size).shuffle(
                buffer_size=shuffle_buffer,
                reshuffle_each_iteration=True
            ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            remaining = dataset.skip(train_size)
            val_dataset = remaining.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            test_dataset = remaining.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
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

    def _process_run_data(self, 
                         run_number: str,
                         selection_config: SelectionConfig,
                         catalog_limit: Optional[int] = None) -> Tuple[List[np.ndarray], Dict]:
        """Process all events from a single run"""
        processed_events = []
        stats = {
            'total_events': 0,
            'processed_events': 0,
            'total_tracks': 0,
            'processing_time': 0.0  # Explicitly use float
        }
        
        print(f"\nProcessing run {run_number}")
        catalog_idx = 0
        
        while True:
            try:
                # Get or download catalog
                catalog_path = self.atlas_manager.get_run_catalog_path(run_number, catalog_idx)
                if not catalog_path.exists():
                    catalog_path = self.atlas_manager.download_run_catalog(run_number, catalog_idx)
                    if catalog_path is None:
                        break
                
                print(f"Processing catalog {catalog_idx}")
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
                                
                            processed_event = self._process_event(raw_event, selection_config)
                            if processed_event is not None:
                                processed_events.append(processed_event)
                                catalog_stats['processed'] += 1
                                stats['total_tracks'] += np.sum(np.any(processed_event != 0, axis=1))
                
                # Update statistics
                catalog_duration = (datetime.now() - catalog_start_time).total_seconds()
                stats['processing_time'] += catalog_duration
                stats['total_events'] += catalog_stats['events']
                stats['processed_events'] += catalog_stats['processed']
                
                # Print catalog summary
                print(f"Catalog {catalog_idx} summary:")
                print(f"  Events processed: {catalog_stats['events']}")
                print(f"  Events passing selection: {catalog_stats['processed']}")
                print(f"  Processing time: {catalog_duration:.1f}s")
                print(f"  Rate: {catalog_stats['events']/catalog_duration:.1f} events/s")
                
                catalog_idx += 1
                if catalog_limit and catalog_idx >= catalog_limit:
                    break
                    
            except Exception as e:
                print(f"Error processing catalog {catalog_idx}: {str(e)}")
                break
        
        # Convert stats to native Python types before returning
        return processed_events, {
            'total_events': int(stats['total_events']),
            'processed_events': int(stats['processed_events']),
            'total_tracks': int(stats['total_tracks']),
            'processing_time': float(stats['processing_time'])
        } 