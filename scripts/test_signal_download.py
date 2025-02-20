from hep_foundation.atlas_data_manager import ATLASDataManager
from hep_foundation.processed_dataset_manager import ProcessedDatasetManager

def test_signal_downloads():
    """Test signal catalog downloads and processing"""
    # First test basic downloads
    manager = ATLASDataManager()
    
    print("\nTesting signal catalog downloads:")
    print(f"Available signal keys: {list(manager.signal_types.keys())}")
    
    for signal_key in manager.signal_types.keys():
        print(f"\nTrying {signal_key}:")
        count = manager.get_signal_catalog_count(signal_key)
        print(f"Found {count} catalogs")
        
        if count > 0:
            path = manager.download_signal_catalog(signal_key, 0)
            if path:
                print(f"Successfully downloaded to: {path}")
            else:
                print("Download failed")

def test_signal_processing():
    """Test signal dataset creation and processing"""
    dataset_manager = ProcessedDatasetManager()
    
    # Create test configuration
    config = {
        'signal_types': ['zprime', 'wprime_qq'],  # Test with two signal types
        'track_selections': {
            'eta': (-2.5, 2.5),
            'chi2_per_ndof': (0.0, 10.0),
        },
        'event_selections': {},
        'max_tracks_per_event': 56,
        'min_tracks_per_event': 3,
        'catalog_limit': 1  # Limit to first catalog for testing
    }
    
    print("\nTesting signal dataset creation:")
    print(f"Configuration: {config}")
    
    try:
        dataset_id, dataset_path = dataset_manager._create_signal_dataset(
            config=config,
            plot_distributions=True
        )
        
        print("\nSignal dataset created successfully:")
        print(f"Dataset ID: {dataset_id}")
        print(f"Dataset path: {dataset_path}")
        
        # Verify dataset contents
        import h5py
        with h5py.File(dataset_path, 'r') as f:
            print("\nDataset attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
                
            print("\nDataset shape:", f['features'].shape)
            print("Number of events:", len(f['features']))
            
    except Exception as e:
        print(f"\nError creating signal dataset: {str(e)}")
        raise

def main():
    """Run all signal processing tests"""
    try:
        print("\n=== Testing Signal Downloads ===")
        test_signal_downloads()
        
        print("\n=== Testing Signal Processing ===")
        test_signal_processing()
        
        print("\nAll tests completed successfully!")
        return 0
    except Exception as e:
        print(f"\nTests failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 