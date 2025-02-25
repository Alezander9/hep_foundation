from pathlib import Path
from typing import Optional
from tqdm import tqdm
import requests
import json
import logging
import sys
from hep_foundation.utils import ATLAS_CATALOG_COUNTS, SIGNAL_CATALOGS

class ATLASDataManager:
    """Manages ATLAS PHYSLITE data access"""
    
    # Add version as a class attribute
    VERSION = "1.0.0"  # Major.Minor.Patch format
    
    def __init__(self, base_dir: str = "atlas_data"):
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.base_dir = Path(base_dir)
        self.base_url = "https://opendata.cern.ch/record/80001/files"
        self.signal_base_url = "https://opendata.cern.ch/record/80011/files"
        self._setup_directories()
        
        # Cache for catalog counts
        self.catalog_counts = {}  # For ATLAS data
        self.signal_catalog_counts = {}  # For signal data
        
        # Signal types mapping
        self.signal_types = SIGNAL_CATALOGS
    
    def get_version(self) -> str:
        """Return the version of the ATLASDataManager"""
        return self.VERSION
    
    def get_catalog_count(self, run_number: str) -> int:
        """
        Discover how many catalog files exist for a run by probing the server
        
        Args:
            run_number: ATLAS run number
            
        Returns:
            Number of available catalog files
        """
        return ATLAS_CATALOG_COUNTS[run_number]
    
    def download_run_catalog(self, run_number: str, index: int = 0) -> Optional[Path]:
        """
        Download a specific run catalog file.
        
        Args:
            run_number: ATLAS run number
            index: Catalog index
            
        Returns:
            Path to the downloaded catalog file or None if file doesn't exist
        """
        padded_run = run_number.zfill(8)
        url = f"/record/80001/files/data16_13TeV_Run_{padded_run}_file_index.json_{index}"
        output_path = self.base_dir / "catalogs" / f"Run_{run_number}_catalog_{index}.root"
        
        try:
            if self._download_file(url, output_path, f"Downloading catalog {index} for Run {run_number}"):
                return output_path
        except Exception as e:
            logging.error(f"Failed to download catalog {index} for run {run_number}: {str(e)}")
            if output_path.exists():
                output_path.unlink()  # Clean up partial download
            return None
    
    def _setup_directories(self):
        """Create necessary directory structure"""
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "catalogs").mkdir(exist_ok=True)
        (self.base_dir / "signal_catalogs").mkdir(exist_ok=True)
    
    def _download_file(self, url: str, output_path: Path, desc: str = None) -> bool:
        """Download a single file if it doesn't exist"""
        if output_path.exists():
            return False
        
        logging.info(f"Downloading file: {url}")    
        response = requests.get(f"https://opendata.cern.ch{url}", stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            # Check if output is interactive
            is_interactive = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
            
            if is_interactive:
                # Use normal progress bar for interactive terminal
                bar_format = None
                mininterval = 0.1
            else:
                # Use simplified progress for log files
                bar_format = '{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                mininterval = 30
            
            with open(output_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                mininterval=mininterval,
                bar_format=bar_format
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            logging.info(f"Download complete: {desc}")
            return True
        else:
            raise Exception(f"Download failed with status code: {response.status_code}")
    
    def get_run_catalog_path(self, run_number: str, index: int = 0) -> Path:
        """Get path to a run catalog file"""
        return self.base_dir / "catalogs" / f"Run_{run_number}_catalog_{index}.root"
    
    def get_signal_catalog_path(self, signal_key: str, index: int = 0) -> Path:
        """Get path to a signal catalog file"""
        return self.base_dir / "signal_catalogs" / f"{signal_key}_catalog_{index}.root"
    
    def get_signal_catalog_count(self, signal_key: str) -> int:
        """Discover how many catalog files exist for a signal type"""
        if signal_key not in self.signal_types:
            raise ValueError(f"Unknown signal key: {signal_key}. Available keys: {list(self.signal_types.keys())}")
            
        if signal_key in self.signal_catalog_counts:
            return self.signal_catalog_counts[signal_key]
            
        signal_name = self.signal_types[signal_key]
        index = 0
        
        while True:
            url = f"/record/80011/files/{signal_name}_file_index.json_{index}"
            response = requests.head(f"https://opendata.cern.ch{url}")
            
            if response.status_code != 200:
                break
                
            index += 1
        
        self.signal_catalog_counts[signal_key] = index
        return index
    
    def download_signal_catalog(self, signal_key: str, index: int = 0) -> Optional[Path]:
        """Download a specific signal catalog file"""
        if signal_key not in self.signal_types:
            raise ValueError(f"Unknown signal key: {signal_key}. Available keys: {list(self.signal_types.keys())}")
            
        signal_name = self.signal_types[signal_key]
        url = f"/record/80011/files/{signal_name}_file_index.json_{index}"
        output_path = self.base_dir / "signal_catalogs" / f"{signal_key}_catalog_{index}.root"
        
        try:
            if self._download_file(url, output_path, f"Downloading catalog {index} for signal {signal_key}"):
                return output_path
            return output_path if output_path.exists() else None
        except Exception as e:
            logging.error(f"Failed to download {signal_key} catalog {index}: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            return None
    
