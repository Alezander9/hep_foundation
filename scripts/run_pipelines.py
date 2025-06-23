#!/usr/bin/env python3
"""
Pipeline Catalog Processor

This script processes YAML configuration files from a catalog_stack directory,
running the full foundation model pipeline for each configuration.

Features:
- Processes all YAML files in catalog_stack/ directory
- Runs full pipeline (train → regression → anomaly) for each config
- Saves logs to logs/ directory (one per catalog)
- Deletes processed catalogs from stack (configs are saved in experiment folders)
- Supports dry-run mode for testing
- Handles errors gracefully and continues processing
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from hep_foundation.config.config_loader import load_pipeline_config
from hep_foundation.config.logging_config import get_logger
from hep_foundation.training.foundation_model_pipeline import FoundationModelPipeline


class PipelineCatalogProcessor:
    """Processes pipeline configuration catalogs from a stack directory."""
    
    def __init__(self, catalog_stack_dir: str = "catalog_stack", logs_dir: str = "logs"):
        """
        Initialize the catalog processor.
        
        Args:
            catalog_stack_dir: Directory containing YAML configuration files to process
            logs_dir: Directory to save log files
        """
        self.catalog_stack_dir = Path(catalog_stack_dir)
        self.logs_dir = Path(logs_dir)
        self.logger = get_logger(__name__)
        
        # Create directories if they don't exist
        self.catalog_stack_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Catalog processor initialized")
        self.logger.info(f"  Catalog stack: {self.catalog_stack_dir.absolute()}")
        self.logger.info(f"  Logs directory: {self.logs_dir.absolute()}")
    
    def find_catalog_files(self) -> List[Path]:
        """Find all YAML configuration files in the catalog stack."""
        yaml_patterns = ["*.yaml", "*.yml"]
        catalog_files = []
        
        for pattern in yaml_patterns:
            catalog_files.extend(self.catalog_stack_dir.glob(pattern))
        
        # Sort by modification time (oldest first) for consistent processing order
        catalog_files.sort(key=lambda x: x.stat().st_mtime)
        
        self.logger.info(f"Found {len(catalog_files)} catalog files to process")
        for i, catalog_file in enumerate(catalog_files, 1):
            self.logger.info(f"  {i}. {catalog_file.name}")
        
        return catalog_files
    
    def load_catalog_config(self, catalog_path: Path) -> dict:
        """
        Load pipeline configuration from a catalog file.
        
        Args:
            catalog_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing all configuration objects
        """
        self.logger.info(f"Loading configuration from: {catalog_path}")
        
        try:
            config = load_pipeline_config(catalog_path)
            
            # Extract all the config objects
            catalog_config = {
                "dataset_config": config["dataset_config"],
                "vae_model_config": config["vae_model_config"],
                "dnn_model_config": config["dnn_model_config"],
                "vae_training_config": config["vae_training_config"],
                "dnn_training_config": config["dnn_training_config"],
                "task_config": config["task_config"],
                "source_config_file": config.get("_source_config_file"),
                "evaluation_config": config.get("evaluation_config"),
            }
            
            self.logger.info("Configuration loaded successfully")
            return catalog_config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {catalog_path}: {str(e)}")
            raise
    
    def setup_catalog_logging(self, catalog_name: str) -> Tuple[Path, logging.FileHandler]:
        """
        Set up logging for a specific catalog.
        
        Args:
            catalog_name: Name of the catalog (without extension)
            
        Returns:
            Tuple of (log_file_path, file_handler)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"pipeline_{catalog_name}_{timestamp}.log"
        
        # Create file handler for this catalog
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        self.logger.info(f"Logging for catalog '{catalog_name}' to: {log_file}")
        return log_file, file_handler
    
    def cleanup_catalog_logging(self, file_handler: logging.FileHandler):
        """Clean up logging handler for a catalog."""
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()
    
    def process_catalog(self, catalog_path: Path, dry_run: bool = False) -> bool:
        """
        Process a single catalog file.
        
        Args:
            catalog_path: Path to the catalog YAML file
            dry_run: If True, don't actually run the pipeline or delete the catalog
            
        Returns:
            True if processing was successful, False otherwise
        """
        catalog_name = catalog_path.stem
        
        self.logger.info("=" * 100)
        self.logger.info(f"PROCESSING CATALOG: {catalog_name}")
        self.logger.info("=" * 100)
        
        # Set up logging for this catalog
        log_file, file_handler = self.setup_catalog_logging(catalog_name)
        
        try:
            # Load configuration
            catalog_config = self.load_catalog_config(catalog_path)
            
            if dry_run:
                self.logger.info("DRY RUN: Would run pipeline with this configuration")
                self.logger.info(f"Dataset runs: {catalog_config['dataset_config'].run_numbers}")
                self.logger.info(f"Signal keys: {catalog_config['dataset_config'].signal_keys}")
                self.logger.info(f"VAE epochs: {catalog_config['vae_training_config'].epochs}")
                self.logger.info(f"DNN epochs: {catalog_config['dnn_training_config'].epochs}")
                return True
            
            # Initialize pipeline
            pipeline = FoundationModelPipeline()
            
            # Set source config file for reproducibility
            if catalog_config.get("source_config_file"):
                pipeline.set_source_config_file(catalog_config["source_config_file"])
            
            # Get evaluation configuration
            evaluation_config = catalog_config.get("evaluation_config")
            data_sizes = evaluation_config.regression_data_sizes if evaluation_config else [1000, 2000, 5000]
            fixed_epochs = evaluation_config.fixed_epochs if evaluation_config else 10
            
            # Run the full pipeline
            self.logger.info("Starting full pipeline execution...")
            success = pipeline.run_full_pipeline(
                dataset_config=catalog_config["dataset_config"],
                task_config=catalog_config["task_config"],
                vae_model_config=catalog_config["vae_model_config"],
                dnn_model_config=catalog_config["dnn_model_config"],
                vae_training_config=catalog_config["vae_training_config"],
                dnn_training_config=catalog_config["dnn_training_config"],
                delete_catalogs=True,  # Clean up intermediate files
                data_sizes=data_sizes,
                fixed_epochs=fixed_epochs,
            )
            
            if success:
                self.logger.info(f"Successfully completed pipeline for catalog: {catalog_name}")
                
                # Delete the catalog file from the stack
                self.logger.info(f"Removing processed catalog from stack: {catalog_path}")
                catalog_path.unlink()
                
                return True
            else:
                self.logger.error(f"Pipeline failed for catalog: {catalog_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing catalog {catalog_name}: {type(e).__name__}: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            return False
            
        finally:
            # Clean up logging
            self.cleanup_catalog_logging(file_handler)
            self.logger.info(f"Log file for catalog '{catalog_name}': {log_file}")
    
    def run(self, dry_run: bool = False, max_catalogs: int = None) -> bool:
        """
        Run the catalog processor.
        
        Args:
            dry_run: If True, don't actually run pipelines or delete catalogs
            max_catalogs: Maximum number of catalogs to process (None for all)
            
        Returns:
            True if all catalogs were processed successfully
        """
        self.logger.info("=" * 100)
        self.logger.info("STARTING PIPELINE CATALOG PROCESSOR")
        if dry_run:
            self.logger.info("DRY RUN MODE - No pipelines will be executed")
        self.logger.info("=" * 100)
        
        # Find all catalog files
        catalog_files = self.find_catalog_files()
        
        if not catalog_files:
            self.logger.warning("No catalog files found in the stack directory")
            return True
        
        # Limit number of catalogs if specified
        if max_catalogs is not None:
            catalog_files = catalog_files[:max_catalogs]
            self.logger.info(f"Processing limited to first {max_catalogs} catalogs")
        
        # Process each catalog
        successful_count = 0
        failed_count = 0
        
        for i, catalog_path in enumerate(catalog_files, 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"CATALOG {i}/{len(catalog_files)}: {catalog_path.name}")
            self.logger.info(f"{'='*50}")
            
            success = self.process_catalog(catalog_path, dry_run=dry_run)
            
            if success:
                successful_count += 1
                self.logger.info(f"✓ Catalog {i} completed successfully")
            else:
                failed_count += 1
                self.logger.error(f"✗ Catalog {i} failed")
        
        # Final summary
        self.logger.info("\n" + "=" * 100)
        self.logger.info("CATALOG PROCESSING SUMMARY")
        self.logger.info("=" * 100)
        self.logger.info(f"Total catalogs processed: {len(catalog_files)}")
        self.logger.info(f"Successful: {successful_count}")
        self.logger.info(f"Failed: {failed_count}")
        
        if failed_count == 0:
            self.logger.info("🎉 All catalogs processed successfully!")
            return True
        else:
            self.logger.warning(f"⚠️  {failed_count} catalog(s) failed processing")
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process foundation model pipeline catalogs from a stack directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipelines.py                    # Process all catalogs in catalog_stack/
  python scripts/run_pipelines.py --dry-run          # Preview what would be processed
  python scripts/run_pipelines.py --max-catalogs 3   # Process only first 3 catalogs
  python scripts/run_pipelines.py --catalog-dir my_configs  # Use custom catalog directory
        """
    )
    
    parser.add_argument(
        "--catalog-dir",
        type=str,
        default="catalog_stack",
        help="Directory containing YAML configuration files (default: catalog_stack)"
    )
    
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory to save log files (default: logs)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview catalogs without running pipelines or deleting files"
    )
    
    parser.add_argument(
        "--max-catalogs",
        type=int,
        default=None,
        help="Maximum number of catalogs to process (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = PipelineCatalogProcessor(
            catalog_stack_dir=args.catalog_dir,
            logs_dir=args.logs_dir
        )
        
        # Run processor
        success = processor.run(
            dry_run=args.dry_run,
            max_catalogs=args.max_catalogs
        )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 