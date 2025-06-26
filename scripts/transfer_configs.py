#!/usr/bin/env python3
"""
Streamlined Configuration Transfer Script

Transfers experiment configuration files to predefined target systems.

Usage:
  python scripts/transfer_configs.py nersc     # Transfer to NERSC
  python scripts/transfer_configs.py home     # Transfer to home computer
  python scripts/transfer_configs.py system   # Transfer to any defined system

Target systems are defined in .env file.
"""

import argparse
import sys
from pathlib import Path

from hep_foundation.config.config_loader import load_pipeline_config
from hep_foundation.config.logging_config import get_logger
from hep_foundation.utils.remote_transfer import TransferManager


class StreamlinedConfigTransfer:
    """Manages configuration transfers to predefined target systems."""

    def __init__(
        self,
        config_file: str = ".env",
        config_stack_dir: str = "_experiment_config_stack",
    ):
        """
        Initialize the transfer manager.

        Args:
            config_file: Path to the transfer configuration file
            config_stack_dir: Directory containing configuration files
        """
        self.logger = get_logger(__name__)
        self.config_stack_dir = Path(config_stack_dir)
        self.transfer_manager = TransferManager(config_file)

        # Ensure directories exist
        self.config_stack_dir.mkdir(exist_ok=True)

        self.logger.info("Streamlined Config Transfer initialized")
        self.logger.info(f"  Config stack: {self.config_stack_dir.absolute()}")
        self.logger.info(
            f"  Available systems: {list(self.transfer_manager.systems.keys())}"
        )

    def find_config_files(self) -> list[Path]:
        """Find all YAML configuration files in the config stack."""
        yaml_patterns = ["*.yaml", "*.yml"]
        config_files = []

        for pattern in yaml_patterns:
            config_files.extend(self.config_stack_dir.glob(pattern))

        # Sort by modification time for consistent ordering
        config_files.sort(key=lambda x: x.stat().st_mtime)

        self.logger.info(f"Found {len(config_files)} config files")
        for i, config_file in enumerate(config_files, 1):
            self.logger.info(f"  {i}. {config_file.name}")

        return config_files

    def validate_config_files(self, config_files: list[Path]) -> list[Path]:
        """Validate configuration files and return only valid ones."""
        self.logger.info("Validating configuration files...")

        valid_configs = []
        invalid_configs = []

        for config_file in config_files:
            try:
                load_pipeline_config(config_file)
                valid_configs.append(config_file)
                self.logger.debug(f"Config validation passed: {config_file.name}")
            except Exception as e:
                invalid_configs.append(config_file)
                self.logger.error(
                    f"Config validation failed: {config_file.name}: {str(e)}"
                )

        if invalid_configs:
            self.logger.warning(f"Found {len(invalid_configs)} invalid config files:")
            for invalid_config in invalid_configs:
                self.logger.warning(f"  - {invalid_config.name}")

        self.logger.info(f"{len(valid_configs)} valid config files ready for transfer")
        return valid_configs

    def transfer_configs(
        self, system: str, dry_run: bool = False, delete_after_transfer: bool = True
    ) -> bool:
        """
        Transfer configuration files to the specified system.

        Args:
            system: Target system name
            dry_run: If True, show what would be transferred without doing it
            delete_after_transfer: If True, delete config files after successful transfer

        Returns:
            True if successful
        """
        # Get transfer client
        transfer_client = self.transfer_manager.create_transfer_client(system)
        if not transfer_client:
            return False

        # Find and validate config files
        config_files = self.find_config_files()
        if not config_files:
            self.logger.warning("No config files found to transfer")
            return True

        valid_configs = self.validate_config_files(config_files)
        if not valid_configs:
            self.logger.error("No valid config files to transfer")
            return False

        # Transfer files using the shared transfer utility
        success = transfer_client.transfer_files_scp(
            files=valid_configs, dry_run=dry_run
        )

        if success and not dry_run and delete_after_transfer:
            self.logger.info("Deleting transferred config files from local stack...")
            for config_file in valid_configs:
                try:
                    config_file.unlink()
                    self.logger.info(f"Deleted local config file: {config_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete {config_file.name}: {e}")

        return success

    def list_systems(self):
        """List all available target systems."""
        self.transfer_manager.list_systems()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Transfer configuration files to predefined target systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/transfer_configs.py nersc           # Transfer to NERSC
  python scripts/transfer_configs.py home            # Transfer to home computer
  python scripts/transfer_configs.py nersc --dry-run # Preview transfer to NERSC
  python scripts/transfer_configs.py --list          # List available systems

System Configuration:
  Edit .env to define target systems with their connection details.
        """,
    )

    parser.add_argument(
        "system", nargs="?", help="Target system name (nersc, home, etc.)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be transferred without actually doing it",
    )

    parser.add_argument(
        "--list", action="store_true", help="List available target systems"
    )

    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep config files in local stack after transfer (default: delete after successful transfer)",
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="_experiment_config_stack",
        help="Local directory containing config files (default: _experiment_config_stack)",
    )

    args = parser.parse_args()

    try:
        # Initialize transfer manager
        manager = StreamlinedConfigTransfer(config_stack_dir=args.config_dir)

        if args.list:
            manager.list_systems()
            return

        if not args.system:
            parser.print_help()
            print(
                "\nError: Please specify a target system or use --list to see available systems"
            )
            sys.exit(1)

        # Transfer configs
        delete_after_transfer = not args.keep_local
        success = manager.transfer_configs(
            args.system.lower(),
            dry_run=args.dry_run,
            delete_after_transfer=delete_after_transfer,
        )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nTransfer interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
