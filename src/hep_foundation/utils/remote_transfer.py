"""
Shared Remote Transfer Utilities

This module provides reusable components for transferring files and directories
to remote systems via SSH/SCP/rsync.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hep_foundation.config.logging_config import get_logger


@dataclass
class SystemConfig:
    """Configuration for a target system."""

    name: str
    host: str
    user: str
    path: str
    ssh_key: str

    def __post_init__(self):
        # Expand SSH key path
        self.ssh_key = os.path.expanduser(self.ssh_key)


class ConfigLoader:
    """Loads system configurations from environment files."""

    def __init__(self, config_file: str = ".env"):
        self.logger = get_logger(__name__)
        self.config_file = Path(config_file)

    def load_system_configs(self) -> dict[str, SystemConfig]:
        """Load system configurations from the config file."""
        if not self.config_file.exists():
            self.logger.error(f"Config file not found: {self.config_file}")
            self.logger.error("Please create .env with system definitions.")
            return {}

        systems = {}

        try:
            with open(self.config_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse KEY=VALUE pairs
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        # Extract system name and setting
                        if "_" in key:
                            system_name, setting = key.split("_", 1)
                            system_name = system_name.lower()
                            setting = setting.lower()

                            if system_name not in systems:
                                systems[system_name] = {}

                            systems[system_name][setting] = value
                        else:
                            self.logger.warning(
                                f"Invalid config format at line {line_num}: {line}"
                            )

            # Convert to SystemConfig objects
            system_configs = {}
            for name, config_dict in systems.items():
                if self._validate_config_dict(name, config_dict):
                    system_configs[name] = SystemConfig(
                        name=name,
                        host=config_dict["host"],
                        user=config_dict["user"],
                        path=config_dict["path"],
                        ssh_key=config_dict["ssh_key"],
                    )

            self.logger.info(f"Loaded {len(system_configs)} system configurations")
            return system_configs

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}

    def _validate_config_dict(self, name: str, config_dict: dict[str, str]) -> bool:
        """Validate that a system config has all required fields."""
        required_fields = ["host", "user", "path", "ssh_key"]

        for field in required_fields:
            if field not in config_dict:
                self.logger.error(
                    f"Missing required field '{field}' for system '{name}'"
                )
                return False

        return True


class RemoteTransfer:
    """Handles file transfers to remote systems."""

    def __init__(self, system_config: SystemConfig):
        self.logger = get_logger(__name__)
        self.config = system_config

    def create_remote_directory(self, remote_path: Optional[str] = None) -> bool:
        """Create the remote directory structure."""
        target_path = remote_path or self.config.path

        try:
            cmd = [
                "ssh",
                "-i",
                self.config.ssh_key,
                f"{self.config.user}@{self.config.host}",
                f"mkdir -p {target_path}",
            ]

            self.logger.info(f"Creating remote directory: {target_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info("Remote directory created/verified")
                return True
            else:
                self.logger.error(f"Failed to create remote directory: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error creating remote directory: {e}")
            return False

    def transfer_files_scp(
        self,
        files: list[Path],
        remote_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> bool:
        """Transfer files using scp."""
        target_path = remote_path or self.config.path

        if dry_run:
            self.logger.info("DRY RUN: Would transfer the following files using scp:")
            self.logger.info(
                f"  Target: {self.config.user}@{self.config.host}:{target_path}"
            )
            self.logger.info(f"  SSH Key: {self.config.ssh_key}")
            for file_path in files:
                self.logger.info(f"  - {file_path}")
            return True

        # Create remote directory
        if not self.create_remote_directory(target_path):
            return False

        try:
            self.logger.info(
                f"Transferring {len(files)} files using scp to {self.config.name.upper()}"
            )

            for file_path in files:
                cmd = [
                    "scp",
                    "-i",
                    self.config.ssh_key,
                    str(file_path),
                    f"{self.config.user}@{self.config.host}:{target_path}/",
                ]

                self.logger.info(f"Transferring {file_path.name}...")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    self.logger.error(
                        f"Failed to transfer {file_path.name}: {result.stderr}"
                    )
                    return False
                else:
                    self.logger.info(f"Successfully transferred {file_path.name}")

            self.logger.info(
                f"All files transferred successfully to {self.config.name.upper()}"
            )
            return True

        except Exception as e:
            self.logger.error(f"SCP transfer failed: {str(e)}")
            return False

    def transfer_directory_rsync(
        self,
        local_dir: Path,
        remote_path: Optional[str] = None,
        compression: bool = True,
        delete_remote: bool = False,
        dry_run: bool = False,
        exclude_patterns: Optional[list[str]] = None,
    ) -> bool:
        """Transfer directory using rsync with optional compression."""
        target_path = remote_path or self.config.path

        # Build rsync command
        cmd = [
            "rsync",
            "-avh",  # archive, verbose, human-readable
            "--progress",  # show progress
        ]

        if compression:
            cmd.append("-z")  # compress during transfer

        if delete_remote:
            cmd.append("--delete")  # delete files not in source

        if dry_run:
            cmd.append("--dry-run")

        # Add exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                cmd.extend(["--exclude", pattern])

        # Add SSH options
        cmd.extend(
            [
                "-e",
                f"ssh -i {self.config.ssh_key}",
                f"{local_dir}/",  # trailing slash important for rsync
                f"{self.config.user}@{self.config.host}:{target_path}/",
            ]
        )

        if dry_run:
            self.logger.info("DRY RUN: Would transfer directory using rsync:")
            self.logger.info(f"  Command: {' '.join(cmd)}")
            return True

        # Create remote directory first
        if not self.create_remote_directory(target_path):
            return False

        try:
            self.logger.info(
                f"Transferring directory {local_dir.name} using rsync to {self.config.name.upper()}"
            )
            self.logger.info(
                f"  Compression: {'enabled' if compression else 'disabled'}"
            )
            self.logger.info(
                f"  Target: {self.config.user}@{self.config.host}:{target_path}"
            )

            result = subprocess.run(cmd, text=True)

            if result.returncode == 0:
                self.logger.info(
                    f"Directory transferred successfully to {self.config.name.upper()}"
                )
                return True
            else:
                self.logger.error(f"rsync failed with return code {result.returncode}")
                return False

        except Exception as e:
            self.logger.error(f"rsync transfer failed: {str(e)}")
            return False


class TransferManager:
    """High-level manager for system transfers."""

    def __init__(self, config_file: str = ".env"):
        self.logger = get_logger(__name__)
        self.config_loader = ConfigLoader(config_file)
        self.systems = self.config_loader.load_system_configs()

    def get_system_config(self, system_name: str) -> Optional[SystemConfig]:
        """Get configuration for a specific system."""
        system_name = system_name.lower()
        return self.systems.get(system_name)

    def list_systems(self):
        """List all available target systems."""
        if not self.systems:
            self.logger.info("No systems configured. Please edit .env")
            return

        self.logger.info("Available target systems:")
        for system_name, config in self.systems.items():
            self.logger.info(f"  {system_name.upper()}:")
            self.logger.info(f"    Host: {config.host}")
            self.logger.info(f"    User: {config.user}")
            self.logger.info(f"    Path: {config.path}")
            self.logger.info(f"    SSH Key: {config.ssh_key}")

    def create_transfer_client(self, system_name: str) -> Optional[RemoteTransfer]:
        """Create a transfer client for the specified system."""
        config = self.get_system_config(system_name)
        if not config:
            self.logger.error(f"System '{system_name}' not found in configuration")
            self.logger.info(f"Available systems: {list(self.systems.keys())}")
            return None

        return RemoteTransfer(config)
