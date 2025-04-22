import json
from pathlib import Path
from typing import Any, Optional, Union

from hep_foundation.logging_config import setup_logging
from hep_foundation.physlite_utilities import (
    PhysliteBranch,
    PhysliteFeatureArrayAggregator,
    PhysliteFeatureArrayFilter,
    PhysliteFeatureArraySelector,
    PhysliteFeatureFilter,
    PhysliteFeatureSelector,
    PhysliteSelectionConfig,
)


class TaskConfig:
    """
    High-level configuration for a specific HEP analysis task.

    A task consists of:
    - Event-level filters to select which events to process
    - Input data selection configuration
    - Optional label configurations for supervised learning tasks

    Attributes:
        event_filters: List of event-level feature filters
        input: PhysliteSelectionConfig for input data
        labels: List of PhysliteSelectionConfig for target labels
    """

    def __init__(
        self,
        event_filters: list[PhysliteFeatureFilter],
        input_config: PhysliteSelectionConfig,
        label_configs: list[PhysliteSelectionConfig] = None,
    ):
        """
        Initialize a task configuration.

        Args:
            event_filters: List of event-level feature filters
            input_config: Configuration for input data selection
            label_configs: Optional list of configurations for output labels
        """
        # Setup logging
        setup_logging()

        self.event_filters = event_filters
        self.input = input_config
        self.labels = label_configs or []

    def __str__(self) -> str:
        return (
            f"TaskConfig(event_filters={len(self.event_filters)}, "
            f"input={str(self.input)}, "
            f"labels={len(self.labels)})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def to_json(self, indent: int = 2) -> str:
        """
        Convert task configuration to a JSON string.

        Args:
            indent: Number of spaces for indentation

        Returns:
            JSON string representation of the task configuration
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert task configuration to a dictionary.

        Returns:
            Dictionary representation of the task configuration
        """
        return {
            "event_filters": [
                {
                    "branch_name": f.branch.name,
                    "min_value": f.min_value,
                    "max_value": f.max_value,
                }
                for f in self.event_filters
            ],
            "input": self._selection_config_to_dict(self.input),
            "labels": [self._selection_config_to_dict(label) for label in self.labels],
        }

    def _selection_config_to_dict(
        self, config: PhysliteSelectionConfig
    ) -> dict[str, Any]:
        """Helper method to convert a PhysliteSelectionConfig to a dictionary."""
        # Convert feature selectors
        feature_selectors = [
            {"branch_name": s.branch.name} for s in config.feature_selectors
        ]

        # Convert feature array aggregators
        feature_array_aggregators = []
        for agg in config.feature_array_aggregators:
            # Convert input branches
            input_branches = [
                {"branch_name": b.branch.name} for b in agg.input_branches
            ]

            # Convert filter branches
            filter_branches = [
                {
                    "branch_name": f.branch.name,
                    "min_value": f.min_value,
                    "max_value": f.max_value,
                }
                for f in agg.filter_branches
            ]

            # Convert sort_by_branch
            sort_by_branch = (
                {"branch_name": agg.sort_by_branch.branch.name}
                if agg.sort_by_branch
                else None
            )

            feature_array_aggregators.append(
                {
                    "input_branches": input_branches,
                    "filter_branches": filter_branches,
                    "sort_by_branch": sort_by_branch,
                    "min_length": agg.min_length,
                    "max_length": agg.max_length,
                }
            )

        return {
            "name": config.name,
            "feature_selectors": feature_selectors,
            "feature_array_aggregators": feature_array_aggregators,
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save task configuration to a JSON file.

        Args:
            file_path: Path where to save the configuration
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TaskConfig":
        """
        Load task configuration from a JSON file.

        Args:
            file_path: Path to the configuration file

        Returns:
            New TaskConfig instance

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path) as f:
            config_dict = json.loads(f.read())

        # Create event filters
        event_filters = []
        for filter_dict in config_dict.get("event_filters", []):
            branch = PhysliteBranch(filter_dict["branch_name"])
            if branch.is_feature:
                event_filters.append(
                    PhysliteFeatureFilter(
                        branch=branch,
                        min_value=filter_dict.get("min_value"),
                        max_value=filter_dict.get("max_value"),
                    )
                )

        # Create input config
        input_config = cls._dict_to_selection_config(config_dict.get("input", {}))

        # Create label configs
        label_configs = []
        for label_dict in config_dict.get("labels", []):
            label_configs.append(cls._dict_to_selection_config(label_dict))

        return cls(
            event_filters=event_filters,
            input_config=input_config,
            label_configs=label_configs,
        )

    @staticmethod
    def _dict_to_selection_config(
        config_dict: dict[str, Any],
    ) -> PhysliteSelectionConfig:
        """Helper method to convert a dictionary to a PhysliteSelectionConfig."""

        # Create feature selectors
        feature_selectors = []
        for selector_dict in config_dict.get("feature_selectors", []):
            branch = PhysliteBranch(selector_dict["branch_name"])
            if branch.is_feature:
                feature_selectors.append(PhysliteFeatureSelector(branch=branch))

        # Create feature array aggregators
        feature_array_aggregators = []
        for agg_dict in config_dict.get("feature_array_aggregators", []):
            # Create input branch selectors
            input_branches = []
            for input_dict in agg_dict.get("input_branches", []):
                branch = PhysliteBranch(input_dict["branch_name"])
                if branch.is_feature_array:
                    input_branches.append(PhysliteFeatureArraySelector(branch=branch))

            # Create filter branch filters
            filter_branches = []
            for filter_dict in agg_dict.get("filter_branches", []):
                branch = PhysliteBranch(filter_dict["branch_name"])
                if branch.is_feature_array:
                    filter_branches.append(
                        PhysliteFeatureArrayFilter(
                            branch=branch,
                            min_value=filter_dict.get("min"),
                            max_value=filter_dict.get("max"),
                        )
                    )

            # Create sort by branch filter
            sort_dict = agg_dict.get("sort_by_branch", {})

            sort_by_branch = None
            if sort_dict and "branch" in sort_dict:
                branch_name = sort_dict.get("branch")
                branch = PhysliteBranch(branch_name)
                if branch.is_feature_array:
                    sort_by_branch = PhysliteFeatureArraySelector(branch=branch)

            # Create aggregator if we have input branches
            if input_branches:
                try:
                    aggregator = PhysliteFeatureArrayAggregator(
                        input_branches=input_branches,
                        filter_branches=filter_branches,
                        sort_by_branch=sort_by_branch,
                        min_length=agg_dict.get("min_length", 1),
                        max_length=agg_dict.get("max_length", 100),
                    )
                    feature_array_aggregators.append(aggregator)
                except Exception:
                    continue

        # Create the selection config
        try:
            return PhysliteSelectionConfig(
                name=config_dict.get("name", "Selection"),
                feature_selectors=feature_selectors,
                feature_array_aggregators=feature_array_aggregators,
            )
        except ValueError:
            # If there are no selectors or aggregators, return a minimal configuration
            if not feature_selectors and not feature_array_aggregators:
                # Try to create one feature selector as a fallback
                for branch_name in config_dict.get("feature_selectors", []):
                    try:
                        branch = PhysliteBranch(branch_name)
                        feature_selectors = [PhysliteFeatureSelector(branch=branch)]
                        break
                    except ValueError:
                        continue

            # Create with whatever we have
            return PhysliteSelectionConfig(
                name=config_dict.get("name", "Selection"),
                feature_selectors=feature_selectors,
                feature_array_aggregators=feature_array_aggregators,
            )

    @classmethod
    def create_from_branch_names(
        cls,
        event_filter_dict: dict[str, dict[str, Optional[float]]],
        input_features: list[str],
        input_array_aggregators: list[dict[str, Any]] = None,
        label_features: list[list[str]] = None,
        label_array_aggregators: list[list[dict[str, Any]]] = None,
    ) -> "TaskConfig":
        """
        Create a TaskConfig from branch names directly.

        This is a convenience method for creating a TaskConfig without having to
        manually create all the intermediate objects.

        Args:
            event_filter_dict: Dictionary mapping branch names to filter ranges for event selection
                Example: {'EventInfoAuxDyn.eventNumber': {'min': 0}}
            input_features: List of branch names for scalar input features
                Example: ['EventInfoAuxDyn.eventNumber', 'EventInfoAuxDyn.mcEventWeight']
            input_array_aggregators: List of dictionaries defining input aggregators
                Example: [{'input_branches': ['InDetTrackParticlesAuxDyn.d0', 'InDetTrackParticlesAuxDyn.z0'],
                           'filter_branches': [{'branch': 'InDetTrackParticlesAuxDyn.pt', 'min': 1000.0}],
                           'sort_by_branch': {'branch': 'InDetTrackParticlesAuxDyn.pt', 'min': 1000.0},
                           'min_length': 5, 'max_length': 100}]
            label_features: List of lists of branch names for scalar label features (one list per label)
            label_array_aggregators: List of lists of aggregator dictionaries for labels (one list per label)

        Returns:
            New TaskConfig instance
        """
        # Create event filters
        event_filters = []
        for branch_name, range_dict in event_filter_dict.items():
            try:
                branch = PhysliteBranch(branch_name)
                if branch.is_feature:
                    # Need to ensure at least one of min/max is specified
                    min_val = range_dict.get("min")
                    max_val = range_dict.get("max")

                    # If both are None, default to min=0 to satisfy requirements
                    if min_val is None and max_val is None:
                        min_val = 0

                    event_filters.append(
                        PhysliteFeatureFilter(
                            branch=branch, min_value=min_val, max_value=max_val
                        )
                    )
            except ValueError:
                # Skip invalid branches
                continue

        # Create input configuration
        input_config = cls._create_selection_config_from_lists(
            feature_names=input_features or [],
            aggregator_list=input_array_aggregators or [],
            name="Input",
        )

        # Create label configurations
        label_configs = []

        # Determine the number of label configurations to create
        n_labels = max(len(label_features or []), len(label_array_aggregators or []))

        for i in range(n_labels):
            # Get the input for this label (or empty list)
            feature_list = (
                (label_features or [])[i] if i < len(label_features or []) else []
            )
            aggregator_list = (
                (label_array_aggregators or [])[i]
                if i < len(label_array_aggregators or [])
                else []
            )

            # Create label configuration
            label_config = cls._create_selection_config_from_lists(
                feature_names=feature_list,
                aggregator_list=aggregator_list,
                name=f"Label_{i + 1}",
            )

            label_configs.append(label_config)

        return cls(
            event_filters=event_filters,
            input_config=input_config,
            label_configs=label_configs,
        )

    @staticmethod
    def _create_selection_config_from_lists(
        feature_names: list[str], aggregator_list: list[dict[str, Any]], name: str
    ) -> PhysliteSelectionConfig:
        """Helper method to create a PhysliteSelectionConfig from lists of names and aggregator dicts."""
        # Create feature selectors
        feature_selectors = []
        for branch_name in feature_names:
            try:
                branch = PhysliteBranch(branch_name)
                if branch.is_feature:
                    selector = PhysliteFeatureSelector(branch=branch)
                    feature_selectors.append(selector)
            except Exception:
                continue

        # Create feature array aggregators
        feature_array_aggregators = []
        for agg_dict in aggregator_list:
            try:
                # Create input branch selectors
                input_branches = []
                for branch_name in agg_dict.get("input_branches", []):
                    try:
                        branch = PhysliteBranch(branch_name)
                        if branch.is_feature_array:
                            selector = PhysliteFeatureArraySelector(branch=branch)
                            input_branches.append(selector)
                    except Exception:
                        continue

                # Create filter branch filters
                filter_branches = []
                for filter_dict in agg_dict.get("filter_branches", []):
                    try:
                        branch_name = filter_dict.get("branch")
                        branch = PhysliteBranch(branch_name)
                        if branch.is_feature_array:
                            filter_branches.append(
                                PhysliteFeatureArrayFilter(
                                    branch=branch,
                                    min_value=filter_dict.get("min"),
                                    max_value=filter_dict.get("max"),
                                )
                            )
                    except Exception:
                        continue

                # Create sort by branch selector
                sort_branch_dict = agg_dict.get("sort_by_branch")
                sort_by_branch = None
                if sort_branch_dict:
                    try:
                        branch_name = sort_branch_dict.get("branch")
                        branch = PhysliteBranch(branch_name)
                        if branch.is_feature_array:
                            sort_by_branch = PhysliteFeatureArraySelector(branch=branch)
                    except Exception:
                        pass

                # Create aggregator if we have input branches
                if input_branches:
                    try:
                        aggregator = PhysliteFeatureArrayAggregator(
                            input_branches=input_branches,
                            filter_branches=filter_branches,
                            sort_by_branch=sort_by_branch,
                            min_length=agg_dict.get("min_length", 1),
                            max_length=agg_dict.get("max_length", 100),
                        )
                        feature_array_aggregators.append(aggregator)
                    except Exception:
                        continue

            except Exception:
                continue

        # Create the selection config
        try:
            return PhysliteSelectionConfig(
                name=name,
                feature_selectors=feature_selectors,
                feature_array_aggregators=feature_array_aggregators,
            )
        except ValueError:
            # If there are no selectors or aggregators, return a minimal configuration
            if not feature_selectors and not feature_array_aggregators:
                # Try to create one feature selector as a fallback
                for branch_name in feature_names:
                    try:
                        branch = PhysliteBranch(branch_name)
                        feature_selectors = [PhysliteFeatureSelector(branch=branch)]
                        break
                    except ValueError:
                        continue

            # Create with whatever we have
            return PhysliteSelectionConfig(
                name=name,
                feature_selectors=feature_selectors,
                feature_array_aggregators=feature_array_aggregators,
            )
