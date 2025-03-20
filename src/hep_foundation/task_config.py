import json
from typing import Dict, List, Optional, Tuple, Union, Set, Any
from pathlib import Path

from hep_foundation.physlite_utilities import (
    PhysliteBranch,
    PhysliteFeatureFilter,
    PhysliteFeatureArrayFilter,
    PhysliteFeatureArrayAggregator,
    PhysliteSelectionConfig
)

class TaskConfig:
    """
    High-level configuration for a specific HEP analysis task.
    
    A task consists of input data selection configuration and
    optional label configurations for supervised learning tasks.
    
    Attributes:
        input: PhysliteSelectionConfig for input data
        labels: List of PhysliteSelectionConfig for target labels
        name: Name of this task
    """
    
    def __init__(self,
                input_config: PhysliteSelectionConfig,
                label_configs: List[PhysliteSelectionConfig] = None,
                name: str = "DefaultTask"):
        """
        Initialize a task configuration.
        
        Args:
            input_config: Configuration for input data selection
            label_configs: Optional list of configurations for output labels
            name: Name of this task
        """
        self.input = input_config
        self.labels = label_configs or []
        self.name = name
    
    def __str__(self) -> str:
        return (f"TaskConfig(name='{self.name}', "
                f"input={str(self.input)}, "
                f"labels={len(self.labels)})")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task configuration to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the task configuration
        """
        return {
            'name': self.name,
            'input': self._selection_config_to_dict(self.input),
            'labels': [self._selection_config_to_dict(label) for label in self.labels]
        }
    
    def _selection_config_to_dict(self, config: PhysliteSelectionConfig) -> Dict[str, Any]:
        """Helper method to convert a PhysliteSelectionConfig to a dictionary."""
        # Convert feature filters
        feature_filters = []
        for f in config.feature_filters:
            feature_filters.append({
                'branch_name': f.branch.name,
                'min_value': f.min_value,
                'max_value': f.max_value
            })
        
        # Convert feature array filters
        feature_array_filters = []
        for f in config.feature_array_filters:
            feature_array_filters.append({
                'branch_name': f.branch.name,
                'min_value': f.min_value,
                'max_value': f.max_value
            })
        
        # Convert feature array aggregators
        feature_array_aggregators = []
        for agg in config.feature_array_aggregators:
            input_branches = []
            for branch in agg.input_branches:
                input_branches.append({
                    'branch_name': branch.branch.name,
                    'min_value': branch.min_value,
                    'max_value': branch.max_value
                })
            
            feature_array_aggregators.append({
                'input_branches': input_branches,
                'sort_by_branch': {
                    'branch_name': agg.sort_by_branch.branch.name,
                    'min_value': agg.sort_by_branch.min_value,
                    'max_value': agg.sort_by_branch.max_value
                },
                'min_length': agg.min_length,
                'max_length': agg.max_length
            })
        
        return {
            'name': config.name,
            'feature_filters': feature_filters,
            'feature_array_filters': feature_array_filters,
            'feature_array_aggregators': feature_array_aggregators
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert task configuration to a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            JSON string representation of the task configuration
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save task configuration to a JSON file.
        
        Args:
            file_path: Path where to save the configuration
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaskConfig':
        """
        Create a TaskConfig from a dictionary.
        
        Args:
            config_dict: Dictionary representation of the task configuration
            
        Returns:
            New TaskConfig instance
            
        Raises:
            ValueError: If the dictionary format is invalid
        """
        # Create input config
        input_config = cls._dict_to_selection_config(config_dict.get('input', {}))
        
        # Create label configs
        label_configs = []
        for label_dict in config_dict.get('labels', []):
            label_configs.append(cls._dict_to_selection_config(label_dict))
        
        return cls(
            input_config=input_config,
            label_configs=label_configs,
            name=config_dict.get('name', 'DefaultTask')
        )
    
    @staticmethod
    def _dict_to_selection_config(config_dict: Dict[str, Any]) -> PhysliteSelectionConfig:
        """Helper method to convert a dictionary to a PhysliteSelectionConfig."""
        # Create feature filters
        feature_filters = []
        for filter_dict in config_dict.get('feature_filters', []):
            branch = PhysliteBranch(filter_dict['branch_name'])
            if branch.is_feature:
                feature_filters.append(PhysliteFeatureFilter(
                    branch=branch,
                    min_value=filter_dict.get('min_value'),
                    max_value=filter_dict.get('max_value')
                ))
        
        # Create feature array filters
        feature_array_filters = []
        for filter_dict in config_dict.get('feature_array_filters', []):
            branch = PhysliteBranch(filter_dict['branch_name'])
            if branch.is_feature_array:
                feature_array_filters.append(PhysliteFeatureArrayFilter(
                    branch=branch,
                    min_value=filter_dict.get('min_value'),
                    max_value=filter_dict.get('max_value')
                ))
        
        # Create feature array aggregators
        feature_array_aggregators = []
        for agg_dict in config_dict.get('feature_array_aggregators', []):
            # Create input branch filters
            input_branches = []
            for input_dict in agg_dict.get('input_branches', []):
                branch = PhysliteBranch(input_dict['branch_name'])
                if branch.is_feature_array:
                    input_branches.append(PhysliteFeatureArrayFilter(
                        branch=branch,
                        min_value=input_dict.get('min_value'),
                        max_value=input_dict.get('max_value')
                    ))
            
            # Create sort by branch filter
            sort_dict = agg_dict.get('sort_by_branch', {})
            if 'branch_name' in sort_dict:
                sort_branch = PhysliteBranch(sort_dict['branch_name'])
                if sort_branch.is_feature_array:
                    sort_by_branch = PhysliteFeatureArrayFilter(
                        branch=sort_branch,
                        min_value=sort_dict.get('min_value'),
                        max_value=sort_dict.get('max_value')
                    )
                    
                    # Create aggregator if we have required components
                    if input_branches and sort_by_branch:
                        feature_array_aggregators.append(PhysliteFeatureArrayAggregator(
                            input_branches=input_branches,
                            sort_by_branch=sort_by_branch,
                            min_length=agg_dict.get('min_length', 1),
                            max_length=agg_dict.get('max_length', 100)
                        ))
        
        # Create the selection config
        return PhysliteSelectionConfig(
            name=config_dict.get('name', 'Selection'),
            feature_filters=feature_filters,
            feature_array_filters=feature_array_filters,
            feature_array_aggregators=feature_array_aggregators
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TaskConfig':
        """
        Create a TaskConfig from a JSON string.
        
        Args:
            json_str: JSON string representation of the task configuration
            
        Returns:
            New TaskConfig instance
            
        Raises:
            ValueError: If the JSON format is invalid
        """
        try:
            config_dict = json.loads(json_str)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'TaskConfig':
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
        
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())
    
    @classmethod
    def create_from_branch_names(cls,
                                input_features: Dict[str, Dict[str, Optional[float]]],
                                input_array_features: Dict[str, Dict[str, Optional[float]]] = None,
                                input_array_aggregators: List[Dict[str, Any]] = None,
                                label_features: List[Dict[str, Dict[str, Optional[float]]]] = None,
                                label_array_features: List[Dict[str, Dict[str, Optional[float]]]] = None,
                                label_array_aggregators: List[List[Dict[str, Any]]] = None,
                                name: str = "CustomTask") -> 'TaskConfig':
        """
        Create a TaskConfig from branch names and filter ranges.
        
        This is a convenience method for creating a TaskConfig without having to 
        manually create all the intermediate objects.
        
        Args:
            input_features: Dictionary mapping branch names to filter ranges for scalar features
                Example: {'EventInfoAuxDyn.eventNumber': {'min': 0}}
            input_array_features: Dictionary mapping branch names to filter ranges for array features
                Example: {'InDetTrackParticlesAuxDyn.d0': {'min': -5.0, 'max': 5.0}}
            input_array_aggregators: List of dictionaries defining aggregators
                Example: [{'input_branches': ['InDetTrackParticlesAuxDyn.d0', 'InDetTrackParticlesAuxDyn.z0'],
                           'sort_by_branch': 'InDetTrackParticlesAuxDyn.pt',
                           'min_length': 5, 'max_length': 100}]
            label_features: List of dictionaries for label scalar features (one per label)
            label_array_features: List of dictionaries for label array features (one per label)
            label_array_aggregators: List of lists of aggregator dictionaries (one list per label)
            name: Name for this task
            
        Returns:
            New TaskConfig instance
            
        Raises:
            ValueError: If the input parameters are invalid
        """
        # Create input configuration
        input_config = cls._create_selection_config_from_dicts(
            feature_dict=input_features or {},
            array_feature_dict=input_array_features or {},
            aggregator_list=input_array_aggregators or [],
            name="Input"
        )
        
        # Create label configurations
        label_configs = []
        
        # Determine the number of label configurations to create
        n_labels = max(
            len(label_features or []),
            len(label_array_features or []),
            len(label_array_aggregators or [])
        )
        
        for i in range(n_labels):
            # Get the dictionaries for this label (or empty)
            feature_dict = (label_features or [])[i] if i < len(label_features or []) else {}
            array_feature_dict = (label_array_features or [])[i] if i < len(label_array_features or []) else {}
            aggregator_list = (label_array_aggregators or [])[i] if i < len(label_array_aggregators or []) else []
            
            # Create label configuration
            label_config = cls._create_selection_config_from_dicts(
                feature_dict=feature_dict,
                array_feature_dict=array_feature_dict,
                aggregator_list=aggregator_list,
                name=f"Label_{i+1}"
            )
            
            label_configs.append(label_config)
        
        return cls(
            input_config=input_config,
            label_configs=label_configs,
            name=name
        )
    
    @staticmethod
    def _create_selection_config_from_dicts(
            feature_dict: Dict[str, Dict[str, Optional[float]]],
            array_feature_dict: Dict[str, Dict[str, Optional[float]]],
            aggregator_list: List[Dict[str, Any]],
            name: str) -> PhysliteSelectionConfig:
        """Helper method to create a PhysliteSelectionConfig from dictionaries."""
        # Create feature filters
        feature_filters = []
        for branch_name, range_dict in feature_dict.items():
            try:
                branch = PhysliteBranch(branch_name)
                if branch.is_feature:
                    feature_filters.append(PhysliteFeatureFilter(
                        branch=branch,
                        min_value=range_dict.get('min'),
                        max_value=range_dict.get('max')
                    ))
            except ValueError:
                # Skip invalid branches
                continue
        
        # Create feature array filters
        feature_array_filters = []
        for branch_name, range_dict in array_feature_dict.items():
            try:
                branch = PhysliteBranch(branch_name)
                if branch.is_feature_array:
                    feature_array_filters.append(PhysliteFeatureArrayFilter(
                        branch=branch,
                        min_value=range_dict.get('min'),
                        max_value=range_dict.get('max')
                    ))
            except ValueError:
                # Skip invalid branches
                continue
                
        # Create feature array aggregators
        feature_array_aggregators = []
        for agg_dict in aggregator_list:
            try:
                # Create input branch filters
                input_branches = []
                for branch_name in agg_dict.get('input_branches', []):
                    # Check if we have range info for this branch
                    range_dict = array_feature_dict.get(branch_name, {})
                    
                    branch = PhysliteBranch(branch_name)
                    if branch.is_feature_array:
                        input_branches.append(PhysliteFeatureArrayFilter(
                            branch=branch,
                            min_value=range_dict.get('min'),
                            max_value=range_dict.get('max')
                        ))
                
                # Create sort by branch filter
                sort_branch_name = agg_dict.get('sort_by_branch')
                if sort_branch_name:
                    # Check if we have range info for this branch
                    range_dict = array_feature_dict.get(sort_branch_name, {})
                    
                    sort_branch = PhysliteBranch(sort_branch_name)
                    if sort_branch.is_feature_array:
                        sort_by_branch = PhysliteFeatureArrayFilter(
                            branch=sort_branch,
                            min_value=range_dict.get('min'),
                            max_value=range_dict.get('max')
                        )
                        
                        # Create aggregator if we have required components
                        if input_branches:
                            feature_array_aggregators.append(PhysliteFeatureArrayAggregator(
                                input_branches=input_branches,
                                sort_by_branch=sort_by_branch,
                                min_length=agg_dict.get('min_length', 1),
                                max_length=agg_dict.get('max_length', 100)
                            ))
            except (ValueError, KeyError):
                # Skip invalid aggregators
                continue
        
        # Create the selection config (may raise ValueError if all lists are empty)
        try:
            return PhysliteSelectionConfig(
                name=name,
                feature_filters=feature_filters,
                feature_array_filters=feature_array_filters,
                feature_array_aggregators=feature_array_aggregators
            )
        except ValueError:
            # If all lists are empty, create with minimal valid configuration
            if not (feature_filters or feature_array_filters or feature_array_aggregators):
                # Try to create a minimal valid feature filter
                for branch_name in feature_dict.keys():
                    try:
                        branch = PhysliteBranch(branch_name)
                        feature_filters = [PhysliteFeatureFilter(branch=branch)]
                        break
                    except ValueError:
                        continue
                
                # If that fails, try with array filters
                if not feature_filters:
                    for branch_name in array_feature_dict.keys():
                        try:
                            branch = PhysliteBranch(branch_name)
                            feature_array_filters = [PhysliteFeatureArrayFilter(branch=branch)]
                            break
                        except ValueError:
                            continue
            
            # Create with whatever we have
            return PhysliteSelectionConfig(
                name=name,
                feature_filters=feature_filters,
                feature_array_filters=feature_array_filters,
                feature_array_aggregators=feature_array_aggregators
            )
