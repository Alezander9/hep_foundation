from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import numpy as np
import logging
from pathlib import Path
import json

# Attempt to import the branch index, with graceful fallback
try:
    from hep_foundation.physlite_branch_index import PHYSLITE_BRANCHES
    HAS_BRANCH_INDEX = True
except ImportError:
    logging.warning("PhysLite branch index not found. Branch validation will be limited.")
    PHYSLITE_BRANCHES = {}
    PHYSLITE_COMMON_BRANCHES = {}
    HAS_BRANCH_INDEX = False

class BranchType(Enum):
    """Enum for branch types based on their shape."""
    UNKNOWN = "unknown"
    FEATURE = "feature"        # Scalar value (single value per event)
    FEATURE_ARRAY = "feature_array"  # Array value (multiple values per event)

def get_branch_info(branch_name: str) -> Tuple[bool, BranchType, Optional[Dict[str, Any]]]:
    """
    Check if a branch name is valid and determine its type.
    
    Args:
        branch_name: Full branch name (e.g., "InDetTrackParticlesAuxDyn.d0")
        
    Returns:
        Tuple containing:
        - Boolean indicating if branch exists
        - BranchType enum value
        - Dictionary with branch information if available, None otherwise
        
    Raises:
        RuntimeError: If branch index is not available
    """
    # If branch index is not available, raise error
    if not HAS_BRANCH_INDEX:
        raise RuntimeError(
            "PhysLite branch index not found. Cannot validate branches or determine their types. "
            "Please ensure the branch index is properly installed and accessible."
        )
    
    # Handle branch names without dots (in "Other" category)
    if '.' not in branch_name:
        if 'Other' in PHYSLITE_BRANCHES and branch_name in PHYSLITE_BRANCHES['Other']:
            branch_info = PHYSLITE_BRANCHES['Other'][branch_name]
            branch_type = _determine_branch_type(branch_info)
            return True, branch_type, branch_info
        return False, BranchType.UNKNOWN, None
    
    # Split branch name by the first dot
    category, feature = branch_name.split('.', 1)
    
    # Check if category and feature exist
    if category not in PHYSLITE_BRANCHES or feature not in PHYSLITE_BRANCHES[category]:
        return False, BranchType.UNKNOWN, None
    
    # Get branch info
    branch_info = PHYSLITE_BRANCHES[category][feature]
    
    # Check if the branch has been verified or has shape information
    is_valid = (
        branch_info.get('verified', False) or 
        branch_info.get('shape') is not None or
        branch_info.get('status') == 'success'
    )
    
    if not is_valid:
        return False, BranchType.UNKNOWN, None
    
    # Determine branch type from shape information
    branch_type = _determine_branch_type(branch_info)
    
    return True, branch_type, branch_info

def _determine_branch_type(branch_info: Dict[str, Any]) -> BranchType:
    """Helper function to determine branch type from its info."""
    if 'shape' not in branch_info or branch_info['shape'] is None:
        return BranchType.UNKNOWN
    
    # Convert string representation of shape to tuple if needed
    shape = branch_info['shape']
    if isinstance(shape, str):
        try:
            # Handle tuple string like "()" or "(10,)"
            shape = eval(shape)
        except:
            return BranchType.UNKNOWN
    
    # Empty tuple or tuple with zeros indicates a scalar (feature)
    if not shape or shape == () or shape == (0,):
        return BranchType.FEATURE
    
    # Non-empty shape indicates an array (feature_array)
    return BranchType.FEATURE_ARRAY

class PhysliteBranch:
    """
    Represents a validated branch in the PhysLite data.
    
    Attributes:
        name: The full branch name
        category: The category part of the branch name
        feature: The feature part of the branch name
        branch_type: The type of branch (feature or feature_array)
        info: Additional information about the branch
    """
    
    def __init__(self, branch_name: str):
        """
        Initialize a PhysliteBranch object.
        
        Args:
            branch_name: Full branch name (e.g., "InDetTrackParticlesAuxDyn.d0")
            
        Raises:
            ValueError: If the branch name is invalid
        """
        is_valid, branch_type, branch_info = get_branch_info(branch_name)
        
        if not is_valid:
            raise ValueError(f"Invalid branch name: {branch_name}")
        
        self.name = branch_name
        self.branch_type = branch_type
        self.info = branch_info
        
        # Split the branch name into category and feature
        if '.' in branch_name:
            self.category, self.feature = branch_name.split('.', 1)
        else:
            self.category = 'Other'
            self.feature = branch_name
    
    @property
    def is_feature(self) -> bool:
        """Check if this branch is a scalar feature (single value per event)."""
        return self.branch_type == BranchType.FEATURE
    
    @property
    def is_feature_array(self) -> bool:
        """Check if this branch is a feature array (multiple values per event)."""
        return self.branch_type == BranchType.FEATURE_ARRAY
    
    def __str__(self) -> str:
        return f"PhysliteBranch({self.name}, type={self.branch_type.value})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_shape(self) -> Optional[Tuple[int, ...]]:
        """Get the shape of this branch if available."""
        if not self.info or 'shape' not in self.info:
            return None
            
        shape = self.info['shape']
        if isinstance(shape, str):
            try:
                # Handle tuple string like "()" or "(10,)"
                shape = eval(shape)
            except:
                return None
        
        return shape
    
    def get_dtype(self) -> Optional[str]:
        """Get the data type of this branch if available."""
        if not self.info or 'dtype' not in self.info:
            return None
        return self.info['dtype']

class PhysliteFeatureSelector:
    """
    Selector for scalar PhysLite features (single value per event).
    
    Attributes:
        branch: The PhysliteBranch to select
    """
    
    def __init__(self, branch: PhysliteBranch):
        """
        Initialize a feature selector.
        
        Args:
            branch: PhysliteBranch to select
            
        Raises:
            ValueError: If branch is not a scalar feature type
        """
        if not branch.is_feature:
            raise ValueError(f"Branch {branch.name} is not a scalar feature. Use PhysliteFeatureArraySelector instead.")
        
        self.branch = branch
    
    def __str__(self) -> str:
        return f"FeatureSelector({self.branch.name})"
    
    def __repr__(self) -> str:
        return self.__str__()

class PhysliteFeatureArraySelector:
    """
    Selector for PhysLite feature arrays (multiple values per event).
    
    Attributes:
        branch: The PhysliteBranch to select
    """
    
    def __init__(self, branch: PhysliteBranch):
        """
        Initialize a feature array selector.
        
        Args:
            branch: PhysliteBranch to select
            
        Raises:
            ValueError: If branch is not a feature array type
        """
        if not branch.is_feature_array:
            raise ValueError(f"Branch {branch.name} is not a feature array. Use PhysliteFeatureSelector instead.")
        
        self.branch = branch
    
    def __str__(self) -> str:
        return f"FeatureArraySelector({self.branch.name})"
    
    def __repr__(self) -> str:
        return self.__str__()

class PhysliteFeatureFilter:
    """
    Filter for scalar PhysLite features (single value per event).
    
    Attributes:
        branch: The PhysliteBranch to filter on
        min_value: Minimum allowed value (None means no minimum)
        max_value: Maximum allowed value (None means no maximum)
    """
    
    def __init__(self, 
                branch: PhysliteBranch, 
                min_value: Optional[float] = None, 
                max_value: Optional[float] = None):
        """
        Initialize a feature filter.
        
        Args:
            branch: PhysliteBranch to filter on
            min_value: Minimum allowed value (None means no minimum)
            max_value: Maximum allowed value (None means no maximum)
            
        Raises:
            ValueError: If branch is not a scalar feature type or if both min_value and max_value are None
        """
        if not branch.is_feature:
            raise ValueError(f"Branch {branch.name} is not a scalar feature. Use PhysliteFeatureArrayFilter instead.")
        
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")
        
        self.branch = branch
        self.min_value = min_value
        self.max_value = max_value
    
    def __str__(self) -> str:
        min_str = str(self.min_value) if self.min_value is not None else "-∞"
        max_str = str(self.max_value) if self.max_value is not None else "∞"
        return f"FeatureFilter({self.branch.name}, range=[{min_str}, {max_str}])"
    
    def __repr__(self) -> str:
        return self.__str__()

class PhysliteFeatureArrayFilter:
    """
    Filter for PhysLite feature arrays (multiple values per event).
    
    Attributes:
        branch: The PhysliteBranch to filter on
        min_value: Minimum allowed value (None means no minimum)
        max_value: Maximum allowed value (None means no maximum)
    """
    
    def __init__(self, 
                branch: PhysliteBranch, 
                min_value: Optional[float] = None, 
                max_value: Optional[float] = None):
        """
        Initialize a feature array filter.
        
        Args:
            branch: PhysliteBranch to filter on
            min_value: Minimum allowed value (None means no minimum)
            max_value: Maximum allowed value (None means no maximum)
            
        Raises:
            ValueError: If branch is not a feature array type or if both min_value and max_value are None
        """
        if not branch.is_feature_array:
            raise ValueError(f"Branch {branch.name} is not a feature array. Use PhysliteFeatureFilter instead.")
            
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")
        
        self.branch = branch
        self.min_value = min_value
        self.max_value = max_value
    
    def __str__(self) -> str:
        min_str = str(self.min_value) if self.min_value is not None else "-∞"
        max_str = str(self.max_value) if self.max_value is not None else "∞"
        return f"FeatureArrayFilter({self.branch.name}, range=[{min_str}, {max_str}])"
    
    def __repr__(self) -> str:
        return self.__str__()

class PhysliteFeatureArrayAggregator:
    """
    Configuration for aggregating multiple feature arrays from PhysLite data.
    
    This class bundles configuration parameters for extracting, filtering, 
    and aggregating feature arrays from PhysLite events.
    
    Attributes:
        input_branches: List of feature array selectors for input data collection
        filter_branches: List of feature array filters for filtering data
        sort_by_branch: Feature array selector to use for sorting (typically pT), or None to keep original order
        min_length: Minimum number of array elements required after filtering
        max_length: Maximum number of array elements to keep (truncation/padding size)
    """
    
    def __init__(self, 
                input_branches: List[PhysliteFeatureArraySelector],
                filter_branches: List[PhysliteFeatureArrayFilter],
                sort_by_branch: Optional[PhysliteFeatureArraySelector] = None,
                min_length: int = 1,
                max_length: int = 100):
        """
        Initialize a feature array aggregator configuration.
        
        Args:
            input_branches: List of feature array selectors to extract
            filter_branches: List of feature array filters to apply filtering
            sort_by_branch: Feature array selector to use for sorting values, or None to keep original order
            min_length: Minimum number of elements required after filtering
            max_length: Maximum number of elements to keep (will pad or truncate)
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if not input_branches:
            raise ValueError("At least one input branch must be provided")
            
        if min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {min_length}")
            
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
            
        if min_length > max_length:
            raise ValueError(f"min_length ({min_length}) cannot be greater than max_length ({max_length})")
        
        self.input_branches = input_branches
        self.filter_branches = filter_branches
        self.sort_by_branch = sort_by_branch
        self.min_length = min_length
        self.max_length = max_length
        
        # Get branch names for easier reference
        self.input_branch_names = [f.branch.name for f in input_branches]
        self.filter_branch_names = [f.branch.name for f in filter_branches]
        self.sort_by_branch_name = sort_by_branch.branch.name if sort_by_branch is not None else None
    
    def __str__(self) -> str:
        input_str = ", ".join(self.input_branch_names)
        filter_str = ", ".join(self.filter_branch_names) if self.filter_branches else "none"
        sort_str = self.sort_by_branch_name if self.sort_by_branch_name is not None else "none"
        return (f"FeatureArrayAggregator(branches=[{input_str}], "
                f"filters=[{filter_str}], "
                f"sort_by={sort_str}, "
                f"length={self.min_length}-{self.max_length})")
    
    def __repr__(self) -> str:
        return self.__str__()

class PhysliteSelectionConfig:
    """
    High-level configuration for PhysLite data selection and feature extraction.
    
    This class bundles together all parameters needed to define what features
    to extract from PhysLite data processing. Filtering is handled separately.
    
    Attributes:
        feature_selectors: List of scalar feature selectors for individual values
        feature_array_aggregators: List of feature array aggregators for collecting arrays
        name: Optional name for this configuration
    """
    
    def __init__(self,
                feature_selectors: List[PhysliteFeatureSelector] = None,
                feature_array_aggregators: List[PhysliteFeatureArrayAggregator] = None,
                name: str = "PhysliteSelection"):
        """
        Initialize a PhysLite selection configuration.
        
        Args:
            feature_selectors: List of scalar feature selectors for individual values
            feature_array_aggregators: List of feature array aggregators for collecting arrays
            name: Optional name for this configuration
            
        Raises:
            ValueError: If no selectors or aggregators are provided
        """
        self.feature_selectors = feature_selectors or []
        self.feature_array_aggregators = feature_array_aggregators or []
        self.name = name
        
        # Ensure at least one selector or aggregator is provided
        if not (self.feature_selectors or self.feature_array_aggregators):
            raise ValueError("At least one feature selector or aggregator must be provided")
    
    def __str__(self) -> str:
        return (f"PhysliteSelectionConfig(name='{self.name}', "
                f"feature_selectors={len(self.feature_selectors)}, "
                f"feature_array_aggregators={len(self.feature_array_aggregators)})")
    
    def __repr__(self) -> str:
        return self.__str__()

    def get_total_feature_size(self) -> int:
        """
        Calculate total feature size by combining scalar features and aggregated array features.
        
        Returns:
            int: Total number of features, calculated as:
                 (number of scalar features) + 
                 sum(aggregator.max_length * len(aggregator.input_branches) for each aggregator)
        """
        # Count scalar features
        total_size = len(self.feature_selectors)
        
        # Add size from each aggregator
        for aggregator in self.feature_array_aggregators:
            # For each aggregator, multiply its max_length by number of input features
            aggregator_size = aggregator.max_length * len(aggregator.input_branches)
            total_size += aggregator_size
        
        return total_size
