from typing import Any, Callable

import numpy as np


class DerivedFeature:
    """Represents a feature calculated from one or more real PhysLite branches."""

    def __init__(
        self,
        name: str,
        function: Callable[..., np.ndarray],
        dependencies: list[str],
        shape: list[int],
        dtype: str,
    ):
        """
        Args:
            name: The name for the derived feature (e.g., "derived.InDetTrackParticlesAuxDyn.eta").
                  Conventionally, these should start with 'derived.'.
            function: The function to compute the feature. It must accept numpy arrays
                      corresponding to the dependencies, in the specified order.
            dependencies: A list of real PhysLite branch names required by the function.
            shape: Expected shape of the output array (e.g., [-1] for variable 1D array).
            dtype: Expected data type of the output array (e.g., "float32").
        """
        self.name = name
        self.function = function
        self.dependencies = dependencies
        self.shape = shape
        self.dtype = dtype
        # Potential future additions: units, description

    def get_branch_info_dict(self) -> dict[str, Any]:
        """Constructs a dictionary similar to the ones in the PhysLite index."""
        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "status": "derived",  # Mark as derived
        }

    def calculate(self, dependency_data: dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the derived feature value using provided dependency data.

        Args:
            dependency_data: A dictionary where keys are dependency branch names
                             and values are the corresponding numpy arrays for the current event/batch.

        Returns:
            A numpy array containing the calculated derived feature values.

        Raises:
            KeyError: If a required dependency is missing from dependency_data.
        """
        try:
            # Prepare arguments in the correct order
            args = [dependency_data[dep] for dep in self.dependencies]
            result = self.function(*args)
            # Ensure output dtype matches expected dtype if possible
            expected_np_dtype = np.dtype(self.dtype)
            if result.dtype != expected_np_dtype:
                try:
                    result = result.astype(expected_np_dtype)
                except TypeError:
                    # Warn if conversion fails, but proceed
                    print(
                        f"Warning: Could not cast derived feature '{self.name}' from {result.dtype} to {self.dtype}"
                    )

            return result

        except KeyError as e:
            raise KeyError(
                f"Missing dependency '{e}' required for derived feature '{self.name}'"
            ) from e
        except Exception as e:
            # Catch potential errors within the calculation function itself
            raise RuntimeError(
                f"Error calculating derived feature '{self.name}': {e}"
            ) from e


# --- Calculation Functions ---
# (These should operate on numpy arrays, as uproot provides them)


def theta_to_eta(theta: np.ndarray) -> np.ndarray:
    """Convert polar angle theta (radians) to pseudorapidity eta."""
    # Add larger epsilon to avoid issues at theta=0 or pi and prevent -inf
    epsilon = 1e-6
    theta = np.clip(theta, epsilon, np.pi - epsilon)

    # Additional safeguarding to prevent extreme values
    tan_half_theta = np.tan(theta / 2.0)
    # Ensure tan_half_theta is not too close to 0 to prevent -inf
    tan_half_theta = np.clip(tan_half_theta, epsilon, 1.0 / epsilon)

    eta = -np.log(tan_half_theta)

    # Final clipping to prevent extreme eta values (typical physics range is roughly -5 to 5)
    eta = np.clip(eta, -10.0, 10.0)

    return eta


def calculate_pt(qOverP: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Calculate pT (in GeV) from qOverP (in MeV⁻¹) and theta (radians)."""
    qOverP_GeV = qOverP / 1000.0  # MeV^-1 to GeV^-1
    # Handle potential division by zero safely
    p_GeV = np.full_like(qOverP_GeV, fill_value=np.inf)
    non_zero_mask = qOverP_GeV != 0
    if np.any(non_zero_mask):
        p_GeV[non_zero_mask] = np.abs(1.0 / qOverP_GeV[non_zero_mask])
    pt_GeV = p_GeV * np.sin(theta)
    return pt_GeV


def calculate_reduced_chi_squared(
    chi_squared: np.ndarray, num_dof: np.ndarray
) -> np.ndarray:
    """Calculate chi-squared per degree of freedom."""
    # Avoid division by zero: return inf where num_dof is zero or less
    # and chi_squared is non-zero. Return 0 if both are zero.
    result = np.full_like(chi_squared, fill_value=np.inf, dtype=np.float32)
    valid_dof_mask = num_dof > 0
    result[valid_dof_mask] = chi_squared[valid_dof_mask] / num_dof[valid_dof_mask]
    # Handle the case where chi_squared is 0 and num_dof is 0 (or less) -> result should be 0
    zero_chi_zero_dof_mask = (chi_squared == 0) & ~valid_dof_mask
    result[zero_chi_zero_dof_mask] = 0
    return result


# --- Registry ---
# Central dictionary mapping derived feature names to their definitions.
# Convention: All derived feature names should start with "derived."
DERIVED_FEATURE_REGISTRY: dict[str, DerivedFeature] = {
    "derived.InDetTrackParticlesAuxDyn.eta": DerivedFeature(
        name="derived.InDetTrackParticlesAuxDyn.eta",
        function=theta_to_eta,
        dependencies=["InDetTrackParticlesAuxDyn.theta"],
        shape=[2],
        dtype="float32",
    ),
    "derived.InDetTrackParticlesAuxDyn.pt": DerivedFeature(
        name="derived.InDetTrackParticlesAuxDyn.pt",
        function=calculate_pt,
        dependencies=[
            "InDetTrackParticlesAuxDyn.qOverP",
            "InDetTrackParticlesAuxDyn.theta",
        ],
        shape=[2],
        dtype="float32",
    ),
    "derived.InDetTrackParticlesAuxDyn.reducedChiSquared": DerivedFeature(
        name="derived.InDetTrackParticlesAuxDyn.reducedChiSquared",
        function=calculate_reduced_chi_squared,
        dependencies=[
            "InDetTrackParticlesAuxDyn.chiSquared",
            "InDetTrackParticlesAuxDyn.numberDoF",
        ],
        shape=[2],  # Assuming shape follows the pattern of other track features
        dtype="float32",
    ),
    # Add other derived features here following the same pattern
}

# --- Helper Functions ---


def is_derived_feature(branch_name: str) -> bool:
    """Check if a branch name corresponds to a defined derived feature."""
    return branch_name in DERIVED_FEATURE_REGISTRY


def get_derived_feature(branch_name: str):
    """Get the DerivedFeature object for a given branch name, if it exists."""
    return DERIVED_FEATURE_REGISTRY.get(branch_name)


def get_dependencies(branch_name: str):
    """Get the list of real branch dependencies for a derived feature name."""
    feature = get_derived_feature(branch_name)
    return feature.dependencies if feature else None
