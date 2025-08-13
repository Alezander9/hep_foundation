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

    # Check for problematic input values
    n_extreme_theta = np.sum((theta < epsilon) | (theta > np.pi - epsilon))

    theta_clipped = np.clip(theta, epsilon, np.pi - epsilon)

    # Additional safeguarding to prevent extreme values
    tan_half_theta = np.tan(theta_clipped / 2.0)
    # Ensure tan_half_theta is not too close to 0 to prevent -inf
    tan_half_theta = np.clip(tan_half_theta, epsilon, 1.0 / epsilon)

    eta = -np.log(tan_half_theta)

    # Count values that need clipping
    n_clipped = np.sum((eta < -10.0) | (eta > 10.0))

    # Final clipping to prevent extreme eta values (typical physics range is roughly -5 to 5)
    eta = np.clip(eta, -10.0, 10.0)

    # Optional debug logging (only log if there are issues)
    if n_extreme_theta > 0 or n_clipped > 0:
        import logging

        logger = logging.getLogger(__name__)
        if n_extreme_theta > 0:
            logger.debug(
                f"eta calculation: {n_extreme_theta} tracks with extreme theta values (clipped)"
            )
        if n_clipped > 0:
            logger.debug(
                f"eta calculation: {n_clipped} tracks clipped to range [-10, 10]"
            )

    return eta


def calculate_pt(qOverP: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Calculate pT (in GeV) from qOverP (in MeV⁻¹) and theta (radians)."""

    # Handle potential division by zero safely
    max_pt = 100.0  # 100 GeV - reasonable upper limit for most particles
    min_abs_qOverP = 1e-9  # Corresponds to ~1 TeV momentum in MeV units
    epsilon_theta = 1e-6  # Small value for theta calculations

    # Check for problematic input values
    n_zero_qOverP = np.sum(np.abs(qOverP) < min_abs_qOverP)
    n_extreme_theta = np.sum((theta < epsilon_theta) | (theta > np.pi - epsilon_theta))

    # Handle very small qOverP by capping momentum rather than replacing with tiny epsilon
    # If |qOverP| is too small, it means momentum would be too large - cap it
    # Special handling for exactly zero qOverP to avoid np.sign(0) = 0 issue
    qOverP_safe = np.where(
        np.abs(qOverP) < min_abs_qOverP,
        np.where(
            qOverP == 0.0,
            min_abs_qOverP,  # Use positive value for exactly zero
            np.sign(qOverP) * min_abs_qOverP,  # Use sign for non-zero but small values
        ),
        qOverP,
    )

    # Calculate momentum magnitude in MeV, then convert to GeV
    p_MeV = np.abs(1.0 / qOverP_safe)
    p_GeV = p_MeV / 1000.0  # Convert MeV to GeV

    # Ensure theta is also safe for sin calculation
    theta_safe = np.clip(theta, epsilon_theta, np.pi - epsilon_theta)
    sin_theta = np.sin(theta_safe)

    # Calculate pT
    pt_GeV = p_GeV * sin_theta

    # Count values that need clipping (should be minimal now)
    n_clipped = np.sum(pt_GeV > max_pt)

    # Final clipping to ensure we stay within reasonable physics range
    pt_GeV = np.clip(pt_GeV, 0.0, max_pt)

    # Optional debug logging (only log if there are issues)
    if n_zero_qOverP > 0 or n_extreme_theta > 0 or n_clipped > 0:
        import logging

        logger = logging.getLogger(__name__)
        if n_zero_qOverP > 0:
            logger.debug(
                f"pt calculation: {n_zero_qOverP} tracks with qOverP ≈ 0 (momentum capped)"
            )
        if n_extreme_theta > 0:
            logger.debug(
                f"pt calculation: {n_extreme_theta} tracks with extreme theta values"
            )
        if n_clipped > 0:
            logger.debug(
                f"pt calculation: {n_clipped} tracks clipped to max_pt={max_pt} GeV"
            )

    return pt_GeV


def calculate_reduced_chi_squared(
    chi_squared: np.ndarray, num_dof: np.ndarray
) -> np.ndarray:
    """Calculate chi-squared per degree of freedom."""
    # Use a large but finite value instead of inf for invalid cases
    max_reduced_chi2 = 1000.0  # Very large but finite reduced chi-squared
    epsilon = 1e-8  # Small value to avoid exact zero division

    # Check for problematic input values
    invalid_dof_mask = num_dof <= epsilon
    n_invalid_dof = np.sum(invalid_dof_mask)
    zero_chi_mask = invalid_dof_mask & (np.abs(chi_squared) < epsilon)
    nonzero_chi_invalid_dof_mask = invalid_dof_mask & (np.abs(chi_squared) >= epsilon)
    n_zero_chi_zero_dof = np.sum(zero_chi_mask)
    n_nonzero_chi_invalid_dof = np.sum(nonzero_chi_invalid_dof_mask)

    # Initialize result array
    result = np.zeros_like(chi_squared, dtype=np.float32)

    # Valid cases: num_dof > 0
    valid_dof_mask = ~invalid_dof_mask
    result[valid_dof_mask] = chi_squared[valid_dof_mask] / num_dof[valid_dof_mask]

    # If chi_squared is also very small/zero, set to 0
    result[zero_chi_mask] = 0.0

    # If chi_squared is non-zero but num_dof is invalid, set to large finite value
    result[nonzero_chi_invalid_dof_mask] = max_reduced_chi2

    # Count values that need clipping
    n_clipped = np.sum(result > max_reduced_chi2)

    # Final clipping to prevent extreme values
    result = np.clip(result, 0.0, max_reduced_chi2)

    # Optional debug logging (only log if there are issues)
    if n_invalid_dof > 0 or n_clipped > 0:
        import logging

        logger = logging.getLogger(__name__)
        if n_zero_chi_zero_dof > 0:
            logger.debug(
                f"reduced χ²: {n_zero_chi_zero_dof} tracks with χ²≈0 and DoF≤0 (set to 0)"
            )
        if n_nonzero_chi_invalid_dof > 0:
            logger.debug(
                f"reduced χ²: {n_nonzero_chi_invalid_dof} tracks with χ²>0 but DoF≤0 (set to {max_reduced_chi2})"
            )
        if n_clipped > 0:
            logger.debug(
                f"reduced χ²: {n_clipped} tracks clipped to max={max_reduced_chi2}"
            )

    return result


# --- MET utilities ---


def calculate_met_norm(mpx: np.ndarray, mpy: np.ndarray) -> np.ndarray:
    """Compute MET magnitude from x/y components.

    Minimal safeguards: replace non-finite inputs with 0.
    """
    mpx_safe = np.where(np.isfinite(mpx), mpx, 0.0)
    mpy_safe = np.where(np.isfinite(mpy), mpy, 0.0)
    met = np.sqrt(mpx_safe * mpx_safe + mpy_safe * mpy_safe)
    return met.astype(np.float32)


def calculate_met_phi(mpx: np.ndarray, mpy: np.ndarray) -> np.ndarray:
    """Compute MET azimuth from x/y components.

    Minimal safeguards: replace non-finite inputs with 0, ensure finite output.
    Range is [-pi, pi] by definition of arctan2.
    """
    mpx_safe = np.where(np.isfinite(mpx), mpx, 0.0)
    mpy_safe = np.where(np.isfinite(mpy), mpy, 0.0)
    phi = np.arctan2(mpy_safe, mpx_safe).astype(np.float32)
    phi = np.where(np.isfinite(phi), phi, 0.0).astype(np.float32)
    return phi


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
    # --- MET derived features (basic magnitude and direction) ---
    "derived.MET_Core_AnalysisMETAuxDyn.met_norm": DerivedFeature(
        name="derived.MET_Core_AnalysisMETAuxDyn.met_norm",
        function=calculate_met_norm,
        dependencies=[
            "MET_Core_AnalysisMETAuxDyn.mpx",
            "MET_Core_AnalysisMETAuxDyn.mpy",
        ],
        shape=[2],
        dtype="float32",
    ),
    "derived.MET_Core_AnalysisMETAuxDyn.met_phi": DerivedFeature(
        name="derived.MET_Core_AnalysisMETAuxDyn.met_phi",
        function=calculate_met_phi,
        dependencies=[
            "MET_Core_AnalysisMETAuxDyn.mpx",
            "MET_Core_AnalysisMETAuxDyn.mpy",
        ],
        shape=[2],
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
