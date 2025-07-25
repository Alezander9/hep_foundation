import logging
from typing import Optional


def log_evaluation_summary(
    results: dict[str, list[float]],
    evaluation_type: str = "regression",
    signal_key: Optional[str] = None,
) -> None:
    """
    Log a summary of evaluation results.

    Args:
        results: Dictionary containing evaluation results
        evaluation_type: Type of evaluation ("regression" or "signal_classification")
        signal_key: Signal key for classification evaluations
    """

    logger = logging.getLogger(__name__)

    try:
        logger.info("=" * 100)
        if evaluation_type == "regression":
            logger.info("Regression Evaluation Results Summary")
        elif evaluation_type == "signal_classification":
            logger.info("Signal Classification Evaluation Results Summary")
            if signal_key:
                logger.info(f"Signal Dataset: {signal_key}")
        logger.info("=" * 100)

        data_sizes = results.get("data_sizes", [])

        for i, data_size in enumerate(data_sizes):
            logger.info(f"Training Events: {data_size}")

            if evaluation_type == "regression":
                logger.info(f"  From Scratch:  {results['From_Scratch'][i]:.6f}")
                logger.info(f"  Fine-Tuned:    {results['Fine_Tuned'][i]:.6f}")
                logger.info(f"  Fixed Encoder: {results['Fixed_Encoder'][i]:.6f}")

                # Calculate improvement ratios
                if results["From_Scratch"][i] > 0:
                    ft_improvement = (
                        (results["From_Scratch"][i] - results["Fine_Tuned"][i])
                        / results["From_Scratch"][i]
                        * 100
                    )
                    fx_improvement = (
                        (results["From_Scratch"][i] - results["Fixed_Encoder"][i])
                        / results["From_Scratch"][i]
                        * 100
                    )
                    logger.info(f"  Fine-Tuned improvement: {ft_improvement:.1f}%")
                    logger.info(f"  Fixed improvement: {fx_improvement:.1f}%")

            elif evaluation_type == "signal_classification":
                logger.info(
                    f"  From Scratch:  Loss: {results['From_Scratch_loss'][i]:.6f}, "
                    f"Accuracy: {results['From_Scratch_accuracy'][i]:.6f}"
                )
                logger.info(
                    f"  Fine-Tuned:    Loss: {results['Fine_Tuned_loss'][i]:.6f}, "
                    f"Accuracy: {results['Fine_Tuned_accuracy'][i]:.6f}"
                )
                logger.info(
                    f"  Fixed Encoder: Loss: {results['Fixed_Encoder_loss'][i]:.6f}, "
                    f"Accuracy: {results['Fixed_Encoder_accuracy'][i]:.6f}"
                )

                # Calculate improvement ratios for accuracy
                scratch_acc = results["From_Scratch_accuracy"][i]
                if scratch_acc < 1.0:  # Avoid division issues
                    ft_acc_improvement = (
                        (results["Fine_Tuned_accuracy"][i] - scratch_acc)
                        / (1.0 - scratch_acc)
                        * 100
                    )
                    fx_acc_improvement = (
                        (results["Fixed_Encoder_accuracy"][i] - scratch_acc)
                        / (1.0 - scratch_acc)
                        * 100
                    )
                    logger.info(
                        f"  Fine-Tuned accuracy improvement: {ft_acc_improvement:.1f}%"
                    )
                    logger.info(
                        f"  Fixed accuracy improvement: {fx_acc_improvement:.1f}%"
                    )

            logger.info("")

    except Exception as e:
        logger.error(f"Failed to log evaluation summary: {str(e)}")
