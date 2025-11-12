"""
Watermark Detection Metrics

Computes detection rate and confidence statistics from extraction results.
"""

from typing import List, Dict, Any


def compute_detection_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute detection statistics from watermark extraction results.

    Args:
        results: List of extraction result dicts, each containing:
            - 'detected': bool, whether watermark was detected
            - 'confidence': float, confidence score (typically 0.0-1.0)

    Returns:
        dict: {
            'detection_rate': float,    # Percentage of images where watermark was detected (0.0-1.0)
            'avg_confidence': float,    # Average confidence score for detected watermarks
            'total_images': int,        # Total number of images evaluated
            'detected_count': int,      # Number of images with watermark detected
            'failed_count': int         # Number of images where watermark was not detected
        }

    Example:
        >>> results = [
        ...     {'detected': True, 'confidence': 0.95},
        ...     {'detected': True, 'confidence': 0.87},
        ...     {'detected': False, 'confidence': 0.12},
        ... ]
        >>> metrics = compute_detection_metrics(results)
        >>> print(f"Detection rate: {metrics['detection_rate']:.2%}")
        Detection rate: 66.67%
    """
    if not results:
        return {
            'detection_rate': 0.0,
            'avg_confidence': 0.0,
            'total_images': 0,
            'detected_count': 0,
            'failed_count': 0
        }

    total = len(results)
    detected = [r for r in results if r.get('detected', False)]
    detected_count = len(detected)
    failed_count = total - detected_count

    # Compute average confidence for detected watermarks
    avg_conf = 0.0
    if detected:
        avg_conf = sum(r.get('confidence', 0.0) for r in detected) / detected_count

    # Compute detection rate
    detection_rate = detected_count / total if total > 0 else 0.0

    return {
        'detection_rate': detection_rate,
        'avg_confidence': avg_conf,
        'total_images': total,
        'detected_count': detected_count,
        'failed_count': failed_count
    }


def compute_detection_metrics_by_strength(
    results_by_strength: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute detection metrics for multiple attack strengths.

    Args:
        results_by_strength: Dict mapping strength values to extraction results
            Example: {
                '1.2': [{'detected': True, 'confidence': 0.95}, ...],
                '1.4': [{'detected': True, 'confidence': 0.87}, ...],
            }

    Returns:
        dict: Metrics for each strength level

    Example:
        >>> results = {
        ...     '1.2': [{'detected': True, 'confidence': 0.95}] * 100,
        ...     '1.4': [{'detected': True, 'confidence': 0.85}] * 100,
        ... }
        >>> metrics = compute_detection_metrics_by_strength(results)
        >>> print(metrics['1.2']['detection_rate'])
        1.0
    """
    metrics_by_strength = {}

    for strength, results in results_by_strength.items():
        metrics_by_strength[str(strength)] = compute_detection_metrics(results)

    return metrics_by_strength


def print_detection_summary(metrics: Dict[str, float], indent: int = 0) -> None:
    """
    Pretty print detection metrics summary.

    Args:
        metrics: Detection metrics dict from compute_detection_metrics()
        indent: Number of spaces for indentation

    Example:
        >>> metrics = compute_detection_metrics(results)
        >>> print_detection_summary(metrics)
        Detection Rate: 95.00%
        Avg Confidence: 0.87
        Total Images: 1000
        Detected: 950
        Failed: 50
    """
    prefix = ' ' * indent
    print(f"{prefix}Detection Rate: {metrics['detection_rate']:.2%}")
    print(f"{prefix}Avg Confidence: {metrics['avg_confidence']:.4f}")
    print(f"{prefix}Total Images: {metrics['total_images']}")
    print(f"{prefix}Detected: {metrics['detected_count']}")
    print(f"{prefix}Failed: {metrics['failed_count']}")
