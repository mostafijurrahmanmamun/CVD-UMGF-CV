import numpy as np

def compute_ece(probs, labels, num_bins=10):
    """
    Computes Expected Calibration Error (ECE) across equal-width confidence bins.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return ece

def evaluate_selective_prediction(probs, vacuity_u, labels, abstention_thresholds=[0.0, 0.05, 0.10, 0.15, 0.20]):
    """
    Evaluates selective prediction accuracy by deferring high-vacuity cases to expert review.
    """
    results = []
    num_samples = len(labels)
    
    for rejection_rate in abstention_thresholds:
        num_deferred = int(num_samples * rejection_rate)
        if num_deferred > 0:
            # Sort samples by vacuity uncertainty descending
            sorted_indices = np.argsort(vacuity_u)[::-1]
            retained_indices = sorted_indices[num_deferred:]
        else:
            retained_indices = np.arange(num_samples)
            
        retained_probs = probs[retained_indices]
        retained_labels = labels[retained_indices]
        
        preds = np.argmax(retained_probs, axis=1)
        acc = np.mean(preds == retained_labels)
        ece = compute_ece(retained_probs, retained_labels)
        
        results.append({
            "rejection_rate": rejection_rate * 100.0,
            "coverage": (len(retained_indices) / num_samples) * 100.0,
            "accuracy": acc,
            "ece": ece,
            "deferred_count": num_deferred
        })
        
    return results
