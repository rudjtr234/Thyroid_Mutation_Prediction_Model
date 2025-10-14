import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)


def comprehensive_evaluation(true_labels, predictions, threshold=0.5):
    """
    모델 예측 결과에 대한 종합 평가 수행
    Args:
        true_labels (list[int]): 실제 라벨 (0/1)
        predictions (list[float]): 예측 확률값
        threshold (float): Positive 판정 기준값 (default=0.5)

    Returns:
        dict: 성능 지표 (AUC, Accuracy, Precision, Recall, F1-score, Confusion Matrix, Report)
    """
    pred_labels = (np.array(predictions) >= threshold).astype(int)

    results = {
        'auc': roc_auc_score(true_labels, predictions) if len(set(true_labels)) > 1 else 0.5,
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, zero_division=0),
        'recall': recall_score(true_labels, pred_labels, zero_division=0),
        'f1_score': f1_score(true_labels, pred_labels, zero_division=0),
        'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist(),
        'classification_report': classification_report(true_labels, pred_labels, zero_division=0)
    }
    return results
