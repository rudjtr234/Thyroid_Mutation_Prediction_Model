"""
evaluation/visualization.py
ABMIL 성능 평가 및 시각화 함수들
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
from typing import List, Dict, Optional, Tuple

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 8)


def plot_confusion_matrix(true_labels: List[int], 
                         predictions: List[float], 
                         threshold: float = 0.5,
                         save_path: Optional[str] = None,
                         title: str = "Confusion Matrix") -> plt.Figure:
    """
    Confusion Matrix 시각화
    
    Args:
        true_labels: 실제 레이블 [0, 1, 1, 0, ...]
        predictions: 예측 확률 [0.2, 0.8, 0.9, 0.1, ...]
        threshold: 분류 임계값
        save_path: 저장 경로
        title: 그래프 제목
    """
    # 확률을 클래스로 변환
    pred_labels = (np.array(predictions) >= threshold).astype(int)
    
    # Confusion matrix 계산
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap 생성
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['BRAF-', 'BRAF+'],
                yticklabels=['BRAF-', 'BRAF+'],
                ax=ax)
    
    # 라벨 및 제목 설정
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # 성능 메트릭 추가
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 메트릭을 텍스트로 추가
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    ax.text(2.2, 1, metrics_text, fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_roc_curve(true_labels: List[int], 
                  predictions: List[float],
                  save_path: Optional[str] = None,
                  title: str = "ROC Curve") -> plt.Figure:
    """
    ROC Curve 시각화
    
    Args:
        true_labels: 실제 레이블
        predictions: 예측 확률
        save_path: 저장 경로
        title: 그래프 제목
    """
    # ROC curve 계산
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ROC curve 그리기
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    
    # 대각선 (랜덤 분류기) 그리기
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.500)')
    
    # 축 설정
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    return fig


def plot_precision_recall_curve(true_labels: List[int], 
                               predictions: List[float],
                               save_path: Optional[str] = None,
                               title: str = "Precision-Recall Curve") -> plt.Figure:
    """
    Precision-Recall Curve 시각화
    """
    # PR curve 계산
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    avg_precision = average_precision_score(true_labels, predictions)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # PR curve 그리기
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR Curve (AP = {avg_precision:.3f})')
    
    # Baseline (클래스 비율) 그리기
    baseline = np.mean(true_labels)
    ax.axhline(y=baseline, color='red', linestyle='--', lw=2,
               label=f'Baseline (AP = {baseline:.3f})')
    
    # 축 설정
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    
    return fig


def plot_learning_curves(train_losses: List[float],
                        train_accs: List[float],
                        val_losses: Optional[List[float]] = None,
                        val_accs: Optional[List[float]] = None,
                        save_path: Optional[str] = None,
                        title: str = "Learning Curves") -> plt.Figure:
    """
    학습 곡선 시각화
    
    Args:
        train_losses: 훈련 손실값들
        train_accs: 훈련 정확도들
        val_losses: 검증 손실값들 (선택사항)
        val_accs: 검증 정확도들 (선택사항)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss 곡선
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 곡선
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    if val_accs:
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    return fig


def plot_cross_validation_results(cv_results: List[Dict],
                                 save_path: Optional[str] = None,
                                 title: str = "Cross-Validation Results") -> plt.Figure:
    """
    Cross-Validation 결과 시각화
    
    Args:
        cv_results: [{'fold': 1, 'true_label': 1, 'pred_label': 0, 'avg_prob': 0.3}, ...]
    """
    # 데이터 준비
    folds = [r['fold'] for r in cv_results]
    true_labels = [r['true_label'] for r in cv_results]
    pred_probs = [r['avg_prob'] for r in cv_results]
    correct = [1 if r['true_label'] == r['pred_label'] else 0 for r in cv_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Fold별 예측 확률
    colors = ['red' if t == 0 else 'blue' for t in true_labels]
    ax1.scatter(folds, pred_probs, c=colors, s=100, alpha=0.7)
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Predicted Probability')
    ax1.set_title('Predictions by Fold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='BRAF-'),
                      Patch(facecolor='blue', label='BRAF+')]
    ax1.legend(handles=legend_elements)
    
    # 2. 정확도 막대 그래프
    ax2.bar(folds, correct, color=['green' if c else 'red' for c in correct], alpha=0.7)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Correct (1) / Incorrect (0)')
    ax2.set_title('Prediction Accuracy by Fold')
    ax2.set_ylim([0, 1.2])
    ax2.grid(True, alpha=0.3)
    
    # 3. 예측 확률 분포
    braf_pos_probs = [p for p, t in zip(pred_probs, true_labels) if t == 1]
    braf_neg_probs = [p for p, t in zip(pred_probs, true_labels) if t == 0]
    
    ax3.hist(braf_neg_probs, bins=10, alpha=0.7, color='red', label='BRAF-', density=True)
    ax3.hist(braf_pos_probs, bins=10, alpha=0.7, color='blue', label='BRAF+', density=True)
    ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Predictions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 성능 요약
    overall_acc = np.mean(correct)
    if len(set(true_labels)) > 1:  # 두 클래스 모두 존재
        from sklearn.metrics import roc_auc_score
        overall_auc = roc_auc_score(true_labels, pred_probs)
        auc_text = f"AUC: {overall_auc:.3f}"
    else:
        auc_text = "AUC: N/A (단일 클래스)"
    
    ax4.text(0.1, 0.7, f"Overall Accuracy: {overall_acc:.3f}", fontsize=16, 
             transform=ax4.transAxes)
    ax4.text(0.1, 0.5, auc_text, fontsize=16, transform=ax4.transAxes)
    ax4.text(0.1, 0.3, f"Total Folds: {len(cv_results)}", fontsize=16, 
             transform=ax4.transAxes)
    ax4.set_title('Overall Performance')
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CV results saved to: {save_path}")
    
    return fig


def plot_model_comparison(results_dict: Dict[str, Dict],
                         save_path: Optional[str] = None,
                         title: str = "Model Performance Comparison") -> plt.Figure:
    """
    여러 모델 성능 비교
    
    Args:
        results_dict: {
            'model_1': {'auc': 0.85, 'accuracy': 0.80, 'precision': 0.75, ...},
            'model_2': {'auc': 0.90, 'accuracy': 0.85, 'precision': 0.80, ...}
        }
    """
    # 메트릭들 추출
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results_dict.keys())
    
    # 각 메트릭별 값들 추출
    metric_values = {}
    for metric in metrics:
        metric_values[metric] = [results_dict[model].get(metric, 0) for model in model_names]
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_names, metric_values[metric], alpha=0.7)
        ax.set_title(metric.upper().replace('_', ' '), fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, metric_values[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # x축 레이블 회전
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 마지막 subplot 제거
    fig.delaxes(axes[-1])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    return fig


def create_comprehensive_report(true_labels: List[int],
                               predictions: List[float],
                               cv_results: Optional[List[Dict]] = None,
                               output_dir: str = "evaluation_results",
                               experiment_name: str = "experiment") -> None:
    """
    종합적인 평가 리포트 생성
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Creating comprehensive evaluation report...")
    print(f"Output directory: {output_dir}")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(
        true_labels, predictions,
        save_path=os.path.join(output_dir, f"{experiment_name}_confusion_matrix.png"),
        title=f"{experiment_name} - Confusion Matrix"
    )
    
    # 2. ROC Curve
    plot_roc_curve(
        true_labels, predictions,
        save_path=os.path.join(output_dir, f"{experiment_name}_roc_curve.png"),
        title=f"{experiment_name} - ROC Curve"
    )
    
    # 3. Precision-Recall Curve
    plot_precision_recall_curve(
        true_labels, predictions,
        save_path=os.path.join(output_dir, f"{experiment_name}_pr_curve.png"),
        title=f"{experiment_name} - Precision-Recall Curve"
    )
    
    # 4. Cross-Validation Results (있다면)
    if cv_results:
        plot_cross_validation_results(
            cv_results,
            save_path=os.path.join(output_dir, f"{experiment_name}_cv_results.png"),
            title=f"{experiment_name} - Cross-Validation Results"
        )
    
    print(f"✅ Report generated successfully!")
    print(f"📁 Check results in: {output_dir}")


if __name__ == "__main__":
    # 테스트용 예시 데이터
    np.random.seed(42)
    
    # 예시 데이터 생성
    true_labels = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
    predictions = [0.1, 0.3, 0.0, 1.0, 0.8, 0.9, 0.2, 0.7, 0.4, 0.1]
    
    cv_results = [
        {'fold': 1, 'true_label': 1, 'pred_label': 0, 'avg_prob': 0.1},
        {'fold': 2, 'true_label': 1, 'pred_label': 0, 'avg_prob': 0.3},
        {'fold': 3, 'true_label': 1, 'pred_label': 0, 'avg_prob': 0.0},
        {'fold': 4, 'true_label': 0, 'pred_label': 1, 'avg_prob': 1.0},
    ]
    
    # 시각화 테스트
    print("Testing visualization functions...")
    
    # 종합 리포트 생성
    create_comprehensive_report(
        true_labels=true_labels,
        predictions=predictions,
        cv_results=cv_results,
        output_dir="test_results",
        experiment_name="abmil_test"
    )
