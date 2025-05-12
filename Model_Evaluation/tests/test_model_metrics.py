# tests/test_inference.py
import pytest
import mlflow

def test_overall_accuracy(predictions):
    accuracy, mse, f1_score, recall, precision = predictions
    acc_pct = accuracy * 100

    mlflow.log_metric("overall_accuracy", acc_pct)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("overall_accuracy_passed", int(acc_pct > 60))
    mlflow.log_metric("overall_mse_passed", int(mse < 80))
    mlflow.log_metric("overall_f1_score_passed", int(f1_score > 0.5))
    mlflow.log_metric("overall_recall_passed", int(recall > 0.4))
    mlflow.log_metric("overall_precision_passed", int(precision > 0.4))

    # assert
    assert acc_pct > 60, f"Overall accuracy too low: {acc_pct:.2f}% (MSE={mse:.4f})"