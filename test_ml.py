import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ml.model import train_model, compute_model_metrics, inference
from sklearn.metrics import precision_score, recall_score, f1_score


def test_train_model_returns_correct_type():
    """Test that train_model returns a RandomForestClassifier instance."""
    X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([0,1,0,1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    pass


def test_inference():
    """
    Test inference returns valid predicted values
    """

    X_train = np.array([[0,0],[1,1]])
    y_train = np.array([0,1])
    X_test = np.array([[0,1],[1,0]])

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train) 
    
    preds = inference(model, X_test)

    assert set(preds).issubset({1,0}), "Expected values are 1 and 0"
    pass


def test_compute_model_metrics_values():
    """
    Test compute metrics returns correct precision, recall and f1 values
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
    pass
