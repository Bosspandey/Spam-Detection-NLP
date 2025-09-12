import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=['Ham', 'Spam']):
    """
    Plot a confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt

def display_metrics(accuracy, cm, report):
    """
    Display model metrics in a formatted way
    """
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)