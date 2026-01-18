import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(path_excel: str = "XSS_dataset.xls", path_csv: str = "XSS_dataset.csv") -> pd.DataFrame:
    """
    Load the XSS dataset.

    Primary source is an Excel file via pandas.read_excel as requested.
    If the Excel file is not found, fall back to a CSV file if available.
    """
    if os.path.exists(path_excel):
        # Use read_excel as requested. Encoding is typically handled internally for Excel.
        df = pd.read_excel(path_excel)
    elif os.path.exists(path_csv):
        # Fallback: some distributions provide CSV instead of XLS.
        # Use a standard read_csv signature compatible with most pandas versions.
        df = pd.read_csv(path_csv, encoding="utf-8")
    else:
        raise FileNotFoundError(
            f"Neither {path_excel} nor {path_csv} could be found in the current directory."
        )

    print("Loaded dataset with shape:", df.shape)
    print(df.head())
    return df


def rename_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename 'Sentence' -> 'text' and 'Label' -> 'label'.
    Drop rows where text/label is NaN.
    Apply minimal preprocessing: lowercase + strip, keeping symbols.
    """
    # Step: rename columns
    df = df.rename(columns={"Sentence": "text", "Label": "label"})

    # Step: drop rows with missing text or label
    df = df.dropna(subset=["text", "label"])

    # Step: minimal preprocessing (preserve symbols such as <, >, /, =)
    df["text"] = df["text"].astype(str).str.lower().str.strip()

    print("After cleaning, dataset shape:", df.shape)
    return df


def split_dataset(df: pd.DataFrame):
    """
    Split dataset into train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"] if "label" in df and df["label"].nunique() > 1 else None,
    )
    return X_train, X_test, y_train, y_test


def vectorize_tfidf(X_train, X_test):
    """
    Apply TF-IDF vectorizer to convert text into numerical vectors.
    """
    tfidf = TfidfVectorizer()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    return tfidf, X_train_vec, X_test_vec


def train_model(X_train_vec, y_train):
    """
    Train Logistic Regression model.
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    return clf


def evaluate_model(clf, X_test_vec, y_test):
    """
    Evaluate the model and print/plot metrics:
    - Confusion matrix heatmap
    - ROC curve
    - Precision-Recall curve
    - Bar chart of weighted F1 / precision / recall
    """
    y_pred = clf.predict(X_test_vec)

    # Text metrics
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nClassification Report:")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred)
    print(report_text)

    # Save key metrics to JSON so Flask UI can display them
    weighted = report_dict["weighted avg"]
    metrics_summary = {
        "accuracy": float(report_dict.get("accuracy", 0.0)),
        "weighted_precision": float(weighted.get("precision", 0.0)),
        "weighted_recall": float(weighted.get("recall", 0.0)),
        "weighted_f1": float(weighted.get("f1-score", 0.0)),
        "confusion_matrix": cm.tolist(),
        "labels": [str(c) for c in getattr(clf, "classes_", [])],
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    # Confusion Matrix Heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=getattr(clf, "classes_", None),
        yticklabels=getattr(clf, "classes_", None),
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ROC Curve & Precision-Recall Curve (require predict_proba)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test_vec)
        # Assume binary classification; take probability of the "positive" class
        if y_proba.shape[1] >= 2:
            pos_index = 1
        else:
            pos_index = 0

        y_score = y_proba[:, pos_index]
        pos_label = clf.classes_[pos_index] if hasattr(clf, "classes_") else 1

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=pos_label)
        plt.figure()
        plt.plot(recall, precision, label="Precision-Recall curve")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()

    # Bar chart for weighted average Precision / Recall / F1
    metrics_names = ["Precision", "Recall", "F1-score"]
    values = [weighted["precision"], weighted["recall"], weighted["f1-score"]]

    plt.figure()
    sns.barplot(x=metrics_names, y=values)
    plt.ylim(0.0, 1.05)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.title("Weighted Average Metrics")
    plt.tight_layout()
    plt.show()


def save_artifacts(clf, tfidf, model_path: str = "xss_model.pkl", vec_path: str = "tfidf_vectorizer.pkl"):
    """
    Save the trained model and TF-IDF vectorizer as .pkl files.
    """
    joblib.dump(clf, model_path)
    joblib.dump(tfidf, vec_path)
    print(f"Saved model to {model_path} and vectorizer to {vec_path}")


def main():
    # 1. Load dataset
    df = load_dataset()

    # 2. Rename columns and preprocess
    df = rename_and_clean(df)

    # 2b. Save basic dataset info for UI (size, label distribution)
    label_counts = df["label"].value_counts()
    dataset_info = {
        "n_samples": int(len(df)),
        "labels": [str(k) for k in label_counts.index.tolist()],
        "label_counts": {str(k): int(v) for k, v in label_counts.to_dict().items()},
    }
    with open("dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_dataset(df)

    # 4. TF-IDF vectorization
    tfidf, X_train_vec, X_test_vec = vectorize_tfidf(X_train, X_test)

    # 5. Train model
    clf = train_model(X_train_vec, y_train)

    # 6. Evaluate model
    evaluate_model(clf, X_test_vec, y_test)

    # 7. Save model and vectorizer
    save_artifacts(clf, tfidf)


if __name__ == "__main__":
    main()


