import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def make_samples_from_df(df: pd.DataFrame, n_samples: int = 10) -> str:
    """
    function to generate samples
    """
    df_label_1 = df[df["Label"] == 1].sample(n_samples, random_state=42)
    df_label_0 = df[df["Label"] == 0].sample(n_samples, random_state=42)
    df_sampled = pd.concat([df_label_1, df_label_0])
    samples = ",\n".join(
        [
            f'"texto: {row["Text"]}" : "etiqueta: {row["Label"]}"'
            for _, row in df_sampled.iterrows()
        ]
    )
    return samples


def make_prompt(samples: str, text: str) -> str:
    """
    function to generate prompt
    """
    prompt = f"""Basado en los ejemplos: {samples}.
    Clasifica el texto en 1 si hay ideación/comportamiento suicida; 0 en caso que no, y retorna la etiqueta.
    texto: {text}
    etiqueta: """.strip()
    return prompt


def plot_confusion_matrix(
    df: pd.DataFrame,
    true_label_column: str = "Label",
    pred_label_column: str = "llm_label",
) -> None:
    """
    Function to plot confusion matrix
    """
    # Crear la matriz de confusión
    cm = confusion_matrix(df[true_label_column], df[pred_label_column], labels=[0, 1])

    # Visualizar la matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Imprimir el informe de clasificación
    print(
        classification_report(
            df[true_label_column], df[pred_label_column], labels=[0, 1]
        )
    )
