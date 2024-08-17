from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> None:
        super().__init__()
        self.df_val = df_val
        self.df_train = df_train
        self.samples = self._make_samples_from_df(n_samples=10)

    def __len__(self):
        return len(self.df_val)

    def __getitem__(self, i):
        text = self.df_val["Text"].iloc[i]
        few_shot_prompt = self._make_prompt(text)
        return few_shot_prompt

    def _make_prompt(self, text: str) -> str:
        prompt = f"""Basado en los ejemplos: {self.samples}.
        Clasifica el texto en 1 si hay ideación/comportamiento suicida; 0 en caso que no, y retorna la etiqueta.
        texto: {text}
        eqtiqueta: """.strip()
        return prompt

    def _make_samples_from_df(self, n_samples: int = 10):
        df_label_1 = self.df_train[self.df_train["Label"] == 1].sample(
            n_samples, random_state=42
        )
        df_label_0 = self.df_train[self.df_train["Label"] == 0].sample(
            n_samples, random_state=42
        )
        df_sampled = pd.concat([df_label_1, df_label_0])
        samples = ",\n ".join(
            [
                f'"texto: {row["Text"]}": "etiqueta: {row["Label"]}"'
                for _, row in df_sampled.iterrows()
            ]
        )
        return samples
