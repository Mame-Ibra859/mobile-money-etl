# Projet 2 — Mobile Money ETL & Fraud Detection

Ce projet met en place une pipeline ETL sur le dataset PaySim, puis applique un modèle d’anomalies `IsolationForest` pour détecter les transactions frauduleuses.

## Objectif

Le but est de :

- charger les données brutes ;
- nettoyer et préparer les colonnes utiles ;
- créer des features métier pour la détection d’anomalies ;
- appliquer un modèle d’isolation des anomalies ;
- exporter les résultats prédits dans le dossier `data/processed/`.

## Données

Le dataset utilisé est le jeu de données PaySim, un dataset synthétique de transactions mobile money destiné à la détection de fraude.

Données brutes :
- `data/raw/PS_20174392719_1491204439457_log.csv`

Données traitées :
- `data/processed/fraud_predictions.csv`

## Structure du projet

```text
mobile-money-etl/
├── data/
│   ├── raw/
│   │   └── PS_20174392719_1491204439457_log.csv
│   └── processed/
│       └── fraud_predictions.csv
├── notebooks/
│   └── README.md
├── src/
│   ├── extract.py
│   ├── transform.py
│   ├── detect_fraud.py
│   └── pipeline.py
├── tests/
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Pipeline ETL

### 1. Extraction

La fonction `extract` lit le fichier CSV brut depuis le dossier `data/raw/`.

```python
import pandas as pd


def extract(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
```

### 2. Transformation

On crée des variables utiles pour distinguer les transactions normales des transactions inhabituelles :

- `balance_change_origin`
- `balance_change_destination`
- `amount_to_balance_ratio`
- `is_large_transaction`

Ces colonnes sont importantes car elles traduisent le comportement de la transaction, pas seulement le montant brut. Par exemple, un transfert peut paraître normal en valeur absolue, mais devenir suspect si le solde du compte source diminue de façon disproportionnée, ou si le montant représente une fraction très élevée du solde disponible. De même, un montant exceptionnellement élevé par rapport aux transactions habituelles est souvent un bon indicateur d’anomalie.

```python
import pandas as pd


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["balance_change_origin"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_change_destination"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["is_large_transaction"] = (df["amount"] > df["amount"].quantile(0.99)).astype(int)

    return df
```

La variable `balance_change_origin` montre si le compte source a été vidé de façon anormale ; `balance_change_destination` montre si le compte destinataire a reçu un montant inhabituel ; `amount_to_balance_ratio` traduit la proportion du montant par rapport au solde initial ; enfin, `is_large_transaction` identifie les transactions hors normes par rapport à la distribution du jeu de données.

### 3. Détection de fraude

Le modèle `IsolationForest` repère les transactions inhabituelles en les traitant comme des anomalies.

```python
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_fraud(df: pd.DataFrame, contamination: float = 0.001) -> pd.DataFrame:
    feature_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balance_change_origin",
        "balance_change_destination",
        "amount_to_balance_ratio",
        "is_large_transaction",
    ]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(df[feature_cols])
    df = df.copy()
    df["fraud_prediction"] = (predictions == -1).astype(int)

    return df
```

### 4. Pipeline complète

```python
from extract import extract
from transform import transform
from detect_fraud import detect_fraud


def main():
    df = extract("data/raw/PS_20174392719_1491204439457_log.csv")
    df = transform(df)
    df = detect_fraud(df)
    df.to_csv("data/processed/fraud_predictions.csv", index=False)


if __name__ == "__main__":
    main()
```

## Lancer le projet localement

1. Créer un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate
```

Sous Windows PowerShell :
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Exécuter la pipeline :
```bash
python src/pipeline.py
```

## Docker

### Dockerfile

```dockerfile
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

CMD ["python", "src/pipeline.py"]
```

### Commandes

```bash
docker build -t mobile-money-etl .
docker run --rm mobile-money-etl
```

## CI GitHub Actions

Le projet peut être validé à chaque push / pull request avec un workflow de test.

```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest -q
```

## Notebook Colab

Le notebook d’analyse exploratoire est accessible via Google Colab :

https://colab.research.google.com/drive/17QCf-el4hpoFAr8P4jRX-iP3G6sJfLav?usp=sharing

Le fichier de documentation associé est dans `notebooks/README.md`.

