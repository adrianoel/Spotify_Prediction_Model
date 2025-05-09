# Spotify Prediction Model
In diesem Projekt geht es um die Frage:  
**"Wie gut lässt sich die Beliebtheit eines Songs auf Spotify mithilfe eines Machine-Learning-Modells vorhersagen?"** 

## Inhaltsverzeichnis
- [Projektüberblick](#projektüberblick)
- [Projektstruktur](#projektstruktur)
- [Voraussetzungen](#voraussetzungen)
- 
- 
-

---

## Projektüberblick
In diesem Projekt werden die Meta- und Audiodaten, die sich ursprünglich von der Spotify API erhalten ließen, analysiert, um herauszufinden, welche Merkmale einflussreiche Faktoren für den Beliebtheitswert darstellen. Dies beinhaltet die Datenbeschaffung, Datenvorverarbeitung, explorative Datenanalyse, Modellierung, Hyperparamter-Optimierung und das Testen von Komponenten (letzter Punkt noch zu implementieren).

## Projektstruktur
Die Projektstruktur ist wie folgt organisiert:

```
Spotify_Prediction_Model/
├── .venv/
├── classification_reports/
    └── log_model_classification_report.csv
    └── rfc_best_model_classification_report.csv
    └── rfc_model_classification_report.csv
    └── simple_model_classification_report.csv
├── data/
├── src/
    └── data_retrieval/
        └── download_from_kagglehub.py
    └── features/
        └── data_prep_for_model.py
    └── models/
        └── final_model.py
    └── visualization/
        └── create_plots.py
    └── __init__.py
├── plots/
    └── .gitkeep
├── presentation_slides_short/
    └── Spotify_Prediction_Model_Präsentation
├── tests/
├── .gitignore
├── .python-version
├── EDA.ipynb
├── Feature_Engineering.ipynb
├── FinalBaseModel.ipynb
├── FinalVisualizations.ipynb
├── Hyperparameter_Tuning.ipynb
├── pyproject.toml
├── README.md
├── SimpleBaselineModel.ipynb
└── uv.lock
```

- **`.venv/`**: Virtuelle Python-Umgebung für das Projekt.
- **`.classification_reports/`**: Classification reports der genutzten Modelle im Laufe des Projekts zum Betrachten und Vergleichen.
- **`data/`**: Ordner für den heruntergeladenen Datensatz.
- **`src/`**: Ordner für die genutzten Skripte:
    - **`src/data_retrieval/`**: Ordner für Skripte zum Erhalten und Speichern von Datensätzen im data/ Ordner.
    - **`src/data_retrieval/`**: Ordner für Skripte für die Datenbereinigung, das Feature Engineering und die Datenvorbereitung sowie der Pipeline eines Modells.
    - **`src/data_retrieval/`**: Ordner für Skripte zum finalen Modell.
    - **`src/data_retrieval/`**: Ordner für Skripte zum Erstellen von ausgewählten Plots zur Visualisierung.
    - **`src/__init__.py`**: Initialisiert den src/ Ordner und dessen Skripte.
- **`plots/`**: Ordner für die durch das Skript erstellten Plots.
- **`presentation_slides_short/`**: Ordner für die reduzierte Abschlusspräsentation des Projekts.
- **`tests/`**: Ordner für Funktionstests der Skripte im Projekt. (noch zu implementieren)
- **`.gitignore`**: Definiert, welche Dateien von der Versionskontrolle ausgeschlossen werden.
- **`.python-version`**: Spezifiziert die Python-Version (>=3.13).
- **`EDA.ipynb`**: Jupyter Notebook für die explorative Datenanalyse.
- **`Feature_Engineering.ipynb`**: Jupyter Notebook für das Feature Engineering.
- **`FinalBaseModel.ipynb`**: Jupyter Notebook zur Implementierung und Testen des finalen Modells mit Verwendung der Pipeline.
- **`FinalVisualizations.ipynb`**: Jupyter Notebook für die Erstellung und Speicherung von Plots zur Visualisierung.
- **`Hyperparameter_Tuning.ipynb`**: Jupyter Notebook für die Hyperparameter-Optimierung des finalen Modells mit Optuna.
- **`pyproject.toml`**: Projektkonfigurationsdatei mit Abhängigkeiten.
- **`README.md`**: Diese Dokumentation.
- **`SimpleBaselineModel.ipynb`**: Jupyter Notebook zur Implementierung eines ersten simplen Baseline Modells als Vergleichsmodell.
- **`uv.lock`**: Lock-Datei für den Paketmanager uv.

## Voraussetzungen

(to be continued)



....

## Projektergebnisse (am Schluss) / Fazit