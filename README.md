# Spotify Prediction Model
In diesem Abschlussprojekt geht es um die Frage:  
**"Wie gut lässt sich die Beliebtheit eines Songs auf Spotify mithilfe eines Machine-Learning-Modells vorhersagen?"** 

## Inhaltsverzeichnis
- [Projektüberblick](#projektüberblick)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Daten herunterladen](#daten-herunterladen)
- [Explorative Datenanalyse (EDA)](#explorative-datenanalyse-eda)
- [Feature Engineering, Modellierung und Visualisierung](#feature-engineering-modellierung-und-visualisierung)
- [Hyperparameter-Optimierung](#hyperparameter-optimierung)
- [Testen (work in progress)](#testen-work-in-progress)
- [Verwendete Technologien und Bibliotheken](#verwendete-technologien-und-bibliotheken)
- [Fazit](#fazit)

---

## Projektüberblick
In diesem Projekt werden die Meta- und Audiodaten, die sich ursprünglich von der Spotify API erhalten ließen, analysiert, um herauszufinden, welche Merkmale einflussreiche Faktoren für den Beliebtheitswert darstellen. Dies beinhaltet die Datenbeschaffung, Datenvorverarbeitung, explorative Datenanalyse, Modellierung, Hyperparamter-Optimierung und das Testen von Komponenten.

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
    └── download_from_kagglehub.py
    └── data_prep_for_model.py
    └── final_model.py
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
    - **`src/download_from_kagglehub.py`**: Skript zum Erhalten und Speichern von Datensätzen im data/ Ordner.
    - **`src/data_prep_for_model.py`**: Skript für die Datenbereinigung, das Feature Engineering und die Datenvorbereitung sowie der Pipeline eines Modells.
    - **`src/final_model.py`**: Skript zum finalen Modell.
    - **`src/create_plots.py`**: Skript zum Erstellen von ausgewählten Plots zur Visualisierung.
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

Für dieses Projekt wird Python und der Paketmanager uv benötigt. Alle weiteren Abhängigkeiten sind in der `pyproject.toml` Datei bzw. in der `uv.lock` dokumentiert.

#### macOS

1. **Homebrew installieren** (falls nicht bereits installiert):
   
   Das Terminal öffnen und folgenden Befehl ausführen:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **uv installieren**:

   Nachdem Homebrew installiert ist, folgenden Befehl im Terminal ausführen:

   ```bash
   brew install uv
   ```


#### Linux

Python und uv mit dem jeweiligen Packetmanager installieren.

Für Debian/Ubuntu mit:

```bash
sudo apt update
sudo apt install python3 python3-pip
pip3 install uv
```

#### Windows

1. **Winget installieren** (falls nicht bereits installiert):

   Winget ist in den neuesten Windows-Versionen standardmäßig installiert. Andernfalls lässt es sich über [Microsoft Store App Installer](https://apps.microsoft.com/store/detail/9NBLGGH4NNS1) installieren.

2. **uv installieren**:

   PowerShell öfnen und folgenden Befehl ausführen:

   ```powershell
   winget install astral-sh.uv
   ```

Sobald Python und uv installiert sind, kann das Projekt initialisiert und die Abhängigkeiten mit uv installiert werden (siehe Anleitungen zur Nutzung von uv).

## Installation

1. **Repository klonen**

   Das Repository kann geklont oder als ZIP-Datei heruntergeladen werden:

   ```bash
   git clone <REPOSITORY_URL>
   cd DPP-Referenzprojekt
   ```

2. **Virtuelle Umgebung einrichten**

   Die virtuelle Umgebung initialisieren und die Abhängigkeiten installieren mit:

   ```bash
   uv sync
   ```

   Dies erstellt die virtuelle Umgebung im Ordner `.venv` und installiert alle benötigten Pakete gemäß `pyproject.toml`.

## Daten herunterladen

Das Skript `download_from_kagglehub.py` ausführen, um den Datensatz herunterzuladen:

```bash
uv run Download_kagglehub.py
```

Dies lädt den Datensatz von Kaggle herunter und verschiebt die Dateien in den Ordner `data/`.

## Explorative Datenanalyse (EDA)

DAs Notebook `EDA.ipynb` in VS Code oder Jupyter öffnen und ausführen, um eine erste Analyse des Datensatzes durchzuführen.

## Feature Engineering, Modellierung und Visualisierung

- **`data_prep_for_model.py`**: Enthält die Funktionen zur Datenbereinigung, dem Feature Engineering und einer Preprocessing-Pipeline inklusive Column Transformer für classifier-Modelle mit verfügbarem Parameter class_weight='balanced'.
- **`final_model.py`**: Enthält die Funktion für das finale Modell und dessen Parameter sowie eine Funktion zum Erhalten der Feature_Importances dieses Modells.
- **`create_plots.py`**: Enthält Funktionen für ausgewählte Plots und die Konfigurationen dieser über matplotlib.
- **`SimpleBaselineModel.ipynb`**: Führt nur die Bereinigung der Daten aus, um ein erstes simples Modell zu erstellen und durchzuführen. Dessen Metriken können später mit dem finalen Modell verglichen werden. Dient auch als schnelles Basismodell für eine erste Vorhersage.
- **`Feature_Engineering.ipynb`**: Führt das Feature Engineering durch, um neue Features (Spalten) im bereinigten Datensatz zu erstellen.
- **`Hyperparameter-Tuning.ipynb`**: Nutzt optuna zum Hyperparameter-Tuning ausgewählter Modelle, die im finalen Modell Anwendung finden. Es kann eine Weile dauern (je nach Geräteleistung), um das Tuning durchzuführen.
- **`FinalBaseModel.ipynb`**: Testet ausgewählte Modelle, vergleicht diese und schließt mit dem final gewählten Modell ab, das die optimalen Hyperparameter enthält.
- **`FinalVisualizations.ipynb`**: Erstellt ausgewählte Plots bzw. Visualisierungen, die im plots/ Ordner abgespeichert werden.

## Testen (work in progress)

- **`test_pipelines.py`**: Enthält Tests für die Pipeline-Funktionen unter Verwendung von pytest.
- **Tests ausführen**:

  ```bash
  uv run -m pytest
  ```

  Dies führt die Tests aus und stellt sicher, dass die Skripte korrekt funktionieren.

## Verwendete Technologien und Bibliotheken

- **Python 3.13**: Programmiersprache.
- **uv**: Paketmanager für Python.
- **Jupyter Notebook**: Interaktive Entwicklungsumgebung.
- **Pandas**: Datenanalyse und -manipulation.
- **NumPy**: Numerische Berechnungen.
- **Matplotlib & Seaborn**: Datenvisualisierung.
- **Scikit-Learn**: Maschinelles Lernen.
- **Statsmodels**: Statistische Modellierung.
- **kagglehub**: Vereinfachtes Herunterladen von Kaggle-Datensätzen.
- **tqdm**: Fortschrittsbalken für Schleifen.
- **Optuna**: Hyperparameter-Optimierung.
- **Plotly**: Interaktive Visualisierungen.
- **pytest**: Framework zum Testen von Python-Code.
- **dotenv**: Lädt Umgebungsvariablen aus einer .env Datei, um sensible Daten wie API.Schlüssel & Co. sicher zu verwalten.
- **spotipy**: API-Helper für die Verbindung und das Abrufen von Daten der Spotify-API. Wurde im Projekt nicht mehr weiter genutzt, da Spotify mittlerweile nicht mehr ermöglicht, wichtige Audio Features durch die API zu erhalten (ohne eine gesonderte Erlaubnis).

## Fazit

Die Ergebnisse des finalen Modells (RandomForestClassifier) sind letztendlich mittelmäßig ausgefallen. 
Dies ist nicht verwunderlich, da neben den von Spotify zugänglichen Metadaten (wenige) sowie die Audiodaten (Audio features) noch viele weitere Faktoren die Beliebtheit eines Songs beeinflussen, die hier nicht weiter untersucht werden konnten (fehlende Daten, kein Zugriff, Schwierigkeit, überhaupt an solche Daten zu kommen).

Darunter fällt unter anderem:
- Saison des Song-Release (und passend zur Saison die Tonalität)
- Influencer, die den Song-Release "pushen" oder eine Community dazu aufgebaut haben, bis der Release stattgefunden hat (sowie auch sonstig Marketing-Maßnahmen, ein einzelnes TikTok-Video kann hier schon viel ausmachen)
- Skandale der Musiker hinter dem Song (oder auch das Fehlen von Skandalen)
- uvm.

Trotzdem kann das finale Modell auf gewisse Aspekte (gerade bei den Audiodaten) für eine mögliche Beeinflussung der Beliebtheit eines Songs in die eine oder andere Richtung aufzeigen. Es benötigt jedoch ein komplexeres Modell mit weiteren Features, die oben genannt wurden, um aussagekräftige Vorhersagen machen zu können. 