# PABLO: Predizione Architetturale Bayesiana per la Ricerca e l'Ottimizzazione di Reti Neurali nella Predizione di Serie Temporali

Questo repository contiene il codice sviluppato per la mia tesi di laurea triennale, incentrata sull'ottimizzazione Bayesiana delle architetture di reti neurali per la predizione di serie temporali. L'applicazione specifica riguarda la stima delle posizioni angolari della mano tramite segnali elettromiografici di superficie (sEMG).

## Struttura della Repository

La repository è organizzata nel seguente modo:

### 📂 Ricerche
Sono presenti quattro cartelle corrispondenti alle diverse ricerche effettuate:
- `L1Full`
- `L1Half`
- `L2Full`
- `L2Half`

Ciascuna di queste cartelle contiene:
- Un file Python relativo alla ricerca.
- Una cartella `best_models/` che include:
  - I file `.keras` relativi ai 10 migliori modelli ottenuti dalla ricerca e al modello benchmark.
  - I file `.json` contenenti i dati relativi all'allenamento dei 10 migliori modelli e del modello benchmark.

### 📂 Datasets
Questa cartella contiene gli script Python utilizzati per la creazione di due dataset:
- **FullDataset**
- **HalfDataset**

I dataset sono generati a partire dal dataset ufficiale disponibile su Kaggle:  
🔗 [BCI Initiative ALVI HCI Challenge](https://www.kaggle.com/c/bci-initiative-alvi-hci-challenge)

### 📂 Notebook
Sono presenti due notebook Python:
- `metrics.ipynb`: Calcola le metriche (MAE, Std MAE, MSE, Std MSE) sul test dataset per i migliori modelli ottenuti e il modello benchmark.
- `graphs.ipynb`: Genera i grafici presenti nella tesi, confrontando le prestazioni canale per canale del miglior modello ottenuto con il modello benchmark.
