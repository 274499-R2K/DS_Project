DS_Project

## Dati (raw)
1) Incolla i dati raw in:
`DS_Project\Data\raw`

## Esecuzione (ordine ESATTO)
Esegui gli script **in questo ordine** (da root del progetto):

1. `python src/preprocess/extract.py`
2. `python src/preprocess/merge.py`
3. `python src/preprocess/trim.py`
4. `python src/modeling/smooth.py`
5. `python src/modeling/window.py`

## Visualizzazione
Con `report/plot_data.py` puoi visualizzare il confronto tra dati **registrati (trimmed)** e **smoothed**, inserendo **il nome del file** come input.

## Note dataset
- Al momento ci sono **solo 12 recordings per classe**.
- Mappatura classi:
  - `wlk` = walking  
  - `lng` = long turns  
  - `srt` = short turns  
  - `sgt` = going straight
