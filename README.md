# spyker
Un tool per il riconoscimento vocale focalizzato sull'analisi semantica e sentimentale per la lingua italiana,

Nella cartella meta_outputs produce:

1. words_dataset.csv contenente le parole (escluse le stopwords) riordinate per frequenza
2. stemmed_words_dataset.csv contenente le parole stemmatizzate riordinate per frequenza

Nella cartella principale produce:
outputs.txt contenente il testo risconosciuto dall'assistente vocale

![Screenshot](mario_draghi.png)


## Utilizzo
Dalla cartella:
```
pip install -r requirements.txt
```

dal terminale:
```
python3 spyker.py
```



