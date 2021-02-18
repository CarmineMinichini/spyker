import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os

def analysis():
    # legge il path del file
    path = os.getcwd()

    # apre il file txt originale registrato
    with open(path + "/outputs.txt", 'r') as file:
        data = file.read().replace('\n', '')

    # apro i dataframe per la sentiment analysis
    pos_words = pd.read_csv(path + "/files/stemmed_words_positive.csv")
    neg_words = pd.read_csv(path + "/files/stemmed_words_negative.csv")

    # apro i dataframe per la meta analisi di nomi e luoghi
    voc_luoghi = pd.read_csv(path + "/files/vocabolario_luoghi.csv").drop("Unnamed: 0",1)
    voc_nomi = pd.read_csv(path + "/files/vocabolario_nomi.csv").drop("Unnamed: 0",1)

    from nltk.corpus import stopwords
    # import stopwords italiane
    stopwords= set(stopwords.words('italian'))
    stopwords

    # tokenizzo ( essenziale per tutto )
    tokens = word_tokenize(data,language="italian")
    print("------.....Meta analisi....--------")
    print("Parole rilevate: {}".format(len(tokens)))

    # rimuovo stopwords
    filtered_speech = [w for w in tokens if not w in stopwords]

    # parole filtrate
    print('Numero di parole filtrate: {}'.format(len(tokens)-len(filtered_speech)))
    # parole effettivamente rimaste
    print('Numero di parole rimaste: {}'.format(len(filtered_speech)))

    # lowercase tutti i token del filtered speech
    for i in range(len(filtered_speech)):
        filtered_speech[i] = filtered_speech[i].lower()

    # crea un dizionario {parola:conteggio}
    dictionary = Counter(filtered_speech)

    # ritorna un dataframe con le parole e il count
    df = pd.DataFrame.from_dict(dictionary, orient='index').reset_index()
    df = df.rename(columns={'index':'word', 0:'counts'})

    # ordino in base al conteggio il dataframe
    sorted_df = df.sort_values(['counts'],ascending=False)
    sorted_df['counts'] = sorted_df['counts'].astype(int)
    sorted_df

    # esporto il dataframe delle parole ordinate ( non stemmatizzate )
    sorted_df.to_csv(path + "/meta_outputs/words_dataset.csv")

    # creo lista per parole da stemmatizzare
    lista = df['word'].to_list()

    from nltk.stem import SnowballStemmer
    stemmer_snowball = SnowballStemmer('italian')
    # lemmatizzo le parole ( che storo in una lista)

    stemmed_list = []
    for word in lista:
        stemmed_word = stemmer_snowball.stem(word)
        stemmed_list.append(stemmed_word)
        stemmed_list


    print("Parole stemmatizzate: \n {} \n ".format(stemmed_list))

    ############## APPROCCIO DI SENTIMENT analysis

    dictionary_stemmed = Counter(stemmed_list)

    # ritorna un dataframe con le parole stemmatizzate e il count
    df_stemmed = pd.DataFrame.from_dict(dictionary_stemmed, orient='index').reset_index()
    df_stemmed = df_stemmed.rename(columns={'index':'word', 0:'counts'})

    # dataframe ordinato delle parole stemmatizate
    sorted_df_stemmed = df_stemmed.sort_values(['counts'],ascending=False)
    sorted_df_stemmed['counts'] = sorted_df_stemmed['counts'].astype(int)
    sorted_df_stemmed.to_csv(path + "/meta_outputs/stemmed_words_dataset.csv")


    # SENTIMENT ANALYSIS

    # creo il dataframe per la sentiment analysis
    df_sentiment = sorted_df_stemmed.copy()

    # positive score
    df_sentiment['positive_score'] = np.where(df_sentiment['word'].isin(pos_words['positive']),1,0)
    # negative score
    df_sentiment['negative_score'] = np.where(df_sentiment['word'].isin(neg_words['positive']),-1,0)

    print("-------------....SENTIMENT ANALYSIS....--------")
    print("Parole positive riscontrate: \n\n {} " .format(df_sentiment[df_sentiment['positive_score']== 1]['word'].to_list()))
    print("___")
    print("Parole negative riscontrate: \n\n {} ".format(df_sentiment[df_sentiment['negative_score']== - 1]['word'].to_list()))


    # data_frame con le parole positive e negative
    words = pd.DataFrame.from_dict({'positive' :df_sentiment[df_sentiment['positive_score']== 1]['word'].to_list(), 'negative' : df_sentiment[df_sentiment['negative_score']== - 1]['word'].to_list()}, orient='index').T


    # PERCENTUALI POSITIVE E NEGATIVE
    D = {u'Positive':df_sentiment['positive_score'].sum()/len(df_sentiment),u'Negative': abs(df_sentiment['negative_score'].sum())/len(df_sentiment)}

    print(""*4)
    print("__________________ Sentimento complessivo: ____________")
    # differenza positivo - negativo
    scarto = (D['Positive'] - D['Negative'])*100


    print("Percentuale parole positive sul totale: {} %".format(round(D['Positive']*100,2)))
    print("Percentuale parole negative sul totale: {} %".format(round(D['Negative']*100,2)))
    print("")

    if scarto < 0 :
        print("Sembra che sia al {} % negativa ".format(round(abs(scarto),2)))
        risultato = "negativo"
    else:
        print("Sembra sia del {} % positiva ".format(round(scarto),2))
        risultato = "positivo"

    # creo sizes and labels per la PIE
    labels = []
    sizes = []

    for x, y in D.items():
        labels.append(x)
        sizes.append(y)

    ########################## meta analisi per nomi propri e luoghi

    # copio il dataframe delle parole non lemmatizzate
    meta_df = sorted_df.copy()

    # identifico i nomi
    identified_names = voc_nomi[voc_nomi['nome'].isin(meta_df['word'])]
    # identifico i luoghi
    identified_places = voc_luoghi[voc_luoghi['luogo'].isin(meta_df['word'])]

    ########################################DASHBOARD

    fig, axs = plt.subplots(1,4,figsize=(10,5))
    # titolo della dashboard
    fig.suptitle('Analizzate {} parole lemmatizzate su {} totali.\n Il bilancio è {} del {} %'.format(len(df_stemmed),len(tokens),risultato,round(abs(scarto),2)))

    # lemmatized count words
    sns.barplot(x=sorted_df_stemmed.counts,y="word",data = sorted_df_stemmed[:10],ax=axs[0],palette="viridis", edgecolor='black',linewidth=2)
    axs[0].set_title("Parole più frequenti",fontsize=8)

    axs[1].pie(sizes,autopct='%1.0f%%',labels=labels, pctdistance=1.2,labeldistance=1.5,colors=['slategray','firebrick'],normalize=False,wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'solid', 'antialiased': True})
    #axs[1].legend(labels,loc="lower right")
    #axs[1].set_title("Sentiment Analysis")

    if len(words) > 0:
        axs[2].table(cellText=words.values,colLabels=words.columns,loc="center")
        axs[2].axis('tight')
        axs[2].axis('off')
    else:
        axs[2].text(0.2,0.5,"Nessuna parola negativa \n Nessuna parola positiva")

        axs[2].axis('off')


    if len(identified_names) > 0:
        axs[3].table(cellText=identified_names.values,colLabels=identified_names.columns,loc="center")
        axs[3].axis('tight')
        axs[3].axis('off')
    else:
        axs[3].text(0.03,0.5,"Nessun nome rilevato")
        axs[3].axis('off')

    plt.tight_layout()
    sns.despine()
    plt.show()

    return
