import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def analysis():
    # legge il percorso
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
    print("__________________ Analisi della pulizia __________________")
    print("")
    print("Numero di parole rilevate: {}".format(len(tokens)))

    # lowercase di tutti i tokens solo PER I BIGRAMMI
    tokens_bg = []
    for i in range(len(tokens)):
        tokens_bg.append(tokens[i].lower())

    # rimuovo stopwords
    filtered_speech = [w for w in tokens if not w in stopwords]
    filtered_speech

    # parole filtrate
    print('Numero di parole filtrate : {}'.format(len(tokens)-len(filtered_speech)))
    # parole effettivamente rimaste

    # lowercase tutti i token del filtered speech
    for i in range(len(filtered_speech)):
        filtered_speech[i] = filtered_speech[i].lower()


    # crea un dizionario {parola:conteggio}
    dictionary = Counter(filtered_speech)

    # ritorna un dataframe con le parole e il count
    df = pd.DataFrame.from_dict(dictionary, orient='index').reset_index()
    df = df.rename(columns={'index':'word', 0:'counts'})

    # riordino il dataframe della parole con conteggio
    sorted_df = df.sort_values(['counts'],ascending=False)
    sorted_df['counts'] = sorted_df['counts'].astype(int)
    # lo esporto
    sorted_df.to_csv(path + "/meta_outputs/words_dataset.csv")

    # creo lista per parole  da stemmatizzare
    lista = df['word'].to_list()

    # importo lo stemmer in italiano
    from nltk.stem import SnowballStemmer
    stemmer_snowball = SnowballStemmer('italian')

    # lemmatizzo le parole ( che storo in una lista)
    stemmed_list = []
    for word in lista:
        stemmed_word = stemmer_snowball.stem(word)
        stemmed_list.append(stemmed_word)
        stemmed_list

    ############## APPROCCIO DI SENTIMENT analysis

    dictionary_stemmed = Counter(stemmed_list)

    # ritorna un dataframe con le parole stemmatizzate e il count
    df_stemmed = pd.DataFrame.from_dict(dictionary_stemmed, orient='index').reset_index()
    df_stemmed = df_stemmed.rename(columns={'index':'word', 0:'counts'})

    # numero di parole che andrà ad analizzare
    print('Numero di parole uniche rimaste: {}'.format(len(df_stemmed)))
    # percentuale parole uniche che andrà ad analizzare
    on_tot = round((len(df_stemmed)/len(tokens))*100,2)
    print("Percentuale di parole uniche che andrà ad analizzare: {} % ".format(on_tot))
    # printo le parole stemmatizzate
    print("Parole stemmatizzate: \n {} \n ".format(stemmed_list))

    # dataframe ordinato delle parole stemmatizate
    sorted_df_stemmed = df_stemmed.sort_values(by="counts",ascending=False)
    # lo esporto
    sorted_df_stemmed.to_csv(path + "/meta_outputs/stemmed_words_dataset.csv")

    ##########################
    ###### SENTIMENT Analysis
    ##########################

    # creo il dataset per la sentiment analysis
    df_sentiment = sorted_df_stemmed.copy()

    df_sentiment['positive_score'] = np.where(df_sentiment['word'].isin(pos_words['positive']),1,0)
    df_sentiment['negative_score'] = np.where(df_sentiment['word'].isin(neg_words['positive']),-1,0)

    # printo le parole positive e le parole negative
    print("__________________ SENTIMENT ANALYSIS __________________")
    print("")
    print("Parole positive riscontrate: \n\n {} " .format(df_sentiment[df_sentiment['positive_score']== 1]['word'].to_list()))
    print("___")
    print("Parole negative riscontrate: \n\n {} ".format(df_sentiment[df_sentiment['negative_score']== - 1]['word'].to_list()))


    # data_frame con le parole positive e negative
    words = pd.DataFrame.from_dict({'positive' :df_sentiment[df_sentiment['positive_score']== 1]['word'].to_list(), 'negative' : df_sentiment[df_sentiment['negative_score']== - 1]['word'].to_list()}, orient='index').T

    # PERCENTUALI POSITIVE E NEGATIVE
    D = {u'Positive':df_sentiment['positive_score'].sum()/len(df_sentiment),u'Negative': abs(df_sentiment['negative_score'].sum())/len(df_sentiment)}

    print("")
    print("__________________ Sentimento complessivo: ____________")
    print("")
    # differenza positivo - negativo
    scarto = (D['Positive'] - D['Negative'])*100


    print("Percentuale parole positive sul totale: {} %".format(round(D['Positive']*100,2)))
    print("Percentuale parole negative sul totale: {} %".format(round(D['Negative']*100,2)))
    print("")

    if scarto < 0 :
        print("Il bilancio è negativo del {} % ".format(round(abs(scarto),2)))
        risultato = "negativo"
    elif scarto==0:
        print("Il bilancio è neutrale : {} % ".format(round(abs(scarto),2)))
        risultato = "neutrale"
    else:
        print("Il bilancio è positivo del {} % ".format(round(scarto),2))
        risultato = "positivo"

    # Generare sizes and labels per la PIE

    # Data to plot
    labels = []
    sizes = []

    for x, y in D.items():
        labels.append(x)
        sizes.append(y)

    ######################## PLOTS

    # copio il dataframe delle parole non lemmatizzate
    meta_df = sorted_df.copy()
    # identifico i nomi
    identified_names = voc_nomi[voc_nomi['nome'].isin(meta_df['word'])]
    # idenfitico i posti
    identified_places = voc_luoghi[voc_luoghi['luogo'].isin(meta_df['word'])]

    print("_____________ Meta Analisi : _____________")

    print("....... Nomi Propri .....")

    for i in np.arange(0,len(identified_names)):
        print("Nome: {} \n di categoria : {}".format(identified_names['nome'].iloc[i],identified_names['categoria'].iloc[i]))
    print("")
    print("....... Luoghi e località .....")

    for i in np.arange(0,len(identified_places)):
        print("Nome del luogo: {} \n località : {},{}".format(identified_places['luogo'].iloc[i],identified_places['categoria'].iloc[i],identified_places['categoriaEstero'].iloc[i]))

    print("")
    print("_____________ Bigrammi/Trigrammi : _____________")
    print("")

    ##########
    # Bigrammi
    ##########
    bigrm = list(nltk.bigrams(tokens_bg))
    # pulisco dalla lista
    bigrm = list(map(lambda x: ' '.join(x), bigrm))
    print("Bigrammi prodotti : {}".format(len(bigrm)))

    # conto i bigrammi simili
    bg_dict= Counter(bigrm)
    bg_dict = dict(bg_dict)
    print("Bigrammi unici: {}".format(len(bg_dict)))

    wc_bg = WordCloud(font_path = path+"/files/mermaid.ttf",background_color='white',max_words=10, contour_width=5, contour_color='black',
                          width=800,height=400).generate_from_frequencies(bg_dict)


    ##########
    # Trigrammi
    ##########
    trigrm = list(nltk.trigrams(tokens_bg))
    # pulisco dalla lista
    trigrm = list(map(lambda x: ' '.join(x), trigrm))
    print(" ")
    print("Trigrammi prodotti : {}".format(len(trigrm)))

    # conto i trigrammi simili
    tr_dict= Counter(trigrm)
    tr_dict = dict(tr_dict)
    print("Trigrammi unici: {}".format(len(tr_dict)))

    wc_tr = WordCloud(font_path = path+"/files/mermaid.ttf",background_color='white',max_words= 10, contour_width=5, contour_color='black',
                          width=800,height=400).generate_from_frequencies(tr_dict)

    #####################################
    # Bigrams and trigrams dataframe
    bg_df = pd.DataFrame.from_dict(bg_dict,orient="index").reset_index().rename(columns={'index':"bigramma",0:"counts"})
    bg_df.sort_values(by="counts",ascending=False,inplace=True)

    tr_df = pd.DataFrame.from_dict(tr_dict,orient="index").reset_index().rename(columns={'index':"trigramma",0:"counts"})
    tr_df.sort_values(by="counts",ascending=False,inplace=True)

    bg_df.to_csv(path + "/meta_outputs/bigrams.csv")
    tr_df.to_csv(path + "/meta_outputs/trigrams.csv")

    ################

    print("_____________ Primi 10 Bigrammi comuni : _____________")

    if len(bg_df)<10:
        rng_bg = len(bg_df)
    else:
        rng_bg = 10

    if len(tr_df)<10:
        rng_tr = len(tr_df)
    else:
        rng_tr = 10

    for i in range(0,rng_bg):
        print(" \n Bigramma: ( {} ) \n Numero di occorrenze: {} ".format(bg_df['bigramma'].iloc[i],bg_df['counts'].iloc[i]))

    print("_____________ Primi 10 Trigrammi comuni : _____________")

    for i in range(0,rng_tr):
        print(" \n Trigramma: ( {} ) \n Numero di occorrenze: {} ".format(tr_df['trigramma'].iloc[i],tr_df['counts'].iloc[i]))

    #################################
    ######
    #####
    #################################
    print("")
    print("Vedi Plots..")
    fig, axs = plt.subplots(2,3,figsize=(12,7))
    # titolo della dashboard
    fig.suptitle('Analizzate {} parole uniche lemmatizzate({} %) su {} totali.\n Il bilancio è {} del {} %'.format(len(df_stemmed),on_tot,len(tokens),risultato,round(abs(scarto),2)))

    # lemmatized count words
    sns.barplot(x=sorted_df_stemmed.counts,y="word",data = sorted_df_stemmed[:10],ax=axs[0,0],palette="viridis", edgecolor='black',linewidth=2)
    axs[0,0].set_title("Parole più frequenti",fontsize=8)

    axs[0,1].pie(sizes,autopct='%1.0f%%',labels=labels, pctdistance=1.2,labeldistance=1.5,colors=['slategray','firebrick'],normalize=False,wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'solid', 'antialiased': True})
    #axs[1].legend(labels,loc="lower right")
    #axs[1].set_title("Sentiment Analysis")

    if len(words) > 0:
        axs[0,2].table(cellText=words.values,colLabels=words.columns,loc="center")
        axs[0,2].axis('tight')
        axs[0,2].axis('off')
    else:
        axs[0,2].text(0.2,0.5,"Nessuna parola negativa \n Nessuna parola positiva")
        axs[0,2].axis('off')


    if len(identified_names) > 0:
        axs[1,0].table(cellText=identified_names.values,colLabels=identified_names.columns,loc="center")
        axs[1,0].axis('tight')
        axs[1,0].axis('off')
    else:
        axs[1,0].text(0.03,0.5,"Nessun nome rilevato")
        axs[1,0].axis('off')

    axs[1,1].imshow(wc_bg, interpolation='bilinear')
    axs[1,1].axis("off")
    axs[1,1].set_title("Bigrammi frequenti",fontsize=7)

    axs[1,2].imshow(wc_tr, interpolation='bilinear')
    axs[1,2].axis("off")
    axs[1,2].set_title("Trigrammi frequenti",fontsize=7)

    plt.tight_layout()
    sns.despine()
    plt.show()

    return
