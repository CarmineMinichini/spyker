import speech_recognition as sr
from analysis import *
# inizializza tokenizer
recognizer_istance = sr.Recognizer()

#tokenizer = nltk.data.load("/Users/carmineminichini/nltk_data/tokenizers/punkt/italian.pickle")

# usi il Microphone come source
with sr.Microphone() as source:
    # elimina noise
    recognizer_istance.adjust_for_ambient_noise(source)
    # pause pause_threshold
    recognizer_istance.pause_threshold = 3.0

    print(".....  ....."*3)
    print(".....  ....."*3)
    print(".....  ....."*3)
    print("")
    print("Dimmi pure..",end="\n")
    # processing audio
    audio = recognizer_istance.listen(source)
    print("Ho capito..",end="\r")
    print("Elaboro...",end="\r")



try:
    # recognizer ita
    text = recognizer_istance.recognize_google(audio,language="it-IT")
    result = '{}'.format(text)
    with open("outputs.txt","w") as f:
            f.write(result)
    # riconosce il testo
    print(" ")
    print("__________________ Testo riconosciuto: __________________")
    print("")
    print(text)
    print(""*2)

    analysis()
    # Exception
except Exception as e:
    print(e)
