import speech_recognition as sr
from analysis import *
# inizializza tokenizer
recognizer_istance = sr.Recognizer()

while True:
    try:
        print(".......... ..... .........."*2)
        print("______ Benvenuto in Spyder ______")
        print(".......... ..... .........."*2)
        print("")
        answer  = input ("Vuoi caricare o registrare un audio? \n")
        if answer.lower()== "caricare":
            file_path = input("Inserisci il percorso completo del file: \n")
            sorgente = sr.AudioFile(file_path)
            risposta = "File caricato!"
            break
        elif answer.lower()=="registrare":
            sorgente = sr.Microphone()
            risposta = "Dimmi pure..."
            break
        else:
            print("......")
            print("Scusa non mi è chiaro quello che hai scritto riproviamo \n ")
    except ValueError:
        print ("Scusa non mi è chiaro quello che hai scritto riproviamo \n ")

with sorgente as source:
    # elimina noise
    recognizer_istance.adjust_for_ambient_noise(source)
    # pause pause_threshold
    recognizer_istance.pause_threshold = 3.0

    print(".....  ....."*3)
    print(".....  ....."*3)
    print(".....  ....."*3)
    print("")
    print(risposta)
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
