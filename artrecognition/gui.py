'''gui.py: ist die graphische Komponente'''

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import requests
import sqlite3
import os
import finder


class WrappingLabel(Label):
    '''Label, dass Zeilenumbrüche automatisch an den vorhandenen Platz anpasst'''
    # https://www.reddit.com/r/learnpython/comments/6dndqz/how_would_you_make_text_that_automatically_wraps/
    def __init__(self, master=None, **kwargs):
        Label.__init__(self, master, **kwargs)
        self.bind('<Configure>', lambda e: self.config(wraplength=self.winfo_width()))


# Funktionen
def erkennen(pfad):
    
    # Oberfläche säubern
    label_bild1.config(image = "")

    # Bilderkennung ausführen
    id = finder.main(pfad)
    if id != False:
        
        # Verbindung mit Datenbank herstellen
        con = sqlite3.connect('db.db')
        c = con.cursor()

        # Bild URL aus Datenbank lesen
        sql = "SELECT link FROM kunstwerke WHERE id=%s"
        c.execute(sql % id)
        url = c.fetchone()

        # Bild online öffnen
        img_programm = Image.open(requests.get(url[0], stream=True).raw)
        img_programm.thumbnail((200, 200))
        img_programm = ImageTk.PhotoImage(image=img_programm)

        # Bild anzeigen
        label_bild1.config(image=img_programm)
        label_bild1.image = img_programm  

        # Alle Kategorien werden aufgelistet
        text = ""
        kategorien = ("Titel: ", "Künstler: ", "Entstehungsjahr: ", "Entstehungsort: ", "Material: ", "Genre: ", "Stil: ", "Preis: ", "Galerie: ")

        # Alle Informationen zum Bild werden aus der Datenbank gelesen
        c.execute("SELECT titel, name, jahr, ort, material, genre, stil, preis, galerie_name FROM kunstwerke WHERE id=%s" % id)
        informationen = c.fetchone()
        
        # Alle Informationen zum Künstler werden aus der Datenbank gelesen
        c.execute("SELECT geburtsdatum, sterbedatum, wikipedia FROM kuenstler WHERE name='%s'" % informationen[1])
        # c.execute("SELECT * FROM kuenstler WHERE name = 'Leonardo da Vinci'")
        informationen += c.fetchone()

        # Text wird mit den Informationen kombiniert
        for i in range (9):
            if informationen[i] != None:
                
                # Geburts und sterbedatum werden hinter den Künstlernamen gesetzt
                if i == 1:
                    text += kategorien[1] + informationen[1] + " (" + informationen[9][-4:] + " bis " + informationen[10][-4:] + ")" + "\n"
                    continue

                text += kategorien[i] + str(informationen[i]) + "\n"

        # wikipedia artikel Link wird angefügt
        text += informationen[11]

        # Text wird zurechtgerückt
        label_erklaerung.config(text=text, justify=LEFT)
        con.close()
        
    else:
        label_erklaerung.configure(text="Kein Ergebnis.")

def datei_oeffnen():
    name = askopenfilename()
    
    # Benutzerbild anzeigen
    pillow_img = Image.open(name)
    pillow_img.thumbnail((200, 200))
    global img 
    img = ImageTk.PhotoImage(image=pillow_img)
    label_bild_benutzer.config(image=img)
    label_bild_benutzer.image = img
    erkennen(name)

def ueber():
    ueber_fenster = Toplevel(fenster)
    ueber_fenster.title("Über uns")
    ueber_fenster.geometry('250x250')
    label_text = WrappingLabel(ueber_fenster, text=ueber_uns)
    label_text.pack(expand=TRUE)

def beispiel():
    # Das wäre das vom Benutzer eingegebene Bild:
    pillow_img = Image.open('eingabetest.jpg')
    pillow_img.thumbnail((200, 200))
    global img 
    img = ImageTk.PhotoImage(image=pillow_img)
    label_bild_benutzer.config(image=img)
    label_bild_benutzer.image = img
    # Das wäre das vom Programm gefundene Bild:
    erkennen('eingabetest.jpg')

def dokumentation():
    # Datei öffnen
    os.startfile('Dokumentation.pdf')

# Variablen
erklaerung = "Dieses Programm hilft dabei, den Namen eines Kunstwerks sowie weitere Informationen zu ermitteln. Die Vorraussetzungen sind: ein Bild des zu findenden Kunstwerks und dass es sich bei dem Künstler um Albrecht Dürer, Leonardo da Vinci oder Mary Cassat handelt. Zum Starten einfach auf den Button unten drücken oder unter Bild - Öffnen."
ueber_uns = "Dies ist das Informatik-Abschlussprojekt von Lauris Schleussner und Lea Patties. Wir sind derzeit Schüler der 10. Klasse des Wilhelm-Ostwald-Gymnasiums. Unsere Motivation ist es, Menschen zu helfen, die den Namen eines Kunstwerks vergessen haben und gleichzeitig eine programmiertechnische Herausforderung zu meistern. Dieses Programm ist ein exemplarischer Prototyp. Für weitere Informationen kann die Dokumentation eingesehen werden."

# Fenster
fenster = Tk()
fenster.title("Kunstwerkerkennungshilfe")
fenster.geometry('450x450')
fenster.rowconfigure([0, 1, 2, 3, 4, 5], weight=1, minsize=20)
fenster.columnconfigure([0, 1, 2, 3], weight=1, minsize=20)
fenster.option_add('*tearOff', False)

# Menü
menubar = Menu(fenster)
fenster.config(menu=menubar)
menu_bild = Menu(menubar)
menubar.add_cascade(menu=menu_bild, label='Bild')
menu_bild.add_command(label='Öffnen', command=datei_oeffnen)
menu_bild.add_command(label='Beispiel', command=beispiel)
menu_bild.add_separator()
menu_bild.add_command(label='Schließen', command=quit)
menu_hilfe = Menu(menubar)
menubar.add_cascade(menu=menu_hilfe, label='Hilfe')
menu_hilfe.add_command(label='Über', command=ueber)
menu_hilfe.add_command(label='Dokumentation', command=dokumentation)

# weitere Komponenten
label_ueberschrift = Label(fenster, text="Kunstwerk-Erkennung", font=12)
label_ueberschrift.grid(row=0, column=0, columnspan=4)

label_erklaerung = WrappingLabel(fenster, text=erklaerung)
label_erklaerung.grid(row=3, column=0, sticky='nsew', rowspan=2, columnspan=4)

label_bild1 = Label(fenster)
label_bild1.grid(row=1, column=2, sticky='nsew', columnspan=2)

label_name1 = Label(fenster, text="gefundenes Bild")
label_name1.grid(row=2, column=2, sticky='nsew', columnspan=2)

label_bild_benutzer = Label(fenster)
label_bild_benutzer.grid(row=1, column=0, sticky='nsew', columnspan=2)

label_benutzer = Label(fenster, text="Benutzer-Bild")
label_benutzer.grid(row=2, column=0, sticky='nsew', columnspan=2)

button_start = Button(fenster, text="Starten", command=datei_oeffnen)
button_start.grid(row=5, column=0, columnspan=4)

fenster.mainloop()