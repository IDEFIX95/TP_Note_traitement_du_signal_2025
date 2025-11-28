# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 12:01:03 2025

@author: romain
"""

import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time


#Fenêtre principale :
    
root = tk.Tk()
root.title("Traitement Audio")
root.geometry("800x600")


#Choisir le fichier audio à traiter :
def choisir_fichier():
    fichier = filedialog.askopenfilename(
        title="Choisir un fichier audio",
        filetypes=[("Fichiers audio", "*.wav *.mp3")]
    )
    if fichier:
        label_fichier.config(text=f"Fichier sélectionné : {fichier}")
        return fichier
    return None

#Affichage de la barre de progression lors du traitement du signal audio :
def lancer_traitement():
    fichier = choisir_fichier()
    if not fichier:
        return

    # Simuler un traitement avec une barre de progression
    progress_bar.start(10)  # Démarre la barre de progression
    for i in range(101):
        time.sleep(0.05)  # Simule un traitement
        progress_bar['value'] = i
        root.update_idletasks()  # Met à jour l'interface

    # À la fin du traitement, afficher le graphique
    afficher_graphique()

#Afficher le graphique de sortie avec Matplotlib :
def afficher_graphique():
    # Exemple de données
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Créer le graphique
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Résultat du traitement audio")

    # Intégrer le graphique dans Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)



#Bouton pour choisir le fichier :
bouton_choisir = tk.Button(root, text="Choisir un fichier audio", command=choisir_fichier)
bouton_choisir.pack(pady=20)

#Label pour afficher le fichier sélectionné :
label_fichier = tk.Label(root, text="Aucun fichier sélectionné")
label_fichier.pack()

#Bouton pour lancer le traitement :
bouton_lancer = tk.Button(root, text="Lancer le traitement", command=lancer_traitement)
bouton_lancer.pack(pady=20)

#Barre de progression :
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=20)

#Execution de l'appli
root.mainloop()

"""Partie pour le traitement du signal audio

def traiter_audio(fichier):
    # Appelle ton algorithme ici
    # Exemple : result = mon_algorithme(fichier)
    return result

"""



















