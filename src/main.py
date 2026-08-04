
from calculs.selection_dossier import iterer_fichiers_csv, selectionner_dossier, obtenir_dossier_sortie


from calculs.E1_lecture import lire_courbe
from calculs.E2_agregation_pas import agreger
from calculs.E3_comblement_lacunes import combler
from calculs.E4_export import bilan_qualite, exporter

import sys
import warnings
import  pandas as pd
import glob
from tkinter import Tk,filedialog
from pathlib import Path



def main():



    dossier_sortie = obtenir_dossier_sortie()
    dossier_fichiers_sources = selectionner_dossier()

    if not dossier_fichiers_sources:
        print("Aucun dossier sélectionné. Fin du programme.")
        return





    courbes, bilans = {}, []
    for chemin in iterer_fichiers_csv(dossier_fichiers_sources):
        brut  = lire_courbe(chemin)   # lecture + normalisation UTC
        res   = agreger(brut, cible="1h")           # passage au pas horaire /!\ pas de chemin dans l'E2 mais présent dans commente
        plein = combler(res)                          # reconstitution des lacunes
        nom_fichier_sortie = dossier_sortie / f"sortie_{chemin.stem}.csv"
        exporter(plein, nom_fichier_sortie)               # export en heure locale

    bilans.append(bilan_qualite(plein))

    print("\n=== Bilan global ===")
    print(bilans)



if __name__ == "__main__":
    main()