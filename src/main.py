
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

    erreur_fichiers = []
    courbes, bilans = {}, []
    for chemin in iterer_fichiers_csv(dossier_fichiers_sources):

        #essai de réalisation de l'ensemble des étapes, s'il y a une erreur, on passe au fichier (PDL) suivant
        try :
            etape = f"Traitement du fichier {chemin.stem}"
            brut  = lire_courbe(chemin)   # lecture + normalisation UTC

            etape = f"Agregation du fichier {chemin.stem}"
            res   = agreger(brut, cible="1h")           # passage au pas horaire /!\ pas de chemin dans l'E2 mais présent dans commente

            etape = f"Comblement du fichier {chemin.stem}"
            plein = combler(res)  

            etape = f"Création du bilan pour {chemin.stem}"
            bilans.append(bilan_qualite(plein))

            etape = f"Export du fichier {chemin.stem}"
            nom_fichier_sortie = dossier_sortie / f"sortie_{chemin.stem}.csv"
            exporter(plein, nom_fichier_sortie)               # export en heure locale
            

        # en cas d'erreur on stocke le chemin du fichier et l'erreur dans une liste pour affichage à la fin
        except Exception as e:
           
            erreur_fichiers.append({"nom": chemin.stem, "erreur": str(e), "etape": etape})
            continue 



    print("\n=== Bilan global ===")
    print(bilans)


    if erreur_fichiers:
        print("\n=== Fichiers avec erreurs ===")
        for fichier in erreur_fichiers:
            print(f"- {fichier['nom']} ({fichier['etape']}): {fichier['erreur']}")

if __name__ == "__main__":
    main()