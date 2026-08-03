
from tkinter import Tk,filedialog
from pathlib import Path
import sys

def selectionner_dossier() -> Path | None:
    """Ouvre une fenêtre pour sélectionner le dossier des fichiers Excel."""
    fenetre = Tk()
    fenetre.withdraw()
    fenetre.attributes("-topmost", True)

    dossier = filedialog.askdirectory(
        title="Sélectionnez le dossier contenant les fichiers Excel"
    )

    fenetre.destroy()

    return Path(dossier) if dossier else None

def iterer_fichiers_csv(dossier: Path):
    """Renvoie successivement chaque fichier CSV du dossier."""
    for chemin in dossier.glob("*.csv"):
        # Ignore les fichiers temporaires créés par Excel
        if not chemin.name.startswith("~$"):
            yield chemin


def obtenir_dossier_sortie() -> Path:
    """Crée et renvoie le dossier 'sortie' à côté du programme."""

    if getattr(sys, "frozen", False):
        # Programme lancé depuis un exécutable PyInstaller
        dossier_programme = Path(sys.executable).resolve().parent
    else:
        # Programme lancé depuis src/main.py
        dossier_programme = Path(__file__).resolve().parent.parent

    dossier_sortie = dossier_programme / "sortie"
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    return dossier_sortie