from dataclasses import dataclass, field
import pandas as pd
# Nom de fuseau IANA. pandas ne connait pas "heure de Paris" : il lui faut
# la cle de la base tzdata, qui embarque l'historique complet des regles de
# changement d'heure. C'est cette base qui sait que le 27/10/2024 dure 25 h.
TZ_LOCALE = "Europe/Paris"

# Constantes de tracabilite. Des chaines nues (au lieu d'un Enum) parce
# qu'elles finissent telles quelles dans une colonne de DataFrame et dans
# le CSV exporte : pas de conversion a faire a l'ecriture ni a la relecture.
MESURE = "MESURE"
MESURE_PARTIELLE = "MESURE_PARTIELLE"
INTERP_COURTE = "INTERP_COURTE"
PROFIL_JOUR_TYPE = "PROFIL_JOUR_TYPE"
ANNEE_N1 = "ANNEE_N-1"
ZERO_FORCE = "ZERO_FORCE"

ORIGINES_REELLES = (MESURE, MESURE_PARTIELLE)

COLONNES = {
    "prm": "Identifiant PRM",
    "debut_demande": "Date de début",
    "fin_demande": "Date de fin",
    "grandeur_physique": "Grandeur physique",
    "grandeur_metier": "Grandeur métier",
    "etape": "Etape métier",
    "unite": "Unité",
    "horodate": "Horodate",
    "valeur": "Valeur",
    "nature": "Nature",
    "pas": "Pas",
    "vraisemblance": "Indice de vraisemblance",
    "etat": "Etat complémentaire",
}

PREFERENCE_ETAPE = {"CORRIGE": 3, "COMPLETE": 2, "BRUT": 1}

@dataclass
class ReglesComblement:
    """Seuils de la hierarchie de comblement.

    Le decorateur @dataclass genere automatiquement __init__ et __repr__.
    On peut donc ecrire ReglesComblement(duree_max_interpolation="6h") sans
    ecrire le constructeur, et modifier un seul seuil sans toucher au code.
    """

    duree_max_interpolation: pd.Timedelta = pd.Timedelta("3h")
    duree_max_jour_type: pd.Timedelta = pd.Timedelta("3D")
    n_jours_reference: int = 4
    decalage_annuel: pd.Timedelta = pd.Timedelta("364D")  # 52 semaines
    recalage_n1: bool = True
    fenetre_recalage: pd.Timedelta = pd.Timedelta("21D")
    taux_reel_minimum: float = 0.80
    # field(default_factory=list) est OBLIGATOIRE pour une valeur par defaut
    # mutable. Ecrire `journal: list = []` partagerait la MEME liste entre
    # toutes les instances : le journal du deuxieme PDL contiendrait celui du
    # premier. default_factory appelle list() a chaque instanciation.
    # C'est le piege classique des arguments par defaut mutables en Python,
    # que dataclass detecte et refuse.
    journal: list = field(default_factory=list)