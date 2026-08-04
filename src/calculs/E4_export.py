from __future__ import annotations
import pandas as pd
from calculs.constantes import ORIGINES_REELLES, TZ_LOCALE
from calculs.constantes import ReglesComblement

# ---------------------------------------------------------------------------
# 4. Journal qualite et export
# ---------------------------------------------------------------------------


def bilan_qualite(res: pd.DataFrame) -> dict:
    """Synthese par PDL : part de donnees reelles, energie, alertes."""
    pas = pd.Timedelta(res.attrs.get("pas_cible", "1h"))
    h = pas / pd.Timedelta("1h")
    # .value_counts() compte les occurrences de chaque valeur distincte.
    # normalize=True renvoie des proportions au lieu d'effectifs.
    # .to_dict() sort de pandas pour un resultat serialisable (JSON, log).
    parts = res["origine"].value_counts(normalize=True).to_dict()
    reel = sum(parts.get(o, 0.0) for o in ORIGINES_REELLES)
    # .groupby("colonne")["autre"].sum() : le pattern d'agregation de base.
    # Le resultat est une Series indexee par les valeurs de la colonne de
    # regroupement. La multiplication s'applique ensuite element par element.
    energie = res.groupby("origine")["puissance_W"].sum() * h / 1000
    return {
        "prm": res.attrs.get("prm"),
        "grandeur_metier": res.attrs.get("grandeur_metier"),
        "n_pas": len(res),
        "taux_reel": reel,
        "repartition_origines": parts,
        # float() explicite : .sum() renvoie un np.float64, qui n'est pas
        # serialisable en JSON par le module standard.
        "energie_totale_kWh": float(res["puissance_W"].sum() * h / 1000),
        "energie_par_origine_kWh": energie.to_dict(),
        "exploitable": reel >= ReglesComblement().taux_reel_minimum,
    }


def exporter(res: pd.DataFrame, chemin: str) -> None:
    """Export en heure locale, avec la colonne d'origine pour tracabilite."""
    sortie = res.copy()
    # On repasse en heure locale UNIQUEMENT ici : c'est ce que l'utilisateur
    # et les outils metier attendent. Tout le calcul s'est fait en UTC.
    sortie.index = sortie.index.tz_convert(TZ_LOCALE)
    sortie.index.name = "horodate_locale"
    # .to_csv ecrit l'index comme premiere colonne par defaut (index=False
    # pour l'omettre — surtout pas ici, l'index EST l'horodate).
    # sep=";" et decimal="," pour un fichier directement ouvrable dans un
    # Excel en configuration francaise.
    # float_format limite le nombre de decimales et donc la taille du fichier.
    sortie.to_csv(chemin, sep=";", decimal=",", float_format="%.3f")

