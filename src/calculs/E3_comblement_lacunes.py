from __future__ import annotations

import datetime as _dt
import numpy as np
import pandas as pd

from calculs.constantes import (
    INTERP_COURTE,
    PROFIL_JOUR_TYPE,
    ANNEE_N1,
    ZERO_FORCE,
    ORIGINES_REELLES,
    TZ_LOCALE,
    ReglesComblement,
        )

# ---------------------------------------------------------------------------
# 3. Comblement des lacunes
# ---------------------------------------------------------------------------





def _paques(annee: int) -> _dt.date:
    """Dimanche de Paques (algorithme de Butcher)."""
    # Aucune bibliotheque impliquee : arithmetique entiere pure.
    # divmod(a, b) renvoie le couple (quotient, reste) en une operation.
    a = annee % 19
    b, c = divmod(annee, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lu = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lu) // 451
    mois, jour = divmod(h + lu - 7 * m + 114, 31)
    return _dt.date(annee, mois, jour + 1)


def jours_feries(annee: int) -> set:
    """Jours feries francais (metropole) pour une annee."""
    # datetime.date (et non pd.Timestamp) : ce sont des dates calendaires
    # sans heure ni fuseau, ce qui evite toute question de changement d'heure.
    p = _paques(annee)
    return {
        _dt.date(annee, 1, 1),
        p + _dt.timedelta(days=1),    # lundi de Paques
        _dt.date(annee, 5, 1),
        _dt.date(annee, 5, 8),
        p + _dt.timedelta(days=39),   # Ascension
        p + _dt.timedelta(days=50),   # lundi de Pentecote
        _dt.date(annee, 7, 14),
        _dt.date(annee, 8, 15),
        _dt.date(annee, 11, 1),
        _dt.date(annee, 11, 11),
        _dt.date(annee, 12, 25),
    }


def type_de_jour(dates: pd.DatetimeIndex) -> pd.Series:
    """Classe chaque date en OUVRE / SAMEDI / DIMANCHE_FERIE."""
    feries = set()
    # .year sur un DatetimeIndex renvoie un Index d'entiers (une valeur par
    # ligne), pas un entier unique. D'ou .min() / .max().
    for a in range(dates.year.min(), dates.year.max() + 1):
        feries |= jours_feries(a)  # |= : union en place de deux ensembles, ici les jours feries de chaque annee. On pourrait aussi faire `feries = feries.union(jours_feries(a))`, mais c'est plus verbeux.

    # .date sur un DatetimeIndex : tableau de datetime.date (heure retiree).
    d = pd.Series(dates.date, index=dates)
    # .dayofweek : 0 = lundi ... 6 = dimanche. Convention ISO.
    jour = pd.Series(dates.dayofweek, index=dates)

    # On part de "OUVRE" partout, puis on ecrase par specificite croissante.
    t = pd.Series("OUVRE", index=dates)
    t[jour == 5] = "SAMEDI"
    # Le | est le "ou" vectorise. Parentheses obligatoires, meme raison
    # de priorite qu'avec &. .isin(ensemble) teste l'appartenance ligne a ligne.
    t[(jour == 6) | d.isin(feries)] = "DIMANCHE_FERIE"
    return t


def combler(df: pd.DataFrame, regles: ReglesComblement | None = None) -> pd.DataFrame:
    """Comble les lacunes selon une hierarchie dependante de leur duree."""
    # `a or b` renvoie b si a est falsy (None ici). Idiome courant pour les
    # valeurs par defaut mutables qu'on ne veut pas mettre dans la signature.
    regles = regles or ReglesComblement()
    res = df.copy()
    pas = pd.Timedelta(res.attrs.get("pas_cible", "1h"))
    tz_local = res.index.tz_convert(TZ_LOCALE)
    types = type_de_jour(tz_local)

    for debut, fin in _blocs_manquants(res["puissance_W"]):
        duree = fin - debut + pas
        # .loc[debut:fin] sur un DatetimeIndex trie : decoupage par tranche,
        # bornes INCLUSES des deux cotes (contrairement au slicing Python
        # habituel, ou la borne haute est exclue). C'est une particularite de
        # .loc qui surprend souvent.
        idx = res.loc[debut:fin].index

        # Comparaison directe de Timedelta : lisible, pas de conversion.
        if duree <= regles.duree_max_interpolation:
            comble = _interpoler(res["puissance_W"], idx)
            origine = INTERP_COURTE
        elif duree <= regles.duree_max_jour_type:
            comble = _profil_jour_type(res, idx, types, pas, regles)
            origine = PROFIL_JOUR_TYPE
        else:
            comble = _annee_precedente(res, idx, regles)
            origine = ANNEE_N1

        # Repli commun : si la methode choisie n'a pas pu aboutir.
        if comble is None or comble.isna().any():
            comble = pd.Series(0.0, index=idx)
            origine = ZERO_FORCE

        # .to_numpy() a droite : `comble` porte deja le bon index, mais
        # l'expliciter protege d'un realignement inattendu.
        res.loc[idx, "puissance_W"] = comble.to_numpy()
        res.loc[idx, "origine"] = origine

        # On accumule des dictionnaires dans une liste, convertie en
        # DataFrame a la fin. Bien plus rapide que de concatener un DataFrame
        # a chaque tour : pandas recopie tout l'objet a chaque concat.
        regles.journal.append(
            {
                "debut_local": debut.tz_convert(TZ_LOCALE),
                "fin_local": fin.tz_convert(TZ_LOCALE),
                "duree": duree,
                "n_pas": len(idx),
                "methode": origine,
                # Timedelta / Timedelta -> float. Ici : nombre d'heures que
                # represente un pas. W x h / 1000 = kWh.
                "energie_ajoutee_kWh": float(
                    comble.sum() * (pas / pd.Timedelta("1h") / 1000)
                ),
            }
        )

    # pd.DataFrame(liste_de_dicts) : les cles deviennent les colonnes.
    res.attrs["journal_comblement"] = pd.DataFrame(regles.journal)
    return res


def _blocs_manquants(s: pd.Series):
    """Renvoie les couples (premier, dernier) index de chaque bloc de NaN."""
    manque = s.isna()
    if not manque.any():
        return []
    # =====================================================================
    # IDIOME A CONNAITRE : numeroter des sequences consecutives
    # =====================================================================
    # .shift() decale la Series d'un cran vers le bas.
    # (manque != manque.shift()) vaut True a chaque CHANGEMENT d'etat.
    # .cumsum() sur des booleens (True=1) accumule ces changements, ce qui
    # attribue un numero identique a toutes les lignes d'un meme bloc :
    #
    #   manque      F  F  T  T  F  T
    #   != shift    T  F  T  F  T  T
    #   cumsum      1  1  2  2  3  4   <- identifiant de bloc
    #
    # On ne garde ([manque]) que les blocs de NaN.
    groupe = (manque != manque.shift()).cumsum()[manque]
    # .groupby(cles) regroupe par valeur de `cles`, puis on itere sur les
    # couples (valeur_de_cle, sous_Series). Le `_` ignore la cle.
    return [(g.index[0], g.index[-1]) for _, g in s[manque].groupby(groupe)]


def _interpoler(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series | None:
    # .interpolate(method="time") interpole lineairement EN TENANT COMPTE
    # des ecarts de temps reels entre points. method="linear" supposerait des
    # points equidistants — faux des qu'il manque une heure.
    # limit_direction="both" autorise aussi l'extrapolation aux extremites.
    plein = s.interpolate(method="time", limit_direction="both")
    v = plein.loc[idx]
    return None if v.isna().any() else v


def _profil_jour_type(
    res: pd.DataFrame,
    idx: pd.DatetimeIndex,
    types: pd.Series,
    pas: pd.Timedelta,
    regles: ReglesComblement,
) -> pd.Series | None:
    """Mediane, position horaire par position horaire, des jours de meme type."""
    local = idx.tz_convert(TZ_LOCALE)
    reels = res["origine"].isin(ORIGINES_REELLES)
    jour_ref = pd.Series(res.index.tz_convert(TZ_LOCALE).date, index=res.index)
    # "Position dans la journee" exprimee en secondes depuis minuit. Sert de
    # cle d'appariement entre le jour a combler et les jours de reference.
    pos = pd.Series(
        res.index.tz_convert(TZ_LOCALE).hour * 3600
        + res.index.tz_convert(TZ_LOCALE).minute * 60,
        index=res.index,
    )

    sortie = pd.Series(np.nan, index=idx)
    # pd.unique() preserve l'ordre d'apparition, contrairement a set().
    for cible_jour in pd.unique(local.date):
        masque_cible = np.array([d == cible_jour for d in local.date])
        if not masque_cible.any():
            continue
        type_cible = types[types.index.tz_convert(TZ_LOCALE).date == cible_jour]
        if type_cible.empty:
            continue
        type_cible = type_cible.iloc[0]

        # Indexation d'un Index par un masque booleen : on garde les dates
        # ou les trois conditions sont vraies. Les .to_numpy() evitent tout
        # realignement entre masques d'origines differentes.
        candidats = res.index[
            reels.to_numpy()
            & (types.to_numpy() == type_cible)
            & (jour_ref.to_numpy() != cible_jour)
        ]
        if len(candidats) == 0:
            return None

        jours = pd.Series(candidats.tz_convert(TZ_LOCALE).date).unique()
        ecart = np.array([abs((d - cible_jour).days) for d in jours])
        # np.argsort renvoie les INDICES qui trieraient le tableau, pas les
        # valeurs triees. jours[np.argsort(ecart)] = les jours ordonnes par
        # proximite. On garde les n premiers, dans un set pour un test
        # d'appartenance en temps constant.
        retenus = set(jours[np.argsort(ecart)][: regles.n_jours_reference])

        sel = res.loc[candidats]
        garde = np.array([d in retenus for d in candidats.tz_convert(TZ_LOCALE).date])
        sel = sel[garde]
        if sel.empty:
            return None
        # .groupby(serie_de_cles) : regroupe par position horaire, toutes
        # dates confondues. .median() est preferee a .mean() : un jour
        # atypique parmi les references ne deforme pas le profil.
        profil = sel["puissance_W"].groupby(pos.loc[sel.index]).median()

        cibles = idx[masque_cible]
        cles = (
            cibles.tz_convert(TZ_LOCALE).hour * 3600
            + cibles.tz_convert(TZ_LOCALE).minute * 60
        )
        # .reindex(cles) : va chercher dans `profil` la valeur correspondant
        # a chaque position horaire cible. C'est un "lookup" vectorise,
        # equivalent d'un RECHERCHEV sur toute la colonne.
        sortie.loc[cibles] = profil.reindex(cles).to_numpy()

    return None if sortie.isna().any() else sortie


def _annee_precedente(
    res: pd.DataFrame, idx: pd.DatetimeIndex, regles: ReglesComblement
) -> pd.Series | None:
    """Recopie la periode a N-1 (decalage 52 semaines), avec recalage."""
    # DatetimeIndex - Timedelta -> DatetimeIndex decale. Vectorise.
    source = idx - regles.decalage_annuel
    # .reindex(dates) va chercher ces dates dans res ; celles qui n'existent
    # pas donnent une ligne de NaN, sans lever d'exception.
    dispo = res.reindex(source)
    # .all() : la reconstitution n'est acceptee que si TOUTE la periode
    # source est faite de mesures reelles (pas de comblement sur comblement).
    if not dispo["origine"].isin(ORIGINES_REELLES).all():
        return None
    valeurs = dispo["puissance_W"].to_numpy()

    ratio = 1.0
    if regles.recalage_n1:
        ratio = _ratio_recalage(res, idx, regles)
        if ratio is None:
            return None
    # index=idx : on remet les valeurs de N-1 aux dates de N.
    return pd.Series(valeurs * ratio, index=idx)


def _ratio_recalage(
    res: pd.DataFrame, idx: pd.DatetimeIndex, regles: ReglesComblement
) -> float | None:
    """Rapport de consommation N/N-1 sur les fenetres encadrant la lacune."""
    f = regles.fenetre_recalage
    fenetres = [
        (idx[0] - f, idx[0]),    # trois semaines avant la lacune
        (idx[-1], idx[-1] + f),  # trois semaines apres
    ]
    num, den = 0.0, 0.0
    for a, b in fenetres:
        cur = res.loc[a:b]
        ref = res.reindex(cur.index - regles.decalage_annuel)
        # On ne compare que les positions ou les DEUX annees sont mesurees,
        # sinon le ratio serait biaise par des periodes non comparables.
        ok = (
            cur["origine"].isin(ORIGINES_REELLES).to_numpy()
            & ref["origine"].isin(ORIGINES_REELLES).to_numpy()
        )
        # np.nansum somme en ignorant les NaN (np.sum renverrait NaN si un
        # seul element l'est). tableau[masque] = filtrage booleen numpy.
        num += float(np.nansum(cur["puissance_W"].to_numpy()[ok]))
        den += float(np.nansum(ref["puissance_W"].to_numpy()[ok]))
    if den <= 0 or num <= 0:
        return 1.0
    ratio = num / den
    # np.clip(x, bas, haut) borne la valeur. Garde-fou : un ratio aberrant
    # (donnees douteuses de part et d'autre) ne doit pas doubler ou annuler
    # un mois entier de consommation reconstituee.
    return float(np.clip(ratio, 0.5, 2.0))
