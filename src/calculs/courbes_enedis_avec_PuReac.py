"""
Traitement de courbes de charge Enedis pour autoconsommation collective.

Chaine : lecture -> separation des series -> normalisation UTC
         -> agregation a pas regulier -> comblement -> journal qualite.

Conventions
-----------
* Un fichier peut contenir PLUSIEURS series (PA en W, PRI en VAr, CONS, PROD).
  La cle d'une serie est le triplet (grandeur physique, grandeur metier, unite).
* La convention d'horodatage (debut ou fin de pas) est DETECTEE, par comparaison
  de la couverture obtenue avec la fenetre declaree dans le fichier.
  Elle peut aussi etre forcee.
* Chaque mesure est ramenee a un intervalle [debut ; debut + pas]. Le reste du
  module ne connait plus que cette representation.
* Travail interne en UTC, restitution en heure locale.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TZ_LOCALE = "Europe/Paris"

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

# Facteurs de conversion vers l'unite de base de chaque grandeur.
FACTEURS_UNITE = {"W": 1.0, "kW": 1000.0, "VA": 1.0, "kVA": 1000.0,
                  "VAR": 1.0, "KVAR": 1000.0, "VARH": 1.0, "WH": 1.0}


# ---------------------------------------------------------------------------
# 1. Lecture
# ---------------------------------------------------------------------------


def lire_series(
    chemin,
    sep: str | None = None,
    encoding: str | None = None,
    decimal: str = ",",
    convention: str = "auto",
    vraisemblance_acceptee: tuple = (0,),
    natures_ecartees: tuple = (),
) -> dict[str, pd.DataFrame]:
    """Lit un export Enedis et renvoie TOUTES les series qu'il contient.

    Un meme fichier peut melanger puissance active (PA, en W) et reactive
    (PRI, en VAr), consommation et production. Les traiter ensemble
    produirait des horodates en doublon et un melange d'unites.

    Renvoie un dictionnaire {"PA/CONS": DataFrame, "PRI/CONS": DataFrame, ...}.
    """
    if encoding is None:
        encoding = _detecter_encodage(chemin)
    if sep is None:
        sep = _detecter_separateur(chemin, encoding)

    brut = pd.read_csv(
        chemin, sep=sep, encoding=encoding, decimal=decimal,
        dtype={COLONNES["prm"]: str},
    )
    brut.columns = [c.strip() for c in brut.columns]

    # Cle de serie : ce qui rend deux lignes incomparables entre elles.
    cles = [COLONNES[k] for k in ("grandeur_physique", "grandeur_metier")
            if COLONNES[k] in brut.columns]
    if not cles:
        return {"?": normaliser(brut, convention=convention,
                                vraisemblance_acceptee=vraisemblance_acceptee,
                                natures_ecartees=natures_ecartees)}

    series = {}
    for valeurs, sous in brut.groupby(cles, sort=False):
        nom = "/".join(str(v) for v in np.atleast_1d(valeurs))
        series[nom] = normaliser(
            sous, convention=convention,
            vraisemblance_acceptee=vraisemblance_acceptee,
            natures_ecartees=natures_ecartees,
        )
    return series


def lire_courbe(
    chemin,
    grandeur_physique: str = "PA",
    grandeur_metier: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Lit un export et renvoie UNE serie (par defaut la puissance active).

    Signale la presence des autres series plutot que de les melanger.
    """
    series = lire_series(chemin, **kwargs)
    candidats = {
        nom: df for nom, df in series.items()
        if nom.split("/")[0] == grandeur_physique
        and (grandeur_metier is None or grandeur_metier in nom.split("/"))
    }
    if not candidats:
        raise ValueError(
            f"Aucune serie {grandeur_physique} dans ce fichier. "
            f"Series disponibles : {sorted(series)}"
        )
    if len(series) > 1:
        warnings.warn(
            f"Le fichier contient {len(series)} series : {sorted(series)}. "
            f"Seule {sorted(candidats)[0]} est renvoyee ; utiliser lire_series() "
            "pour toutes les obtenir.",
            stacklevel=2,
        )
    if len(candidats) > 1:
        raise ValueError(
            f"Plusieurs series {grandeur_physique} : {sorted(candidats)}. "
            "Preciser grandeur_metier."
        )
    return next(iter(candidats.values()))


def _detecter_encodage(chemin) -> str:
    with open(chemin, "rb") as f:
        tete = f.read(4096)
    if tete.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        tete.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def _detecter_separateur(chemin, encoding: str) -> str:
    with open(chemin, encoding=encoding) as f:
        entete = f.readline()
    comptes = {s: entete.count(s) for s in (";", "\t", ",")}
    return max(comptes, key=comptes.get)


def _to_datetime(serie: pd.Series) -> pd.Series:
    """Parse des horodates ISO (2024-06-23 00:30:00) ou francaises."""
    echantillon = str(serie.dropna().iloc[0]).strip()
    iso = len(echantillon) >= 4 and echantillon[:4].isdigit()
    return pd.to_datetime(serie, format="mixed", dayfirst=not iso)


def _parse_pas(v) -> pd.Timedelta:
    """Parse un pas ISO 8601 (PT30M) ou une valeur en minutes."""
    if isinstance(v, pd.Timedelta):
        return v
    if pd.isna(v):
        return pd.NaT
    s = str(v).strip().upper()
    if s.startswith("P"):
        return pd.Timedelta(s)
    return pd.Timedelta(minutes=float(s.replace(",", ".")))


def _detecter_convention(
    index: pd.DatetimeIndex,
    pas: pd.Series,
    debut_demande,
    fin_demande,
) -> tuple[str, str]:
    """Determine si l'horodate marque le debut ou la fin de l'intervalle.

    Principe : sous chaque hypothese, on calcule la periode reellement
    couverte par les mesures, et on la compare a la fenetre declaree dans les
    colonnes "Date de debut" / "Date de fin". La bonne convention est celle
    qui tombe juste.

    Exemple sur un fichier au pas 5 min, fenetre 23/06/2024 00:00 -> 23/06/2026 :
      horodates observees      00:00  ->  22/06/2026 23:55
      hypothese "fin"    : couverture 22/06/2024 23:55 -> 22/06/2026 23:55 (ecart 10 min)
      hypothese "debut"  : couverture 23/06/2024 00:00 -> 23/06/2026 00:00 (ecart 0)
    """
    if debut_demande is None or fin_demande is None:
        return "fin", "aucune fenetre declaree, convention 'fin' par defaut"

    h0, hN = index[0], index[-1]  # index suppose trie
    pas0, pasN = pas.iloc[0], pas.iloc[-1]

    hypotheses = {
        "fin": (h0 - pas0, hN),
        "debut": (h0, hN + pasN),
    }
    ecarts = {
        nom: abs((a - debut_demande).total_seconds())
        + abs((b - fin_demande).total_seconds())
        for nom, (a, b) in hypotheses.items()
    }
    meilleure = min(ecarts, key=ecarts.get)

    if ecarts["fin"] == ecarts["debut"]:
        return "fin", "hypotheses indiscernables, convention 'fin' par defaut"
    return meilleure, (
        f"ecart a la fenetre declaree : debut={ecarts['debut']:.0f}s, "
        f"fin={ecarts['fin']:.0f}s"
    )


def normaliser(
    df: pd.DataFrame,
    convention: str = "auto",
    vraisemblance_acceptee: tuple = (0,),
    natures_ecartees: tuple = (),
) -> pd.DataFrame:
    """Normalise UNE serie homogene (une seule grandeur, une seule unite).

    Renvoie un DataFrame indexe sur l'horodate en UTC, avec les colonnes :
      valeur (unite de base), pas (Timedelta), debut (Timestamp UTC),
      nature, vraisemblance, etape.

    La colonne `debut` est le resultat de la resolution de convention : a
    partir d'ici, une mesure est l'intervalle [debut ; debut + pas].
    """
    df = df.copy()
    c = COLONNES

    # --- PRM ---
    prm = None
    if c["prm"] in df.columns:
        prm = str(df[c["prm"]].iloc[0]).strip()
        if "E+" in prm.upper() or "," in prm:
            warnings.warn(
                f"PRM corrompu par une ouverture Excel ({prm!r}) : les 14 chiffres "
                "sont irrecuperables. Reprendre le fichier brut Enedis.",
                stacklevel=2,
            )

    # --- Unite : propre a la serie, jamais lue sur une autre grandeur ---
    unite = str(df[c["unite"]].iloc[0]).strip() if c["unite"] in df.columns else "W"
    if c["unite"] in df.columns and df[c["unite"]].nunique() > 1:
        raise ValueError(
            f"Plusieurs unites dans la meme serie : {sorted(df[c['unite']].unique())}. "
            "Utiliser lire_series() pour separer les grandeurs."
        )
    facteur = FACTEURS_UNITE.get(unite.upper(), 1.0)

    # --- Fenetre declaree (necessaire avant la detection de convention) ---
    fenetre = {}
    for cle in ("debut_demande", "fin_demande"):
        if COLONNES[cle] in df.columns:
            t = _to_datetime(df[COLONNES[cle]].head(1)).iloc[0]
            fenetre[cle] = (
                t.tz_localize(TZ_LOCALE, ambiguous=True, nonexistent="shift_forward")
                .tz_convert("UTC")
            )

    # --- Horodate : localisation AVANT tout tri ---
    # ambiguous="infer" tranche entre les deux passages de l'heure d'automne
    # en s'appuyant sur l'ORDRE DU FICHIER, qui est l'ordre chronologique reel.
    # Trier en heure locale naive avant de localiser reordonnerait
    # 02:00, 02:30, 02:00, 02:30 en 02:00, 02:00, 02:30, 02:30 et detruirait
    # justement l'information dont infer a besoin. Le tri vient donc apres,
    # sur l'index UTC, ou il est sans risque.
    # La separation des series par lire_series() est en revanche indispensable
    # avant cet appel : sur un fichier concatenant PA puis PRI, la sequence
    # n'est pas chronologique et infer echouerait.
    h = _to_datetime(df[c["horodate"]])
    if h.dt.tz is None:
        try:
            h = h.dt.tz_localize(
                TZ_LOCALE, ambiguous="infer", nonexistent="shift_forward"
            )
        except (ValueError, pd.errors.OutOfBoundsDatetime) as err:
            # Repli : sur l'heure repetee, la 1re occurrence est en heure d'ete.
            # Les entrees non ambigues ignorent cette valeur.
            warnings.warn(
                f"Passage a l'heure d'hiver non inferable ({err}) ; repli sur "
                "l'ordre d'apparition (1re occurrence = heure d'ete). "
                "Verifier que le fichier est bien en ordre chronologique.",
                stacklevel=2,
            )
            h = h.dt.tz_localize(
                TZ_LOCALE,
                ambiguous=~h.duplicated(keep="first").to_numpy(),
                nonexistent="shift_forward",
            )
    h = h.dt.tz_convert("UTC")

    out = pd.DataFrame(
        {
            "valeur": pd.to_numeric(df[c["valeur"]], errors="coerce").to_numpy()
            * facteur,
            "pas": df[c["pas"]].map(_parse_pas).to_numpy(),
        },
        index=pd.DatetimeIndex(h.to_numpy(), name="horodate", tz="UTC"),
    )
    for cle in ("nature", "vraisemblance", "etape"):
        if COLONNES[cle] in df.columns:
            out[cle] = df[COLONNES[cle]].values

    # --- Dedoublonnage (vrais doublons uniquement, apres separation UTC) ---
    if out.index.has_duplicates:
        rang = (
            out["etape"].map(PREFERENCE_ETAPE).fillna(0)
            if "etape" in out.columns
            else pd.Series(0, index=out.index)
        )
        out = out.assign(_r=rang.values).sort_values("_r")
        n_avant = len(out)
        out = out[~out.index.duplicated(keep="last")].drop(columns="_r")
        warnings.warn(
            f"{n_avant - len(out)} horodates en doublon supprimees "
            "(priorite CORRIGE > COMPLETE > BRUT).",
            stacklevel=2,
        )
    out = out.sort_index()
    out = out.dropna(subset=["pas"])
    if out.empty:
        raise ValueError("Serie vide apres normalisation (colonne 'Pas' illisible).")

    # --- Convention d'horodatage ---
    if convention == "auto":
        convention, motif = _detecter_convention(
            out.index, out["pas"],
            fenetre.get("debut_demande"), fenetre.get("fin_demande"),
        )
    else:
        motif = "forcee par l'appelant"
    if convention not in ("debut", "fin"):
        raise ValueError("convention doit valoir 'auto', 'debut' ou 'fin'.")

    # A partir d'ici, plus aucune hypothese : une mesure est un intervalle.
    out["debut"] = out.index - out["pas"] if convention == "fin" else out.index

    # --- Filtres qualite ---
    # Indice de vraisemblance : une valeur ABSENTE (null, NaN) signifie
    # "pas d'information", et non "donnee douteuse". Ne rien ecarter dans ce
    # cas. C'est la difference avec un indice explicitement renseigne.
    ecarte = pd.Series(False, index=out.index)
    if "vraisemblance" in out.columns:
        renseigne = out["vraisemblance"].notna()
        suspect = renseigne & ~out["vraisemblance"].isin(vraisemblance_acceptee)
        if suspect.any():
            codes = sorted(out.loc[suspect, "vraisemblance"].unique())
            warnings.warn(
                f"{int(suspect.sum())} valeur(s) ecartee(s) sur indice de "
                f"vraisemblance {codes} -> traitees comme lacunes.",
                stacklevel=2,
            )
        ecarte |= suspect

    # Nature : certains exports n'ont pas d'indice de vraisemblance et
    # portent l'information qualite ici. Les codes varient selon le type de
    # comptage, donc rien n'est ecarte par defaut.
    if "nature" in out.columns and natures_ecartees:
        suspect = out["nature"].isin(natures_ecartees)
        if suspect.any():
            warnings.warn(
                f"{int(suspect.sum())} valeur(s) ecartee(s) sur nature "
                f"{sorted(out.loc[suspect, 'nature'].unique())}.",
                stacklevel=2,
            )
        ecarte |= suspect

    out.loc[ecarte, "valeur"] = np.nan
    out = out.dropna(subset=["valeur"])

    # --- Metadonnees ---
    out.attrs["prm"] = prm
    out.attrs["unite_source"] = unite
    out.attrs["convention"] = convention
    out.attrs["convention_motif"] = motif
    out.attrs["n_ecartees"] = int(ecarte.sum())
    for cle in ("grandeur_metier", "grandeur_physique"):
        if COLONNES[cle] in df.columns:
            out.attrs[cle] = str(df[COLONNES[cle]].iloc[0]).strip()
    out.attrs.update(fenetre)
    return out


def controler_pas_declare(df: pd.DataFrame) -> pd.DataFrame:
    """Compare le pas declare a l'ecart reel entre debuts d'intervalle."""
    debut = df["debut"] if "debut" in df.columns else df.index.to_series()
    delta = debut.diff()
    ecart = delta - df["pas"].shift()
    return pd.DataFrame(
        {
            "pas_declare": df["pas"],
            "delta_observe": delta,
            "ecart": ecart,
            "trou": ecart > pd.Timedelta(0),
            "recouvrement": ecart < pd.Timedelta(0),
        }
    )


def diagnostic(chemin, **kwargs) -> pd.DataFrame:
    """Inventaire d'un fichier avant traitement : series, pas, qualite.

    A executer systematiquement sur un nouvel export : c'est ce qui aurait
    signale d'emblee la presence de deux grandeurs et la convention inversee.
    """
    series = lire_series(chemin, **kwargs)
    lignes = []
    for nom, df in series.items():
        pas = df["pas"].value_counts()
        natures = df["nature"].value_counts() if "nature" in df.columns else {}
        lignes.append({
            "serie": nom,
            "prm": df.attrs.get("prm"),
            "unite": df.attrs.get("unite_source"),
            "n_lignes": len(df),
            "convention": df.attrs.get("convention"),
            "debut": df["debut"].min().tz_convert(TZ_LOCALE),
            "fin": (df["debut"] + df["pas"]).max().tz_convert(TZ_LOCALE),
            "pas": ", ".join(f"{str(k).split()[-1]}x{v}" for k, v in pas.items()),
            "natures": dict(natures),
            "ecartees": df.attrs.get("n_ecartees", 0),
        })
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 2. Agregation a pas regulier
# ---------------------------------------------------------------------------


def pgcd_pas(pas: pd.Series, cible: pd.Timedelta) -> pd.Timedelta:
    valides = pas.dropna().unique()
    secondes = [int(pd.Timedelta(p).total_seconds()) for p in valides]
    secondes.append(int(cible.total_seconds()))
    return pd.Timedelta(seconds=int(np.gcd.reduce(secondes)))


def agreger(
    df: pd.DataFrame,
    cible: str | pd.Timedelta = "1h",
    seuil_couverture: float = 0.75,
    fenetre: tuple | None = None,
) -> pd.DataFrame:
    """Agrege une courbe a pas variable vers un pas regulier.

    Integration d'une fonction en escalier : chaque mesure est projetee sur
    une grille au PGCD des pas, puis moyennee.

    S'appuie sur la colonne `debut` produite par normaliser() : aucune
    hypothese de convention d'horodatage n'est faite ici.
    """
    cible = pd.Timedelta(cible)
    df = df.dropna(subset=["pas"])
    if df.empty:
        raise ValueError("Aucune ligne exploitable (colonne 'pas' vide).")
    if "debut" not in df.columns:
        raise ValueError(
            "Colonne 'debut' absente : passer par normaliser()/lire_series()."
        )

    pas_fin = pgcd_pas(df["pas"], cible)

    if fenetre is not None:
        t0, t1 = (pd.Timestamp(x).tz_convert("UTC") for x in fenetre)
    else:
        t0 = df.attrs.get("debut_demande", df["debut"].min())
        t1 = df.attrs.get("fin_demande", (df["debut"] + df["pas"]).max())
    t0 = pd.Timestamp(t0).ceil(cible)
    t1 = pd.Timestamp(t1).floor(cible)

    # --- Explosion sur la grille fine ---
    n = (df["pas"] // pas_fin).astype("int64").to_numpy()
    debuts = pd.DatetimeIndex(df["debut"]).tz_localize(None).to_numpy()
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    sous_debuts = np.repeat(debuts, n) + offsets * pas_fin.to_timedelta64()
    valeurs = np.repeat(df["valeur"].to_numpy(), n)

    fine = pd.Series(
        valeurs, index=pd.DatetimeIndex(sous_debuts).tz_localize("UTC")
    )
    fine = fine[~fine.index.duplicated(keep="last")].sort_index()

    grille = pd.date_range(t0, t1, freq=pas_fin, inclusive="left", tz="UTC")
    fine = fine.reindex(grille)

    n_max = int(cible / pas_fin)
    bloc = fine.resample(cible)
    couverture = bloc.count() / n_max
    puissance = bloc.mean().where(couverture >= seuil_couverture)

    origine = pd.Series(pd.NA, index=puissance.index, dtype="object")
    origine[couverture >= 1.0] = MESURE
    origine[(couverture >= seuil_couverture) & (couverture < 1.0)] = MESURE_PARTIELLE

    res = pd.DataFrame(
        {"puissance": puissance, "couverture": couverture, "origine": origine}
    )
    res.attrs.update(df.attrs)
    res.attrs["pas_grille_fine"] = pas_fin
    res.attrs["pas_cible"] = cible
    return res


# ---------------------------------------------------------------------------
# 3. Comblement des lacunes
# ---------------------------------------------------------------------------


@dataclass
class ReglesComblement:
    duree_max_interpolation: pd.Timedelta = pd.Timedelta("3h")
    duree_max_jour_type: pd.Timedelta = pd.Timedelta("3D")
    n_jours_reference: int = 4
    decalage_annuel: pd.Timedelta = pd.Timedelta("364D")
    recalage_n1: bool = True
    fenetre_recalage: pd.Timedelta = pd.Timedelta("21D")
    taux_reel_minimum: float = 0.80
    journal: list = field(default_factory=list)


def _paques(annee: int) -> _dt.date:
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
    p = _paques(annee)
    return {
        _dt.date(annee, 1, 1),
        p + _dt.timedelta(days=1),
        _dt.date(annee, 5, 1),
        _dt.date(annee, 5, 8),
        p + _dt.timedelta(days=39),
        p + _dt.timedelta(days=50),
        _dt.date(annee, 7, 14),
        _dt.date(annee, 8, 15),
        _dt.date(annee, 11, 1),
        _dt.date(annee, 11, 11),
        _dt.date(annee, 12, 25),
    }


def type_de_jour(dates: pd.DatetimeIndex) -> pd.Series:
    feries = set()
    for a in range(dates.year.min(), dates.year.max() + 1):
        feries |= jours_feries(a)
    d = pd.Series(dates.date, index=dates)
    jour = pd.Series(dates.dayofweek, index=dates)
    t = pd.Series("OUVRE", index=dates)
    t[jour == 5] = "SAMEDI"
    t[(jour == 6) | d.isin(feries)] = "DIMANCHE_FERIE"
    return t


def combler(df: pd.DataFrame, regles: ReglesComblement | None = None) -> pd.DataFrame:
    regles = regles or ReglesComblement()
    res = df.copy()
    pas = pd.Timedelta(res.attrs.get("pas_cible", "1h"))
    types = type_de_jour(res.index.tz_convert(TZ_LOCALE))

    for debut, fin in _blocs_manquants(res["puissance"]):
        duree = fin - debut + pas
        idx = res.loc[debut:fin].index

        if duree <= regles.duree_max_interpolation:
            comble = _interpoler(res["puissance"], idx)
            origine = INTERP_COURTE
        elif duree <= regles.duree_max_jour_type:
            comble = _profil_jour_type(res, idx, types, pas, regles)
            origine = PROFIL_JOUR_TYPE
        else:
            comble = _annee_precedente(res, idx, regles)
            origine = ANNEE_N1

        if comble is None or comble.isna().any():
            comble = pd.Series(0.0, index=idx)
            origine = ZERO_FORCE

        res.loc[idx, "puissance"] = comble.to_numpy()
        res.loc[idx, "origine"] = origine
        regles.journal.append({
            "debut_local": debut.tz_convert(TZ_LOCALE),
            "fin_local": fin.tz_convert(TZ_LOCALE),
            "duree": duree,
            "n_pas": len(idx),
            "methode": origine,
            "energie_ajoutee_kWh": float(
                comble.sum() * pas / pd.Timedelta("1h") / 1000
            ),
        })

    res.attrs["journal_comblement"] = pd.DataFrame(regles.journal)
    return res


def _blocs_manquants(s: pd.Series):
    manque = s.isna()
    if not manque.any():
        return []
    groupe = (manque != manque.shift()).cumsum()[manque]
    return [(g.index[0], g.index[-1]) for _, g in s[manque].groupby(groupe)]


def _interpoler(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series | None:
    plein = s.interpolate(method="time", limit_direction="both")
    v = plein.loc[idx]
    return None if v.isna().any() else v


def _profil_jour_type(res, idx, types, pas, regles) -> pd.Series | None:
    local = idx.tz_convert(TZ_LOCALE)
    reels = res["origine"].isin(ORIGINES_REELLES)
    jour_ref = pd.Series(res.index.tz_convert(TZ_LOCALE).date, index=res.index)
    pos = pd.Series(
        res.index.tz_convert(TZ_LOCALE).hour * 3600
        + res.index.tz_convert(TZ_LOCALE).minute * 60,
        index=res.index,
    )

    sortie = pd.Series(np.nan, index=idx)
    for cible_jour in pd.unique(local.date):
        masque_cible = np.array([d == cible_jour for d in local.date])
        if not masque_cible.any():
            continue
        type_cible = types[types.index.tz_convert(TZ_LOCALE).date == cible_jour]
        if type_cible.empty:
            continue
        type_cible = type_cible.iloc[0]

        candidats = res.index[
            reels.to_numpy()
            & (types.to_numpy() == type_cible)
            & (jour_ref.to_numpy() != cible_jour)
        ]
        if len(candidats) == 0:
            return None
        jours = pd.Series(candidats.tz_convert(TZ_LOCALE).date).unique()
        ecart = np.array([abs((d - cible_jour).days) for d in jours])
        retenus = set(jours[np.argsort(ecart)][: regles.n_jours_reference])

        sel = res.loc[candidats]
        garde = np.array([d in retenus for d in candidats.tz_convert(TZ_LOCALE).date])
        sel = sel[garde]
        if sel.empty:
            return None
        profil = sel["puissance"].groupby(pos.loc[sel.index]).median()

        cibles = idx[masque_cible]
        cles = (
            cibles.tz_convert(TZ_LOCALE).hour * 3600
            + cibles.tz_convert(TZ_LOCALE).minute * 60
        )
        sortie.loc[cibles] = profil.reindex(cles).to_numpy()

    return None if sortie.isna().any() else sortie


def _annee_precedente(res, idx, regles) -> pd.Series | None:
    source = idx - regles.decalage_annuel
    dispo = res.reindex(source)
    if not dispo["origine"].isin(ORIGINES_REELLES).all():
        return None
    valeurs = dispo["puissance"].to_numpy()
    ratio = 1.0
    if regles.recalage_n1:
        ratio = _ratio_recalage(res, idx, regles)
        if ratio is None:
            return None
    return pd.Series(valeurs * ratio, index=idx)


def _ratio_recalage(res, idx, regles) -> float | None:
    f = regles.fenetre_recalage
    num, den = 0.0, 0.0
    for a, b in [(idx[0] - f, idx[0]), (idx[-1], idx[-1] + f)]:
        cur = res.loc[a:b]
        ref = res.reindex(cur.index - regles.decalage_annuel)
        ok = (
            cur["origine"].isin(ORIGINES_REELLES).to_numpy()
            & ref["origine"].isin(ORIGINES_REELLES).to_numpy()
        )
        num += float(np.nansum(cur["puissance"].to_numpy()[ok]))
        den += float(np.nansum(ref["puissance"].to_numpy()[ok]))
    if den <= 0 or num <= 0:
        return 1.0
    return float(np.clip(num / den, 0.5, 2.0))


# ---------------------------------------------------------------------------
# 4. Journal qualite et export
# ---------------------------------------------------------------------------


def bilan_qualite(res: pd.DataFrame) -> dict:
    pas = pd.Timedelta(res.attrs.get("pas_cible", "1h"))
    h = pas / pd.Timedelta("1h")
    unite = res.attrs.get("unite_source", "W")
    parts = res["origine"].value_counts(normalize=True).to_dict()
    reel = sum(parts.get(o, 0.0) for o in ORIGINES_REELLES)
    return {
        "prm": res.attrs.get("prm"),
        "serie": f"{res.attrs.get('grandeur_physique')}/"
                 f"{res.attrs.get('grandeur_metier')}",
        "unite": unite,
        "convention": res.attrs.get("convention"),
        "n_pas": len(res),
        "taux_reel": reel,
        "repartition_origines": parts,
        "energie_totale_k": float(res["puissance"].sum() * h / 1000),
        "energie_par_origine_k": (
            res.groupby("origine")["puissance"].sum() * h / 1000
        ).to_dict(),
        "exploitable": reel >= ReglesComblement().taux_reel_minimum,
    }


def exporter(res: pd.DataFrame, chemin: str) -> None:
    sortie = res.copy()
    sortie.index = sortie.index.tz_convert(TZ_LOCALE)
    sortie.index.name = "horodate_locale"
    sortie.to_csv(chemin, sep=";", decimal=",", float_format="%.3f")
