"""
Traitement de courbes de charge Enedis pour autoconsommation collective.
=========================================================================

VERSION COMMENTEE. Le code est identique a courbes_enedis.py ; seuls les
commentaires changent. Ils expliquent l'usage des bibliotheques (pandas,
numpy) plutot que la logique metier.

Bibliotheques utilisees
-----------------------
pandas  : manipulation de donnees tabulaires *indexees*. C'est l'index qui
          fait toute la difference avec une liste de dictionnaires : pandas
          sait aligner deux series sur leurs dates communes, decouper par
          tranche temporelle, regrouper. On s'en sert ici parce que toute
          la chaine est une histoire de dates.
numpy   : calcul vectorise sur des tableaux homogenes. pandas est construit
          au-dessus. On y descend quand on veut de la vitesse ou une
          operation que pandas n'expose pas (np.repeat, np.gcd).
datetime: bibliotheque standard. Utilisee uniquement pour les dates
          "calendaires" pures (jours feries), ou l'heure ne joue aucun role.

Les deux types pandas centraux ici
----------------------------------
Timestamp : un instant. Equivalent pandas de datetime.datetime, mais avec
            des methodes en plus (.floor, .ceil, .tz_convert).
Timedelta : une duree. Equivalent de datetime.timedelta. Timestamp - Timestamp
            donne un Timedelta ; Timestamp + Timedelta donne un Timestamp.
            C'est ce qui permet d'ecrire l'arithmetique temporelle
            naturellement, sans jamais convertir en secondes a la main.

Conventions retenues
--------------------
* Horodatage en FIN de pas : la ligne (Horodate=t, Pas=d) decrit la puissance
  moyenne sur l'intervalle [t - d ; t].
* La colonne "Pas" (ISO 8601, ex. PT30M) fait foi pour la duree couverte.
* Travail interne en UTC, restitution en Europe/Paris.
* Puissances en W.
"""

# `from __future__ import annotations` rend les annotations de type purement
# textuelles. Concretement : on peut ecrire `-> pd.DataFrame` ou
# `str | None` meme sur d'anciennes versions de Python, sans que
# l'interpreteur n'evalue l'expression a la definition de la fonction.
# Sans cet import, `ReglesComblement | None` planterait dans la signature
# de combler(), la classe n'etant pas encore definie a ce moment-la.
from __future__ import annotations

# Alias `_dt` : le underscore signale "usage interne au module", et evite
# de confondre datetime (module standard) avec les Timestamp de pandas.
import datetime as _dt

# `warnings` fait partie de la bibliotheque standard. warnings.warn() emet
# un message non bloquant : le programme continue. C'est le bon outil pour
# signaler "j'ai modifie tes donnees" — une exception arreterait tout, un
# print() serait invisible si l'appelant redirige la sortie, alors qu'un
# warning peut etre capture, filtre ou transforme en erreur par l'appelant.
import warnings

# dataclass : decorateur standard qui genere __init__, __repr__ et __eq__
# a partir des attributs annotes. `field` sert aux valeurs par defaut
# mutables (voir ReglesComblement plus bas).
from dataclasses import dataclass, field

import numpy as np
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


# ---------------------------------------------------------------------------
# 1. Lecture
# ---------------------------------------------------------------------------

# Table de correspondance nom_interne -> nom dans le fichier Enedis.
# Si Enedis renomme une colonne, on modifie ici et nulle part ailleurs.
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


def lire_courbe(
    chemin,
    sep: str | None = None,
    encoding: str | None = None,
    decimal: str = ",",
    vraisemblance_acceptee: tuple = (0,),
) -> pd.DataFrame:
    """Lit un export Enedis et renvoie un DataFrame normalise."""
    if encoding is None:
        encoding = _detecter_encodage(chemin)
    if sep is None:
        sep = _detecter_separateur(chemin, encoding)

    # pd.read_csv est la porte d'entree de pandas. Les quatre parametres
    # utilises ici meritent chacun une explication :
    #
    # sep      : le separateur de colonnes. Enedis utilise ";" ou tabulation
    #            selon l'export ; la virgule serait ambigue avec le decimal.
    #
    # encoding : comment traduire les octets du fichier en caracteres   . Une
    #            erreur ici donne "Date de dÃ©but" au lieu de "Date de début",
    #            et le lookup dans COLONNES echoue.
    #
    # decimal  : le caractere separateur decimal. Avec decimal=",", pandas
    #            lit "1234,5" comme le nombre 1234.5. Sans, il le lit comme
    #            une chaine de caracteres et toute la colonne devient du
    #            texte — les calculs echoueront plus loin, mais silencieusement
    #            au debut (concatener des chaines ne leve pas d'erreur).
    #
    # dtype    : force le type d'une colonne. C'est LE parametre critique ici.
    #            Par defaut pandas devine, et sur "19905643910746" il devine
    #            int64. C'est correct, mais si le fichier contient
    #            "1,99056E+13" (fichier deja passe par Excel) il devine float
    #            et le PRM est perdu. Forcer str desactive toute conversion.
    df = pd.read_csv(
        chemin,
        sep=sep,
        encoding=encoding,
        decimal=decimal,
        dtype={COLONNES["prm"]: str},
        keep_default_na=True,  # "NA", "", "NaN"... deviennent des NaN
    )

    # df.columns est un Index (pas une liste Python) contenant les noms de
    # colonnes. On peut le remplacer par n'importe quelle sequence de meme
    # longueur. Ici on retire les espaces parasites : un en-tete
    # "Valeur " (avec espace final) ferait echouer df["Valeur"].
    df.columns = [c.strip() for c in df.columns]
    return normaliser(df, vraisemblance_acceptee=vraisemblance_acceptee)


def _detecter_encodage(chemin) -> str:
    """UTF-8 si la premiere ligne se decode, cp1252 sinon."""
    # Ouverture en mode binaire "rb" : on veut les octets bruts, justement
    # parce qu'on ne sait pas encore comment les interpreter.
    with open(chemin, "rb") as f:
        tete = f.read(4096)
    # BOM : 3 octets que Windows ajoute parfois en tete de fichier UTF-8.
    # L'encodage "utf-8-sig" les consomme ; "utf-8" les laisserait dans le
    # nom de la premiere colonne.
    if tete.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        # Si ces octets ne forment pas de l'UTF-8 valide, .decode() leve
        # UnicodeDecodeError. C'est un test fiable : l'UTF-8 a une structure
        # tres contrainte, du texte latin-1 accentue echoue presque toujours.
        tete.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"  # l'encodage Windows occidental, defaut d'Excel FR


def _detecter_separateur(chemin, encoding: str) -> str:
    """Separateur le plus frequent dans l'en-tete parmi ; \\t ,"""
    with open(chemin, encoding=encoding) as f:
        entete = f.readline()
    comptes = {s: entete.count(s) for s in (";", "\t", ",")}
    # max(dict, key=dict.get) renvoie la CLE dont la valeur est maximale.
    # Sans le parametre key, max() sur un dict renverrait la plus grande cle
    # au sens alphabetique — pas du tout ce qu'on veut.
    return max(comptes, key=comptes.get)


def _to_datetime(serie: pd.Series) -> pd.Series:
    """Parse des horodates ISO ou francaises."""
    # .dropna() retire les valeurs manquantes ; .iloc[0] prend le premier
    # element PAR POSITION. Distinction essentielle en pandas :
    #   .iloc[0] = premier element (position)
    #   .loc[0]  = element dont l'etiquette d'index vaut 0
    # Sur un DataFrame filtre, les etiquettes ont des trous : .loc[0] peut
    # lever KeyError la ou .iloc[0] fonctionne toujours.
    echantillon = str(serie.dropna().iloc[0]).strip()
    iso = len(echantillon) >= 4 and echantillon[:4].isdigit()

    # pd.to_datetime convertit une Series de chaines en Series de Timestamp.
    #
    # format="mixed" : autorise un format different par ligne. Plus lent
    #   qu'un format fixe, mais tolerant aux exports heterogenes.
    #
    # dayfirst : lever d'ambiguite pour "03/04/2024". True -> 3 avril,
    #   False -> 4 mars. C'est un piege classique : sur deux ans de donnees,
    #   seules les 12 premieres journees de chaque mois sont ambigues, donc
    #   l'erreur ne se voit pas sur un echantillon et corrompt ~40 % des
    #   dates. D'ou la detection prealable du format ISO.
    return pd.to_datetime(serie, format="mixed", dayfirst=not iso)


def normaliser(
    df: pd.DataFrame, vraisemblance_acceptee: tuple = (0,)
) -> pd.DataFrame:
    """Normalise un DataFrame deja charge."""
    # .copy() : sans lui, les modifications remonteraient dans le DataFrame
    # de l'appelant. pandas passe les objets par reference comme tout Python.
    df = df.copy()
    c = COLONNES

    # enlève les puissances qui ne sont pas en W,  que la puissance apparente, pas de puissance réactive
    
    if c["grandeur_physique"] in df.columns:
        df = df[
            df[c["grandeur_physique"]]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq("PA")
        ]

    if c["vraisemblance"] in df.columns:
       df[c["vraisemblance"]] = df[c["vraisemblance"]].fillna(0)

    # --- PRM : garde-fou contre la corruption par Excel ---
    prm = None
    if c["prm"] in df.columns:
        # df[nom] renvoie une Series (une colonne). .iloc[0] = premiere valeur.
        prm = str(df[c["prm"]].iloc[0]).strip()
        if "E+" in prm.upper() or "," in prm:
            # {prm!r} applique repr() : affiche les guillemets, ce qui rend
            # visibles les espaces parasites eventuels.
            # stacklevel=2 fait pointer le warning vers la ligne de l'APPELANT
            # plutot que vers ce warnings.warn() : bien plus utile a deboguer.
            warnings.warn(
                f"PRM corrompu par une ouverture Excel ({prm!r}) : les 14 chiffres "
                "sont irrecuperables. Reprendre le fichier brut Enedis ou fournir "
                "le PRM explicitement.",
                stacklevel=2,
            )

    # --- Unite ---
    unite = str(df[c["unite"]].iloc[0]).strip() if c["unite"] in df.columns else "W"
    # dict.get(cle, defaut) : pas de KeyError si l'unite est inconnue.
    facteur = {"W": 1.0, "kW": 1000.0, "VA": 1.0, "kVA": 1000.0}.get(unite, 1.0)

    # --- Horodate -> UTC ---
    h = _to_datetime(df[c["horodate"]])

    # L'accesseur .dt : sur une Series de Timestamp, il donne acces aux
    # methodes et attributs de date. Meme principe que .str pour le texte.
    #   serie.dt.hour, serie.dt.dayofweek, serie.dt.tz_localize(...)
    # On ne peut PAS ecrire serie.hour directement : les Series n'exposent
    # pas les attributs de leurs elements, il faut passer par l'accesseur.
    if h.dt.tz is None:
        # tz_localize ATTACHE un fuseau a des dates naives. Il ne change
        # aucune valeur affichee : il declare "ces heures etaient de l'heure
        # de Paris". A distinguer de tz_convert, qui traduit d'un fuseau vers
        # un autre en decalant les valeurs.
        #
        # ambiguous="infer" : le dernier dimanche d'octobre, 02:30 local
        # existe deux fois (une fois en UTC+2, une fois en UTC+1). "infer"
        # deduit lequel est lequel en exigeant que la serie reste croissante.
        # C'est ce qui separe les 4 pretendus doublons de votre fichier.
        # Alternatives : ambiguous=True (toujours le premier, donc perte),
        # ambiguous="NaT" (marque en manquant), ou un tableau de booleens.
        #
        # nonexistent="shift_forward" : le dernier dimanche de mars, 02:30
        # local n'existe pas (on saute de 02:00 a 03:00). "shift_forward"
        # decale vers l'avant plutot que de lever une exception.
        h = h.dt.tz_localize(
            TZ_LOCALE, ambiguous="infer", nonexistent="shift_forward"
        )
    # tz_convert : meme instant, exprime dans un autre fuseau. Passer en UTC
    # rend l'arithmetique lineaire (une heure vaut toujours une heure).
    h = h.dt.tz_convert("UTC")

    # ======================================================================
    # PIEGE MAJEUR : l'alignement automatique de pandas
    # ======================================================================
    # Quand on construit un DataFrame a partir de Series, pandas ALIGNE ces
    # Series sur l'index fourni. Ici, df["Valeur"] porte le RangeIndex du
    # fichier (0, 1, 2, ...) alors que index= vaut des dates. Aucune
    # etiquette ne correspond, donc pandas remplit tout de NaN. Sans erreur.
    #
    # .to_numpy() supprime l'index et ne transmet que les valeurs brutes :
    # pandas les place alors positionnellement, ce qu'on veut.
    #
    # Regle a retenir : dans un constructeur pd.DataFrame({...}, index=...),
    # toute Series venant d'ailleurs doit passer par .to_numpy() ou .values.
    # ======================================================================
    out = pd.DataFrame(
        {
            # pd.to_numeric convertit texte -> nombre.
            # errors="coerce" : ce qui n'est pas convertible devient NaN au
            # lieu de lever une exception. Preferable ici : une ligne
            # corrompue ne doit pas faire echouer 33 000 lignes saines.
            "valeur": pd.to_numeric(df[c["valeur"]], errors="coerce").to_numpy()
            * facteur,
            # .map(fonction) applique la fonction element par element et
            # renvoie une Series. Equivalent d'une comprehension de liste,
            # mais qui preserve l'index.
            "pas": df[c["pas"]].map(_parse_pas).to_numpy(),
        },
        # pd.DatetimeIndex : un index specialise pour les dates. Il debloque
        # .resample(), le decoupage par chaine (df.loc["2025-01"]), les
        # operations de fuseau. C'est lui qui rend pandas interessant ici.
        index=pd.DatetimeIndex(h.to_numpy(), name="horodate", tz="UTC"),
    )
    for cle in ("nature", "vraisemblance", "etape"):
        if COLONNES[cle] in df.columns:
            # .values est l'ancien equivalent de .to_numpy(). Meme raison :
            # eviter l'alignement.
            out[cle] = df[COLONNES[cle]].values

    # --- Dedoublonnage ---
    # .has_duplicates : test rapide sur l'index, evite le travail inutile.
    if out.index.has_duplicates:
        # .map(dict) : traduit chaque valeur via un dictionnaire. Les valeurs
        # absentes du dict deviennent NaN, d'ou le .fillna(0).
        rang = (
            out["etape"].map(PREFERENCE_ETAPE).fillna(0)
            if "etape" in out.columns
            else pd.Series(0, index=out.index)
        )
        # .assign(col=...) ajoute une colonne et renvoie une COPIE. Style
        # chaine, contrairement a df["col"] = ... qui modifie sur place.
        out = out.assign(_r=rang.values).sort_values("_r")
        n_avant = len(out)
        # .duplicated(keep="last") renvoie un masque booleen : True sur les
        # doublons SAUF le dernier. Comme on vient de trier par rang
        # croissant, le dernier est celui de plus forte priorite.
        # Le ~ inverse le masque (equivalent de `not` mais vectorise ;
        # `not` sur une Series leve une exception, il faut ~).
        out = out[~out.index.duplicated(keep="last")].drop(columns="_r")
        warnings.warn(
            f"{n_avant - len(out)} horodates en doublon supprimees "
            "(priorite CORRIGE > COMPLETE > BRUT).",
            stacklevel=2,
        )

    # Le tri est un prerequis de .resample(), .loc[a:b] et .interpolate().
    out = out.sort_index()

    # --- Indice de vraisemblance ---
    if "vraisemblance" in out.columns:
        # .isin(collection) : masque booleen "cette valeur est-elle dans la
        # collection ?". Vectorise, et lisible meme avec plusieurs valeurs.
        suspect = ~out["vraisemblance"].isin(vraisemblance_acceptee)
        # .any() sur un masque : au moins un True. .all() : tous True.
        if suspect.any():
            # .loc[masque, colonne] = valeur : affectation par masque.
            # A privilegier sur out[masque]["valeur"] = ..., qui modifierait
            # une copie temporaire et n'aurait aucun effet (SettingWithCopy).
            out.loc[suspect, "valeur"] = np.nan
            warnings.warn(
                f"{int(suspect.sum())} valeur(s) ecartee(s) sur indice de "
                # .sum() sur un masque booleen compte les True (False=0, True=1).
                f"vraisemblance {sorted(out.loc[suspect, 'vraisemblance'].unique())} "
                "-> traitees comme lacunes.",
                stacklevel=2,
            )
    # subset= : ne supprime la ligne que si CETTE colonne est NaN.
    out = out.dropna(subset=["valeur"])

    # --- Metadonnees ---
    # .attrs est un dictionnaire libre attache au DataFrame. Pratique pour
    # transporter le PRM et le fuseau sans polluer les colonnes.
    # Attention : .attrs n'est pas garanti de survivre a toutes les
    # operations pandas — d'ou les res.attrs.update(...) explicites plus loin.
    out.attrs["prm"] = prm
    out.attrs["unite_source"] = unite
    for cle in ("grandeur_metier", "grandeur_physique"):
        if COLONNES[cle] in df.columns:
            out.attrs[cle] = str(df[COLONNES[cle]].iloc[0]).strip()
    for cle in ("debut_demande", "fin_demande"):
        if COLONNES[cle] in df.columns:
            # .head(1) garde une Series (pour reutiliser _to_datetime qui
            # attend une Series), .iloc[0] extrait ensuite le Timestamp.
            t = _to_datetime(df[COLONNES[cle]].head(1)).iloc[0]
            # Sur un Timestamp isole, pas d'accesseur .dt : les methodes sont
            # directement disponibles. ambiguous=True suffit ici, ces bornes
            # tombent a minuit.
            out.attrs[cle] = (
                t.tz_localize(TZ_LOCALE, ambiguous=True, nonexistent="shift_forward")
                .tz_convert("UTC")
            )
    return out


def _parse_pas(v) -> pd.Timedelta:
    """Parse un pas ISO 8601 (PT30M) ou une valeur en minutes."""
    if isinstance(v, pd.Timedelta):
        return v
    # pd.isna() gere tous les "vides" pandas d'un coup : np.nan, None, NaT
    # (Not a Time), pd.NA. Un simple `v is None` en raterait la plupart.
    if pd.isna(v):
        # NaT = la valeur manquante des types temporels. NaN ne conviendrait
        # pas : la colonne doit rester de type timedelta64.
        return pd.NaT
    s = str(v).strip().upper()
    if s.startswith("P"):
        # pd.Timedelta parse nativement l'ISO 8601 des durees : "PT30M",
        # "PT1H15M", "P1DT2H". Aucune dependance externe necessaire.
        return pd.Timedelta(s)
    return pd.Timedelta(minutes=float(s.replace(",", ".")))


def controler_pas_declare(df: pd.DataFrame) -> pd.DataFrame:
    """Compare le pas declare a l'ecart reel avec l'horodate precedente."""
    # .to_series() transforme l'index en colonne (index ET valeurs identiques),
    # necessaire car .diff() n'existe pas sur un Index.
    # .diff() calcule element[i] - element[i-1]. Le premier vaut NaT.
    delta = df.index.to_series().diff()
    # Soustraction de deux Series de Timedelta : element par element, avec
    # alignement sur l'index (ici identique de part et d'autre, donc sans
    # surprise). Le resultat est un Timedelta signe.
    ecart = delta - df["pas"]
    return pd.DataFrame(
        {
            "pas_declare": df["pas"],
            "delta_observe": delta,
            "ecart": ecart,
            # Comparaison vectorisee : renvoie une Series de booleens, pas un
            # booleen unique. pd.Timedelta(0) est la duree nulle.
            "trou": ecart > pd.Timedelta(0),
            "recouvrement": ecart < pd.Timedelta(0),
        }
    )
    # Ici toutes les Series partagent deja le meme index (celui de df), donc
    # l'alignement joue en notre faveur : pas de .to_numpy() necessaire.


# ---------------------------------------------------------------------------
# 2. Agregation a pas regulier
# ---------------------------------------------------------------------------


def pgcd_pas(pas: pd.Series, cible: pd.Timedelta) -> pd.Timedelta:
    """PGCD des pas presents, borne par le pas cible."""
    # .unique() renvoie un tableau numpy des valeurs distinctes, dans l'ordre
    # d'apparition (contrairement a set(), qui perd l'ordre).
    valides = pas.dropna().unique()
    # On repasse en secondes entieres : np.gcd travaille sur des entiers.
    secondes = [int(pd.Timedelta(p).total_seconds()) for p in valides]
    secondes.append(int(cible.total_seconds()))
    # np.gcd est une "ufunc" (fonction universelle numpy) a deux arguments.
    # .reduce() l'applique en cascade sur toute la liste :
    #   gcd.reduce([1800, 3600, 300]) = gcd(gcd(1800, 3600), 300) = 300
    # Le meme mecanisme existe sur np.add (.reduce = somme), np.maximum, etc.
    return pd.Timedelta(seconds=int(np.gcd.reduce(secondes)))


def agreger(
    df: pd.DataFrame,
    cible: str | pd.Timedelta = "1h",
    seuil_couverture: float = 0.75,
    fenetre: tuple | None = None,
    chemin: str | None = None,
) -> pd.DataFrame:
    """Agrege une courbe a pas variable vers un pas regulier."""
    # pd.Timedelta accepte une chaine ("1h", "30min", "PT30M") ou un
    # Timedelta. Cette ligne rend la fonction tolerante aux deux.
    cible = pd.Timedelta(cible)
    df = df.dropna(subset=["pas"])
    if df.empty:
        raise ValueError(f"Aucune ligne exploitable (colonne 'pas' vide).{chemin.name} lignes dans le DataFrame.")

    pas_fin = pgcd_pas(df["pas"], cible)

    # --- Bornes de la grille ---
    if fenetre is not None:
        # Expression generatrice depaquetee en deux variables. Fonctionne
        # parce qu'elle produit exactement deux elements.
        t0, t1 = (pd.Timestamp(x).tz_convert("UTC") for x in fenetre)
    else:
        # .attrs.get(cle, defaut) : la fenetre demandee si connue, sinon
        # deduite des donnees. (df.index - df["pas"]) soustrait une Series de
        # Timedelta a un DatetimeIndex -> DatetimeIndex des debuts d'intervalle.
        t0 = df.attrs.get("debut_demande", (df.index - df["pas"]).min())
        t1 = df.attrs.get("fin_demande", df.index.max())
    # .ceil / .floor arrondissent un Timestamp a un multiple de duree, comme
    # math.ceil mais sur des dates. Garantit que la grille tombe pile sur des
    # frontieres d'heure.
    t0 = pd.Timestamp(t0).ceil(cible)
    t1 = pd.Timestamp(t1).floor(cible)

    # =====================================================================
    # EXPLOSION : chaque ligne devient n sous-intervalles de duree pas_fin
    # =====================================================================
    # L'idee : une ligne "1500 W pendant 30 min" avec une grille fine a
    # 5 min devient 6 lignes "1500 W" espacees de 5 min. Une fois toutes les
    # lignes ramenees a un pas identique, la moyenne redevient legitime.
    #
    # Ecrit en boucle Python, ce serait 33 000 iterations. En numpy, c'est
    # trois appels vectorises, environ 100 fois plus rapide.

    # Timedelta // Timedelta -> entier. Nombre de sous-pas par ligne.
    n = (df["pas"] // pas_fin).astype("int64").to_numpy()

    # PIEGE : sur un index tz-aware, .to_numpy() renvoie un tableau de type
    # `object` (des Timestamp Python), sur lequel l'arithmetique numpy
    # echoue. tz_localize(None) retire l'etiquette de fuseau sans decaler
    # les valeurs (elles sont deja en UTC) et rend un vrai datetime64.
    fins = df.index.tz_localize(None).to_numpy()
    durees = df["pas"].to_numpy()

    # np.repeat(tableau, n) duplique chaque element autant de fois que dit n.
    #   np.repeat([10, 20], [3, 2]) -> [10, 10, 10, 20, 20]
    # A ne pas confondre avec np.tile, qui repete le motif complet.
    #
    # offsets doit valoir 0,1,2 puis 0,1 dans l'exemple ci-dessus. Astuce
    # classique pour l'obtenir sans boucle :
    #   np.arange(5)               -> [0, 1, 2, 3, 4]      (compteur global)
    #   np.cumsum(n) - n           -> [0, 3]               (debut de chaque bloc)
    #   np.repeat(ces debuts, n)   -> [0, 0, 0, 3, 3]
    #   difference                 -> [0, 1, 2, 0, 1]      (compteur local)
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)

    # Arithmetique vectorisee sur des datetime64/timedelta64 : numpy sait le
    # faire nativement. .to_timedelta64() convertit le Timedelta pandas en
    # son equivalent numpy, requis pour multiplier par un tableau d'entiers.
    debuts_fins = (
        np.repeat(fins, n)          # fin de l'intervalle d'origine
        - np.repeat(durees, n)      # -> debut de l'intervalle d'origine
        + offsets * pas_fin.to_timedelta64()   # -> debut du sous-pas
    )
    valeurs = np.repeat(df["valeur"].to_numpy(), n)

    # On reconstruit une Series pandas : tableau de valeurs + index de dates.
    # tz_localize("UTC") reattache le fuseau retire plus haut.
    fine = pd.Series(
        valeurs, index=pd.DatetimeIndex(debuts_fins).tz_localize("UTC")
    )
    fine = fine[~fine.index.duplicated(keep="last")].sort_index()

    # pd.date_range genere un index de dates regulier.
    #   freq=          : le pas (Timedelta ou chaine "5min")
    #   inclusive="left": borne finale exclue, pour ne pas creer un pas de trop
    #   tz=            : le fuseau, indispensable pour comparer a `fine`
    grille = pd.date_range(t0, t1, freq=pas_fin, inclusive="left", tz="UTC")

    # .reindex(nouvel_index) : reordonne la Series sur l'index fourni.
    # Les dates absentes de `fine` recoivent NaN, les dates en trop sont
    # supprimees. C'est le mecanisme qui materialise les trous : ils
    # deviennent des NaN a une position precise, au lieu d'etre des absences.
    # A distinguer de .loc[] qui leverait KeyError sur une date absente.
    fine = fine.reindex(grille)

    # .resample(duree) regroupe par tranche temporelle. C'est un GroupBy
    # specialise sur DatetimeIndex : il faut donc lui enchainer une fonction
    # d'agregation (.mean(), .sum(), .count(), .max()...).
    n_max = int(cible / pas_fin)
    bloc = fine.resample(cible)

    # .count() compte les valeurs NON-NaN de chaque tranche. Rapporte au
    # nombre attendu, cela donne le taux de couverture reel de chaque heure.
    couverture = bloc.count() / n_max

    # .mean() ignore les NaN par defaut. Sur une heure a moitie couverte,
    # cela revient a supposer que la partie manquante ressemble a la partie
    # mesuree — hypothese acceptable au-dessus du seuil, pas en dessous.
    #
    # .where(condition) garde la valeur quand la condition est vraie et met
    # NaN sinon. Attention au sens : c'est l'inverse de .mask(). Ici, sous le
    # seuil, l'heure redevient une lacune que combler() traitera.
    puissance = bloc.mean().where(couverture >= seuil_couverture)

    # dtype="object" : la colonne contiendra des chaines. Sans cette
    # precision, pandas creerait une colonne de type float et refuserait
    # ensuite d'y ecrire "MESURE".
    origine = pd.Series(pd.NA, index=puissance.index, dtype="object")
    # Affectation par masque booleen : seules les lignes ou le masque est
    # True sont modifiees.
    origine[couverture >= 1.0] = MESURE
    # Le & est le "et" vectorise (le `and` de Python ne marche pas sur les
    # Series). Les parentheses sont OBLIGATOIRES : & est prioritaire sur >=,
    # donc sans elles Python evaluerait seuil & couverture d'abord.
    origine[(couverture >= seuil_couverture) & (couverture < 1.0)] = MESURE_PARTIELLE

    # Ici les trois Series partagent le meme index (issu du meme resample),
    # donc l'alignement de pandas fait exactement ce qu'on veut.
    res = pd.DataFrame(
        {"puissance_W": puissance, "couverture": couverture, "origine": origine}
    )
    # .attrs ne se propage pas a travers resample : on le recopie a la main.
    res.attrs.update(df.attrs)
    res.attrs["pas_grille_fine"] = pas_fin
    res.attrs["pas_cible"] = cible
    return res


# ---------------------------------------------------------------------------
# 3. Comblement des lacunes
# ---------------------------------------------------------------------------


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
        feries |= jours_feries(a)  # |= : union en place de deux ensembles

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
