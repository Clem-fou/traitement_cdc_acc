
from future import annotations
import numpy as np
import pandas as pd
import warnings

# Nom de fuseau IANA. pandas ne connait pas "heure de Paris" : il lui faut
# la cle de la base tzdata, qui embarque l'historique complet des regles de
# changement d'heure. C'est cette base qui sait que le 27/10/2024 dure 25 h.
from calculs.constantes import TZ_LOCALE, COLONNES, PREFERENCE_ETAPE



# ---------------------------------------------------------------------------
# 1. Lecture
# ---------------------------------------------------------------------------



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
        tete = f.read(4096) # les premiers 4096 octets du fichier. Suffisant pour detecter l'encodage, et evite de charger un fichier entier.
    # BOM : 3 octets que Windows ajoute parfois en tete de fichier UTF-8.
    # L'encodage "utf-8-sig" les consomme ; "utf-8" les laisserait dans le
    # nom de la premiere colonne.
    if tete.startswith(b"\xef\xbb\xbf"): #3 octets de la signature UTF-8
        return "utf-8-sig"
    try:
        # Si ces octets ne forment pas de l'UTF-8 valide, .decode() leve
        # UnicodeDecodeError. C'est un test fiable : l'UTF-8 a une structure
        # tres contrainte, du texte latin-1 accentue echoue presque toujours.
        tete.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"  # l'encodage Windows occidental, defaut d'Excel FR au cas ou l'UTF-8 echoue. Il est compatible avec ISO-8859-1, mais pas avec UTF-8.


def _detecter_separateur(chemin, encoding: str) -> str:
    """Separateur le plus frequent dans l'en-tete parmi ; \\t ,"""
    with open(chemin, encoding=encoding) as f:
        entete = f.readline()
    comptes = {s: entete.count(s) for s in (";", "\t", ",")} #séparateur le plus fréquent dans l'en-tête parmi ; \t , (tabulation)
    # max(dict, key=dict.get) renvoie la CLE dont la valeur est maximale.
    # Sans le parametre key, max() sur un dict renverrait la plus grande cle
    # au sens alphabetique — pas du tout ce qu'on veut.
    return max(comptes, key=comptes.get) # on veut le séparateur le plus fréquent, pas la plus grande cle alphabetiquement


def _to_datetime(serie: pd.Series) -> pd.Series: # _to_datetime : fonction interne, pas pour l'appelant
    """Parse des horodates ISO ou francaises."""
    # .dropna() retire les valeurs manquantes ; .iloc[0] prend le premier
    # element PAR POSITION. Distinction essentielle en pandas :
    #   .iloc[0] = premier element (position)
    #   .loc[0]  = element dont l'etiquette d'index vaut 0
    # Sur un DataFrame filtre, les etiquettes ont des trous : .loc[0] peut
    # lever KeyError la ou .iloc[0] fonctionne toujours.
    echantillon = str(serie.dropna().iloc[0]).strip()
    iso = len(echantillon) >= 4 and echantillon[:4].isdigit() # un format ISO commence par l'annee sur 4 chiffres, un format francais commence par le jour (1 ou 2 chiffres) ou le mois (1 ou 2 chiffres). On peut donc detecter le format en regardant les 4 premiers caracteres.

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

    # enlève les puissances qui ne sont pas en W,  que la puissance apparente, pas de puissance réactive
    if c["grandeur_physique"] and c["prm"] in df.columns:
        df = df[
            df[c["grandeur_physique"]]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq("PA")
        ]
        raise ValueError(
        f"Aucune ligne de puissance active (PA) n'a été trouvée.{df[c['prm']].iloc[0]}"
    ) 

    
    # pour les tarifs jaunes, null est rentré et pas 0 : permet d'uniformiser les données pour le calcul de l'énergie
    if c["vraisemblance"] in df.columns:
       df[c["vraisemblance"]] = df[c["vraisemblance"]].fillna(0)    

    # --- Unite ---
    unite = str(df[c["unite"]].iloc[0]).strip() if c["unite"] in df.columns else "W"
    # dict.get(cle, defaut) : pas de KeyError si l'unite est inconnue.
    facteurs_pris_en_compte = {"W": 1.0, "kW": 1000.0, "VA": 1.0, "kVA": 1000.0}
    facteur = facteurs_pris_en_compte.get(unite, 1.0)
    if unite not in facteurs_pris_en_compte:
        warnings.warn(f"Unité inconnue : {unite!r}", stacklevel=2) #!r pour voir les guillemets et les espaces parasites


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
         
         
         #vérifier que l'ordre n'est pas modifier.

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
            "pas": df[c["pas"]].map(_parse_pas).to_numpy(), # renvoie un Timedelta, qui est le type pandas pour les durées. 
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
            out[cle] = df[COLONNES[cle]].values # renvoie un tableau numpy

    # --- Dedoublonnage --- permet de garder la ligne de plus forte priorite (CORRIGE > COMPLETE > BRUT)
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
        out = out.assign(_r=rang.values).sort_values("_r") #trie par ordre croissant dans la colonne _r
        n_avant = len(out)
        # .duplicated(keep="last") renvoie un masque booleen : True sur les
        # doublons SAUF le dernier. Comme on vient de trier par rang
        # croissant, le dernier est celui de plus forte priorite.
        # Le ~ inverse le masque (equivalent de `not` mais vectorise ;
        # `not` sur une Series leve une exception, il faut ~).
        out = out[~out.index.duplicated(keep="last")].drop(columns="_r") #on garde la dernière ligne (classé dans l'ordre croissant)
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
        suspect = ~out["vraisemblance"].isin(vraisemblance_acceptee) # masque booleen pour identifier les valeurs non acceptees
        # .any() sur un masque : au moins un True. .all() : tous True.
        if suspect.any(): #renvoie true si au moins une valeur est suspecte
            # .loc[masque, colonne] = valeur : affectation par masque.
            # A privilegier sur out[masque]["valeur"] = ..., qui modifierait
            # une copie temporaire et n'aurait aucun effet (SettingWithCopy).
            out.loc[suspect, "valeur"] = np.nan # remplace les valeurs suspectes par NaN (PAS PRISE EN COMPTE DANS LES CALCULS)
            warnings.warn(
                f"{int(suspect.sum())} valeur(s) ecartee(s) sur indice de "
                # .sum() sur un masque booleen compte les True (False=0, True=1).
                f"vraisemblance {sorted(out.loc[suspect, 'vraisemblance'].unique())} "
                "-> traitees comme lacunes.",
                stacklevel=2,
            )

    # subset= : ne supprime la ligne que si CETTE colonne est NaN. SUPPRESSION DE LIGNE SI LA VALEUR EST MANQUANTE, PAS DE SUPPRESSION SI LE PAS EST MANQUANT (PAS DE SUPPRESSION DE LIGNE SI LE PAS EST MANQUANT)
    out = out.dropna(subset=["valeur"])

    # --- Metadonnees --- 
    # transporter le PRM et le fuseau sans polluer les colonnes.
    # Attention : .attrs n'est pas garanti de survivre a toutes les
    #permet d’associer des métadonnées à tes données sans créer de nouvelles colonnes.
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
    """Parse un pas ISO 8601 (PT30M) ou une valeur en minutes.
    lire une valeur, reconnaître son format et la convertir dans un type utilisable par Python
    """ 
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
    return pd.Timedelta(minutes=float(s.replace(",", "."))) # renvoie un Timedelta, qui est le type pandas pour les durées. On peut ensuite faire des calculs avec ces objets, comme additionner ou soustraire des dates.


def controler_pas_declare(df: pd.DataFrame) -> pd.DataFrame: #reçoit déjà un DataFrame normalisé, avec l'index horodate et la colonne pas
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

