from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from calculs.constantes import MESURE, MESURE_PARTIELLE




# ---------------------------------------------------------------------------
# 2. Agregation a pas regulier
# ---------------------------------------------------------------------------

"""prend en résultat le dataframe de la fonction lecture
Celle ci a été normalisée. elle contient une colonne "pas" qui contient le pas de temps de chaque ligne, et une colonne "valeur" qui contient la valeur mesurée.
A noter que les valeur qui ne semblaient pas vraisemblables ont été remplacées par des NaN, et que les lignes avec des NaN dans la colonne valeur ont été supprimées.
Donc il peut manquer des lignes dans le dataframe de sortie par rapport au dataframe d'entrée, celles qui ne sont pas vraisemblables.

Certaines métadonnées ont été ajoutées au dataframe, comme le PRM et l'unité de la grandeur physique.
agreger() va créer un nouveau dataframe, avec un pas régulier, et des métadonnées supplémentaires, comme le pas de la grille fine et le pas cible.

"""

def pgcd_pas(pas: pd.Series, cible: pd.Timedelta) -> pd.Timedelta:
    """PGCD des pas presents, borne par le pas cible.
    plus grand pas de temps qui divise exactement tous les pas présents ainsi que le pas cible.
    """
    # .unique() renvoie un tableau numpy des valeurs distinctes, dans l'ordre
    # d'apparition (contrairement a set(), qui perd l'ordre).
    valides = pas.dropna().unique() #tous les pas présents, sans doublons ni NaN. On ne peut pas calculer le pgcd d'un NaN.
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
    
) -> pd.DataFrame:
    """Agrege une courbe a pas variable vers un pas regulier."""
    # pd.Timedelta accepte une chaine ("1h", "30min", "PT30M") ou un
    # Timedelta. Cette ligne rend la fonction tolerante aux deux.
    cible = pd.Timedelta(cible)
    df = df.dropna(subset=["pas"]) #supprime les lignes dans lesquelles la colonne "pas" contient une valeur manquante
    if df.empty:
        raise ValueError(f"Aucune ligne exploitable (colonne 'pas' vide).{df.attrs.get('prm','PRM Inconnu')} lignes dans le DataFrame.")

    pas_fin = pgcd_pas(df["pas"], cible)

    # --- Bornes de la grille --- limites de début et de fin de la grille. Si elles sont fournies, on les utilise. Sinon, on les déduit des données.
    # permet d'étudier et de créer le pas de temps sur une période spécifique, plutôt que sur toute la période des données. Cela peut être utile pour se concentrer sur une période d'intérêt ou pour exclure des périodes de données manquantes ou erronées.
    if fenetre is not None:
        # Expression generatrice depaquetee en deux variables. Fonctionne
        # parce qu'elle produit exactement deux elements.
        #si l'on veut une date de début et de fin de la grille, on peut les passer en argument. Sinon, elles sont déduites des données.
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
    t0 = pd.Timestamp(t0).ceil(cible) #ceil, qui signifie « plafond », arrondit vers la prochaine heure 8 : 17 => 9h
    t1 = pd.Timestamp(t1).floor(cible) #floor, qui signifie « plancher », arrondit vers l'heure précédente 8 : 17 => 8h

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
    fins = df.index.tz_localize(None).to_numpy() # car NumPy manipule parfois difficilement directement ces dates dans les calculs vectorisés.
    durees = df["pas"].to_numpy() # convertit la colonne "pas" en tableau numpy de type timedelta64[ns], pour pouvoir faire des calculs vectorisés.


   
    # np.repeat(tableau, n) duplique chaque element autant de fois que dit n.
    #   np.repeat([10, 20], [3, 2]) -> [10, 10, 10, 20, 20]
    # A ne pas confondre avec np.tile, qui repete le motif complet.
    #
    # ______offsets doit valoir 0,1,2 puis 0,1 dans l'exemple ci-dessus. Astuce________
    # classique pour l'obtenir sans boucle :
    #   np.arange(5)               -> [0, 1, 2, 3, 4]      (compteur global)
    # n.sum() = nombre total de sous-intervalles. On va créer un tableau de cette taille,.
    # np.cumsum(n) -> [3, 5]               (fin de chaque bloc, ici le 3e et le 5e sous-pas 3 valeur 10, 2 valeur 20)
    #   np.cumsum(n) - n   =  [3, 5] - [3, 2]      -> [0, 3]               (debut de chaque bloc) 
    #   np.repeat(ces debuts, n)   -> [0, 0, 0, 3, 3]
    #   difference                 -> [0, 1, 2, 0, 1]      (compteur local)
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) # Elle permet de créer un compteur qui recommence à zéro pour chaque mesure.

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
        valeurs, index=pd.DatetimeIndex(debuts_fins).tz_localize("UTC") # associe les valeurs aux dates correspondantes, en recréant un index de type DatetimeIndex avec le fuseau UTC
    )
    fine = fine[~fine.index.duplicated(keep="last")].sort_index() #supprimer les dates en doubles, remet dans l'ordre croissant. keep="last" : on garde la dernière valeur si doublon, car c'est la plus récente.

    # pd.date_range genere un index de dates regulier.
    #   freq=          : le pas (Timedelta ou chaine "5min")
    #   inclusive="left": borne finale exclue, pour ne pas creer un pas de trop
    #   tz=            : le fuseau, indispensable pour comparer a `fine`
    grille = pd.date_range(t0, t1, freq=pas_fin, inclusive="left", tz="UTC") #sans inclure la date de fin, pour ne pas créer un pas de trop. Le fuseau est indispensable pour comparer à `fine`, qui est en UTC.

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
    puissance = bloc.mean().where(couverture >= seuil_couverture) # si < seuil, met NaN. Si >= seuil, garde la valeur moyenne. Cela permet de ne pas prendre en compte les heures avec trop peu de données pour calculer une moyenne fiable.

    #pour laisser les valeurs à 0 if couverture et bloc / nmax pour conserver les heures à 0 et faire une moyenne en les prenant en compte.


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
