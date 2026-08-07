# traitement_cdc_acc

Pipeline de traitement des **courbes de charge (CDC)** Enedis pour un ensemble de Points de Livraison (PDL) : lecture des flux bruts, agrégation au pas de temps cible, complément des lacunes de mesure, puis export des courbes consolidées.

> ⚠️ Ce README a été généré à partir de l'arborescence du projet, du nom des modules et des fichiers de sortie fournis. Il décrit l'intention probable de chaque étape ; à ajuster si le comportement réel des scripts diffère.

## Objectif

À partir des flux de courbes de charge fournis par Enedis (via les données SGE), le projet :

1. lit les fichiers de courbe de charge bruts d'un ou plusieurs PDL,
2. agrège les points au pas de temps souhaité (ex. 30 min → 1 h),
3. détecte et comble les lacunes de mesure (points manquants ou `couverture < 1`),
4. exporte, pour chaque PDL, une courbe de charge propre et continue (`sortie_<PDL>.csv`).

## Structure du projet

```
traitement_cdc_acc/
├── data/                         # Données d'entrée (flux Enedis bruts)
├── sortie_total_v1/               # Résultats consolidés (courbes de charge nettoyées)
├── src/
│   ├── calculs/
│   │   ├── constantes.py          # Constantes du projet (pas de temps, chemins, seuils…)
│   │   ├── E1_lecture.py          # Lecture et parsing des fichiers bruts Enedis
│   │   ├── E2_agregation_pas.py   # Agrégation/rééchantillonnage au pas de temps cible
│   │   ├── E3_complement_lacunes.py # Détection et complément des données manquantes
│   │   ├── E4_export.py           # Export des courbes de charge finales (sortie_*.csv)
│   │   └── selection_dossier.py   # Sélection du/des dossier(s) de données à traiter
│   ├── sortie/                    # Résultats intermédiaires/finaux du traitement
├── __init__.py
├── __main__.py
├── main.py                        # Point d'entrée du pipeline
├── tests/                         # Tests unitaires
├── enedis-guide-de-flux-des-abonnements-...  # Documentation Enedis de référence
├── .gitignore
└── README.md
```

## Format des données

### Entrée
Flux de courbe de charge Enedis (format `.csv`), conformes au guide de flux d'abonnement Enedis.

### Sortie (`sortie_<PDL>.csv`)
Fichier `;`-séparé, une ligne par point temporel :

| Colonne          | Description                                                          |
|------------------|-----------------------------------------------------------------------|
| `horodate_locale`| Horodatage local avec offset (ex. `2024-06-23 00:00:00+02:00`)       |
| `puissance_W`    | Puissance moyenne sur le pas, en watts (décimale à virgule)          |
| `couverture`     | Taux de couverture de la mesure sur le pas (`1,000` = complet)       |
| `origine`        | Origine du point : `MESURE`, `MESURE_PARTIELLE`, ou valeur comblée   |

Exemple :
```
horodate_locale;puissance_W;couverture;origine
2024-06-23 00:00:00+02:00;1750,000;1,000;MESURE
2026-06-22 23:00:00+02:00;2090,909;0,917;MESURE_PARTIELLE
```

#### Explication de l'origine


7 sorties de la colone `origine` sont possibles

1. MESURE = "MESURE" simplement la mesure relevée par Enedis.
2. MESURE_PARTIELLE = "MESURE_PARTIELLE" Dans le cas ou il manquerait une période sur l'heure. En echo avec  `couverture` qui donne la couverure de l'heure
3. INTERP_COURTE = "INTERP_COURTE" Si l'interruption de consommation est inférieur à 3h, on fait une moyenne de la valeur avant et après la période de 3 h pour la puissance
4. PROFIL_JOUR_TYPE = "PROFIL_JOUR_TYPE" Dans le cas d'une interruption entre 3h et 10 jours, on fait la médiane de la puissance des 4 périodes des jours identique (OUVRE, SAMEDI, DIMANCHE) les plus proches
5. ANNEE_N1 = "ANNEE_N-1" # si l'interruption est supérieur à 10 jours, on prend la valeur de l'année d'avant (ou d'après si pas avant)
6. ZERO_FORCE = "ZERO_FORCE" Aucune solution d'évaluation de donnée ne fonctionne, on est obligé de mettre 0
7. MISE_A_ZERO_OBLIGATOIRE = "MISE_A_ZERO_CONSIGNE" #nous avons choisi de mettre 0 parout ou nous n'avons pas les données, exception faite de MESURE_PARTIELLE
````

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # sous Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py
```

Selon la configuration (`selection_dossier.py` / `constantes.py`), le script :
- sélectionne le dossier de données à traiter (`data/`),
- exécute successivement lecture → agrégation → complément des lacunes → export,
- écrit les fichiers `sortie_<PDL>.csv` dans le dossier de sortie configuré.

## Tests

```bash
pytest tests/
```

## Dépendances

Voir `requirements.txt`. Bibliothèques principales utilisées : `pandas` pour la manipulation des courbes de charge, `numpy` pour les calculs numériques, gestion des fuseaux horaires pour les horodates avec offset.

## À compléter

- Détail précis des règles de complément des lacunes (interpolation, moyenne glissante, valeur précédente, etc.)
- Pas de temps cible configurable (30 min, 1 h…)
- Format exact des fichiers d'entrée Enedis attendus