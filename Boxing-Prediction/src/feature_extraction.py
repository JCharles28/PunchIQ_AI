import pandas as pd
import numpy as np

def extraire_features(nom_A: str, nom_B: str, df: pd.DataFrame) -> list:
    """
    Prend deux noms de boxeurs et le DataFrame nettoyé.
    Retourne un vecteur de features comparées (list de floats).
    """
    # Rechercher les deux boxeurs
    boxeur_A = df[df['name'].str.lower() == nom_A.lower()]
    boxeur_B = df[df['name'].str.lower() == nom_B.lower()]

    # Vérifier qu'ils existent
    if boxeur_A.empty or boxeur_B.empty:
        raise ValueError("Un ou les deux boxeurs ne sont pas présents dans le dataset.")

    # Convertir en dicts pour simplifier l'accès
    a = boxeur_A.iloc[0]
    b = boxeur_B.iloc[0]

    # Calculer les features comparées
    features = {
        "age_diff": a['age'] - b['age'],
        "ko_rate_diff": a['ko_rate'] - b['ko_rate'],
        "height_diff": a['height_m'] - b['height_m'],
        "reach_diff": a['reach_cm'] - b['reach_cm'],
        "wins_diff": a['wins'] - b['wins'],
        "looses_diff": a['looses'] - b['looses'],
        "draws_diff": a['draws'] - b['draws'],
        "stance_diff": 1 if a['stance'] != b['stance'] else 0,  # 1 si garde différente
    }

    return list(features.values())
