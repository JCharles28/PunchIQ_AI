import pandas as pd
import random
from feature_extraction import extraire_features

def generer_paires(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    lignes = []

    noms_disponibles = df['name'].tolist()

    for _ in range(n):
        nomA, nomB = random.sample(noms_disponibles, 2)

        try:
            features = extraire_features(nomA, nomB, df)

            # Vainqueur simulé (règle simple) : celui avec le plus de victoires
            vainqueur = "A" if df[df["name"] == nomA]["wins"].values[0] >= df[df["name"] == nomB]["wins"].values[0] else "B"

            ligne = features + [vainqueur, nomA, nomB]  # On garde les noms aussi
            lignes.append(ligne)

        except Exception as e:
            print(f"Erreur avec {nomA} vs {nomB} : {e}")
            continue

    # Colonnes
    colonnes_features = [
        "age_diff", "ko_rate_diff", "height_diff", "reach_diff",
        "wins_diff", "looses_diff", "draws_diff", "stance_diff"
    ]
    colonnes_finales = colonnes_features + ["winner", "boxeur_A", "boxeur_B"]

    df_final = pd.DataFrame(lignes, columns=colonnes_finales)
    return df_final

if __name__ == "__main__":
    df_clean = pd.read_csv("data/fighters_cleaned.csv")
    df_train = generer_paires(df_clean, n=1000)
    df_train.to_csv("data/train_winner.csv", index=False)
    print("✅ Dataset d'entraînement généré dans data/train_winner.csv")
