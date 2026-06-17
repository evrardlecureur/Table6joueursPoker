# winamax_metrics.py

"""
Configuration de la Vérité Terrain pour le format SPACE KO (MTT PKO Micro Buy-in 0.50€) sur Winamax.
Ces statistiques représentent les moyennes (mean) et écarts-types (std_dev) de la population
des tables 6-Max micro-limites en format Bounty, collectées à partir de données réelles de trackers.

En format PKO, les joueurs chassent les primes : le VPIP moyen est naturellement plus élevé 
et le Fold to C-Bet plus bas que sur un tournoi standard ou du Cash Game.
"""

WINAMAX_MICRO_POOL = {
    "VPIP": {
        "mean": 0.28,       # Fréquence moyenne globale de participation volontaire au pot (Élevé en PKO)
        "std_dev": 0.06     # Écart-type
    },
    "PFR": {
        "mean": 0.18,       # Fréquence moyenne globale de relance préflop
        "std_dev": 0.05     # Écart-type
    },
    "3BET": {
        "mean": 0.075,      # Fréquence moyenne globale de 3-bet
        "std_dev": 0.025    # Écart-type
    },
    "FOLD_TO_CBET": {
        "mean": 0.42,       # Fréquence moyenne globale de fold face à un continuation bet au flop (Bas en PKO)
        "std_dev": 0.08     # Écart-type
    },
    "WTSD": {
        "mean": 0.28,       # Fréquence moyenne de passage à l'abattage (Went To Showdown)
        "std_dev": 0.04     # Écart-type (les calling stations à >36% et les joueurs serrés/nits à <20%)
    }
}
