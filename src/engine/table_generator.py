# table_generator.py

from typing import List
from src.engine.winamax_metrics import WINAMAX_MICRO_POOL
from src.engine.base_heuristic_bot import BaseHeuristicBot

def clip_stat(val: float, min_val: float = 0.02, max_val: float = 0.98) -> float:
    """Limite les statistiques dans une plage réaliste pour éviter les comportements extrêmes impossibles."""
    return max(min_val, min(max_val, val))

# Extraction des valeurs moyennes et écarts-types de la Vérité Terrain
V_MEAN, V_STD = WINAMAX_MICRO_POOL["VPIP"]["mean"], WINAMAX_MICRO_POOL["VPIP"]["std_dev"]
P_MEAN, P_STD = WINAMAX_MICRO_POOL["PFR"]["mean"], WINAMAX_MICRO_POOL["PFR"]["std_dev"]
T_MEAN, T_STD = WINAMAX_MICRO_POOL["3BET"]["mean"], WINAMAX_MICRO_POOL["3BET"]["std_dev"]
F_MEAN, F_STD = WINAMAX_MICRO_POOL["FOLD_TO_CBET"]["mean"], WINAMAX_MICRO_POOL["FOLD_TO_CBET"]["std_dev"]
W_MEAN, W_STD = WINAMAX_MICRO_POOL["WTSD"]["mean"], WINAMAX_MICRO_POOL["WTSD"]["std_dev"]


# --- INSTANCIATION DES 5 PROFILS STÉRÉOTYPÉS DE MICRO-LIMITES ---

class NitBot(BaseHeuristicBot):
    """
    Profil 'The Nit' (Serré-Passif/Serré-Agressif extrême) :
    Joue très peu de mains, ne prend aucun risque et jette sa main au moindre doute.
    """
    def __init__(self, name: str = "The_Nit"):
        vpip = clip_stat(V_MEAN - 1.5 * V_STD)       # 24% - 9% = 15%
        pfr = clip_stat(P_MEAN - 1.2 * P_STD)        # 18% - 6% = 12%
        three_bet = clip_stat(T_MEAN - 1.4 * T_STD)  # 7.5% - 3.5% = 4%
        fold_to_cbet = clip_stat(F_MEAN + 1.5 * F_STD) # 50% + 12% = 62%
        wtsd = clip_stat(W_MEAN - 1.5 * W_STD)       # 28% - 6% = 22%
        
        # Bruit cognitif très faible : joueur extrêmement discipliné et prévisible
        super().__init__(name, vpip, pfr, three_bet, fold_to_cbet, wtsd, tilt_factor=0.02)


class CallingStationBot(BaseHeuristicBot):
    """
    Profil 'The Calling Station' (Large-Passif) :
    Aime voir beaucoup de flops, relance très peu, mais paye tout jusqu'à la river.
    """
    def __init__(self, name: str = "Calling_Station"):
        vpip = clip_stat(V_MEAN + 1.5 * V_STD)       # 24% + 9% = 33%
        pfr = clip_stat(P_MEAN - 1.6 * P_STD)        # 18% - 8% = 10%
        three_bet = clip_stat(T_MEAN - 1.8 * T_STD)  # 7.5% - 4.5% = 3%
        fold_to_cbet = clip_stat(F_MEAN - 1.8 * F_STD) # 50% - 14.4% = 35.6%
        wtsd = clip_stat(W_MEAN + 2.0 * W_STD)       # 28% + 8% = 36%
        
        # Bruit cognitif modéré : suit bêtement ses tirages et ses paires sans logique
        super().__init__(name, vpip, pfr, three_bet, fold_to_cbet, wtsd, tilt_factor=0.05)


class ManiacBot(BaseHeuristicBot):
    """
    Profil 'The Maniac' (Large-Agressif extrême) :
    Surjoue ses mains, bluffe à outrance, 3-bet énormément et refuse de folder.
    """
    def __init__(self, name: str = "The_Maniac"):
        vpip = clip_stat(V_MEAN + 2.0 * V_STD)       # 24% + 12% = 36%
        pfr = clip_stat(P_MEAN + 2.0 * P_STD)        # 18% + 10% = 28%
        three_bet = clip_stat(T_MEAN + 1.8 * T_STD)  # 7.5% + 4.5% = 12%
        fold_to_cbet = clip_stat(F_MEAN - 1.2 * F_STD) # 50% - 9.6% = 40.4%
        wtsd = clip_stat(W_MEAN + 0.5 * W_STD)       # 28% + 2% = 30%
        
        # Bruit cognitif très élevé : comportement irrationnel, agressivité aléatoire
        super().__init__(name, vpip, pfr, three_bet, fold_to_cbet, wtsd, tilt_factor=0.15)


class TAGBot(BaseHeuristicBot):
    """
    Profil 'The TAG' (Tight-Aggressive) :
    Le bon joueur régulier standard des micro-limites. Joue serré, agressif et structuré.
    """
    def __init__(self, name: str = "The_TAG"):
        vpip = clip_stat(V_MEAN - 0.5 * V_STD)       # 24% - 3% = 21%
        pfr = clip_stat(P_MEAN + 0.0 * P_STD)        # 18%
        three_bet = clip_stat(T_MEAN + 0.4 * T_STD)  # 7.5% + 1% = 8.5%
        fold_to_cbet = clip_stat(F_MEAN + 0.2 * F_STD) # 50% + 1.6% = 51.6%
        wtsd = clip_stat(W_MEAN - 0.5 * W_STD)       # 28% - 2% = 26%
        
        # Bruit cognitif minimal : joueur lucide et concentré sur sa force réelle
        super().__init__(name, vpip, pfr, three_bet, fold_to_cbet, wtsd, tilt_factor=0.01)


class FishBot(BaseHeuristicBot):
    """
    Profil 'The Fish' (Large-Passif typique / Récréatif standard) :
    Le joueur récréatif classique de Winamax micro-stakes. Erreurs de sizing, passif mais collant.
    """
    def __init__(self, name: str = "The_Fish"):
        vpip = clip_stat(V_MEAN + 1.0 * V_STD)       # 24% + 6% = 30%
        pfr = clip_stat(P_MEAN - 1.2 * P_STD)        # 18% - 6% = 12%
        three_bet = clip_stat(T_MEAN - 1.0 * T_STD)  # 7.5% - 2.5% = 5%
        fold_to_cbet = clip_stat(F_MEAN - 0.5 * F_STD) # 50% - 4% = 46%
        wtsd = clip_stat(W_MEAN + 1.0 * W_STD)       # 28% + 4% = 32%
        
        # Bruit cognitif élevé : beaucoup d'erreurs d'appréciation post-flop
        super().__init__(name, vpip, pfr, three_bet, fold_to_cbet, wtsd, tilt_factor=0.09)


# --- FONCTION DE GÉNÉRATION D'OPPOSANTS POUR LA TABLE 6-MAX ---

def get_macro_adversaries() -> List[BaseHeuristicBot]:
    """
    Génère et retourne la liste des 5 Macro-Adversaires heuristiques.
    Ces 5 profils occuperont les sièges libres de la table d'entraînement (6-Max),
    permettant à notre agent RL principal de s'entraîner de manière asymétrique.
    """
    return [
        NitBot("Nit_Bot_1"),
        CallingStationBot("Station_Bot_2"),
        ManiacBot("Maniac_Bot_3"),
        TAGBot("TAG_Bot_4"),
        FishBot("Fish_Bot_5")
    ]
