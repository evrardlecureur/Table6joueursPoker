# base_heuristic_bot.py

import numpy as np
from typing import Tuple, List

try:
    from numba import jit
except ImportError:
    def jit(nopython=True, cache=True):
        return lambda f: f

from src.agents.base_agent import BaseAgent
from src.engine.game_state import GameState, ACTION_FOLD, ACTION_CALL, ACTION_RAISE, ROUND_PREFLOP
from src.engine.preflop_equity import PREFLOP_EQUITY
from src.engine.hand_evaluator import evaluate_5_cards, evaluate_7_cards, evaluate_draws


# --- OPTIMISATION JIT NUMBA POUR L'ÉVALUATION POST-FLOP ---

@jit(nopython=True, cache=True)
def evaluate_active_cards(cards: np.ndarray) -> int:
    """
    Évalue les cartes actives (sans les zéros du board non encore distribués).
    Évite de considérer les cartes nulles comme des cartes réelles.
    """
    n = len(cards)
    if n < 5:
        return 0
    elif n == 5:
        return evaluate_5_cards(cards)
    elif n == 6:
        max_score = -1
        temp = np.zeros(5, dtype=np.int32)
        for i in range(6):
            idx = 0
            for j in range(6):
                if j != i:
                    temp[idx] = cards[j]
                    idx += 1
            score = evaluate_5_cards(temp)
            if score > max_score:
                max_score = score
        return max_score
    else:
        # n >= 7, on prend les 7 premières cartes actives
        arr_7 = np.zeros(7, dtype=np.int32)
        for i in range(7):
            arr_7[i] = cards[i]
        return evaluate_7_cards(arr_7)


@jit(nopython=True, cache=True)
def get_surrogate_equity(hole_cards: np.ndarray, board_cards: np.ndarray, num_board: int) -> float:
    """
    Calcule la Surrogate Equity combinant force absolue de la main et potentiel de tirage.
    Retourne une valeur normalisée dans [0.0, 1.0].
    Tout s'exécute en temps constant O(1) sans allocation dynamique sur le tas.
    """
    # Récupération des cartes actives uniquement
    active = np.zeros(7, dtype=np.int32)
    active[0] = hole_cards[0]
    active[1] = hole_cards[1]
    for i in range(num_board):
        active[2 + i] = board_cards[i]
        
    num_active = 2 + num_board
    active_slice = active[:num_active]
    
    # Force absolue (normalisée par la force maximale possible ~10 << 20)
    score = evaluate_active_cards(active_slice)
    abs_strength = float(score) / float(10 << 20)
    
    if num_board < 5:
        # Potentiel de tirage (Flop/Turn)
        draws = evaluate_draws(active_slice)
        norm_outs = draws[3]  # normalized_outs = min(15.0, outs) / 15.0
        
        # Combinaison : max de la force actuelle et du potentiel du tirage
        equity = max(abs_strength, norm_outs * 0.6 + abs_strength * 0.4)
    else:
        # River : pas de tirage possible, seule la force absolue compte
        equity = abs_strength
        
    return equity


@jit(nopython=True, cache=True)
def postflop_score_to_percentile(score: float) -> float:
    """
    Mappe de manière continue et en temps O(1) un score de Surrogate Equity
    vers un percentile réaliste dans la distribution globale des mains post-flop.
    """
    if score < 0.2:
        # Hauteur : représente les 30% pires mains
        return (score / 0.2) * 0.3
    elif score < 0.3:
        # Une Paire : représente 45% de la distribution (percentile 30% à 75%)
        return 0.3 + ((score - 0.2) / 0.1) * 0.45
    elif score < 0.4:
        # Double Paire : représente 15% (percentile 75% à 90%)
        return 0.75 + ((score - 0.3) / 0.1) * 0.15
    elif score < 0.6:
        # Brelan, Quinte, Couleur : représente 7% (percentile 90% à 97%)
        return 0.9 + ((score - 0.4) / 0.2) * 0.07
    else:
        # Full, Carré, Quinte Flush : représente les top 3% des mains (percentile 97% à 100%)
        return 0.97 + ((score - 0.6) / 0.4) * 0.03


# --- PRÉ-CALCUL DU PERCENTILE PRÉFLOP DES MAINS DE DÉPART ---

def _init_preflop_percentiles() -> dict:
    """
    Génère le dictionnaire de percentiles préflop pondérés par la probabilité
    combinatoire d'obtenir chaque main (1326 combos au total).
    S'exécute en O(1) au chargement du module.
    """
    combos_weights = {}
    for key in PREFLOP_EQUITY.keys():
        if len(key) == 2:  # Paire (ex: AA) -> 6 combinaisons
            combos_weights[key] = 6
        elif key.endswith('s'):  # Suited (ex: AKs) -> 4 combinaisons
            combos_weights[key] = 4
        elif key.endswith('o'):  # Offsuit (ex: AKo) -> 12 combinaisons
            combos_weights[key] = 12
            
    # Tri des mains par equity croissante
    sorted_hands = sorted(PREFLOP_EQUITY.items(), key=lambda x: x[1])
    
    cum_weight = 0.0
    percentiles = {}
    total_combos = 1326.0
    
    for hand, eq in sorted_hands:
        w = combos_weights.get(hand, 0)
        cum_weight += w
        percentiles[hand] = cum_weight / total_combos
        
    return percentiles


PREFLOP_PERCENTILES = _init_preflop_percentiles()


def get_preflop_key(c1: int, c2: int) -> str:
    """
    Extrait la clé de la main préflop (ex: 'AKo', '72s', 'QQ') à partir des entiers de cartes.
    """
    v1 = (c1 >> 8) & 0xF
    v2 = (c2 >> 8) & 0xF
    s1 = (c1 >> 12) & 0xF
    s2 = (c2 >> 12) & 0xF
    
    ranks_str = "23456789TJQKA"
    r1 = ranks_str[v1]
    r2 = ranks_str[v2]
    
    if v1 == v2:
        return f"{r1}{r2}"
    if v1 > v2:
        suffix = "s" if s1 == s2 else "o"
        return f"{r1}{r2}{suffix}"
    else:
        suffix = "s" if s1 == s2 else "o"
        return f"{r2}{r1}{suffix}"


# --- CLASSE MÈRE BASEHEURISTICBOT ---

class BaseHeuristicBot(BaseAgent):
    """
    Classe mère simulant le joueur moyen d'après ses statistiques de vérité terrain.
    Prend ses décisions en temps constant O(1) par mapping de percentiles de mains.
    """
    def __init__(self, name: str, vpip: float, pfr: float, three_bet: float, fold_to_cbet: float, wtsd: float, tilt_factor: float = 0.05):
        super().__init__(name)
        self.vpip = vpip
        self.pfr = pfr
        self.three_bet = three_bet
        self.fold_to_cbet = fold_to_cbet
        self.wtsd = wtsd
        self.tilt_factor = tilt_factor

    def select_action(self, game_state: GameState) -> Tuple[int, int]:
        valid_actions = game_state.get_valid_actions()
        if not valid_actions:
            return ACTION_FOLD, 0
            
        p = game_state.players[game_state.current_player_idx]
        to_call = game_state.highest_bet - p.current_bet
        
        # Dictionnaire pour un accès rapide aux actions
        action_map = {act[0]: (act[1], act[2]) for act in valid_actions}
        
        # SPACE KO: Ajout de 12 500 jetons au Pot_Size perçu si on couvre un joueur à tapis
        bounty_value = 0
        if to_call > 0:
            for opp in game_state.players:
                if opp.seat_id != p.seat_id and opp.in_play and opp.is_all_in:
                    if p.chips + p.current_bet >= opp.total_contribution:
                        bounty_value += 12500
                        break
        perceived_pot = game_state.pot + bounty_value

        
        # --- PHASE PRÉFLOP ---
        if game_state.current_round == ROUND_PREFLOP:
            # Récupération de la main préflop et de son percentile
            hand_key = get_preflop_key(p.cards[0], p.cards[1])
            true_percentile = PREFLOP_PERCENTILES.get(hand_key, 0.5)
            
            # Injection de bruit cognitif (simule l'appréciation subjective et le tilt)
            perceived_percentile = np.clip(true_percentile + np.random.normal(0, self.tilt_factor), 0.0, 1.0)
            
            # 1. Aucun enjeu majeur (Pot non ouvert ou limps uniquement)
            if to_call == 0:
                if perceived_percentile >= 1.0 - self.pfr:
                    # Relance d'ouverture standard (3 BB + 1 BB par limper)
                    if ACTION_RAISE in action_map:
                        min_raise, max_raise = action_map[ACTION_RAISE]
                        num_limpers = sum(1 for pl in game_state.players if pl.in_play and pl.total_contribution == game_state.bb and pl.seat_id != p.seat_id)
                        raise_amount = min(max_raise, max(min_raise, game_state.bb * (3 + num_limpers)))
                        return ACTION_RAISE, raise_amount
                
                if perceived_percentile >= 1.0 - self.vpip:
                    # Limp/Call standard
                    if ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                        
                # Si c'est gratuit de checker
                if ACTION_CALL in action_map and action_map[ACTION_CALL][0] == 0:
                    return ACTION_CALL, 0
                return ACTION_FOLD, 0
                
            # 2. Il y a une relance devant nous
            else:
                # 3-bet automatique avec le top range
                if perceived_percentile >= 1.0 - self.three_bet:
                    if ACTION_RAISE in action_map:
                        min_raise, max_raise = action_map[ACTION_RAISE]
                        raise_amount = min(max_raise, max(min_raise, to_call * 3))
                        return ACTION_RAISE, raise_amount
                
                # Défense de blindes / calling standard préflop proportionnel à VPIP et inversement au montant
                defending_multiplier = max(0.2, 0.6 - (to_call / game_state.bb) * 0.1)
                if perceived_percentile >= 1.0 - (self.vpip * defending_multiplier):
                    if ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                        
                return ACTION_FOLD, 0

        # --- PHASES POSTFLOP (Flop, Turn, River) ---
        else:
            # Calcul de la Surrogate Equity via Numba
            true_equity = get_surrogate_equity(p.cards, game_state.board, game_state.current_round + 2) # Flop=1 -> 3 board, Turn=2 -> 4 board, River=3 -> 5 board
            
            # Injection de bruit cognitif post-flop
            perceived_equity = np.clip(true_equity + np.random.normal(0, self.tilt_factor), 0.0, 1.0)
            
            # Conversion de la Surrogate Equity perçue en percentile
            P = postflop_score_to_percentile(perceived_equity)
            
            # 1. Personne n'a misé (to_call == 0) -> Choix entre Check et Bet
            if to_call == 0:
                if P >= 1.0 - self.pfr:
                    # On mise en fonction de la force perçue de notre main
                    if ACTION_RAISE in action_map:
                        min_raise, max_raise = action_map[ACTION_RAISE]
                        bet_fraction = 0.66 if P >= 0.90 else 0.40  # Grosse mise avec un monstre, moyenne avec top paire/tirage
                        bet_size = min(max_raise, max(min_raise, int(perceived_pot * bet_fraction)))
                        return ACTION_RAISE, bet_size
                
                # Check gratuit
                if ACTION_CALL in action_map:
                    return ACTION_CALL, 0
                return ACTION_FOLD, 0
                
            # 2. Face à une mise adverse (to_call > 0)
            else:
                if game_state.current_round < ROUND_RIVER:
                    # Flop et Turn : Dicté par Fold to C-Bet
                    # Ajustement de la sticky-ness en fonction du bet-sizing adverse et des bounties
                    effective_fold_threshold = self.fold_to_cbet * (0.5 + to_call / perceived_pot)
                    effective_fold_threshold = min(0.85, max(0.20, effective_fold_threshold))
                    
                    if P < effective_fold_threshold:
                        return ACTION_FOLD, 0
                        
                    # Choix entre Call et Relance
                    # On relance avec le haut de notre range de continuation (ratio PFR/VPIP)
                    raise_ratio = min(0.5, self.pfr / max(0.1, self.vpip))
                    raise_threshold = 1.0 - (1.0 - effective_fold_threshold) * raise_ratio
                    
                    if P >= raise_threshold and ACTION_RAISE in action_map:
                        min_raise, max_raise = action_map[ACTION_RAISE]
                        raise_amount = min(max_raise, max(min_raise, to_call * 3))
                        return ACTION_RAISE, raise_amount
                        
                    if ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                        
                else:
                    # River : Dicté par WTSD (Went to Showdown)
                    # Plus le WTSD est élevé, plus le joueur paye les mises à la river
                    effective_river_fold_threshold = 1.0 - self.wtsd * (1.5 - to_call / perceived_pot)
                    effective_river_fold_threshold = min(0.90, max(0.15, effective_river_fold_threshold))
                    
                    if P < effective_river_fold_threshold:
                        return ACTION_FOLD, 0
                        
                    # Relance uniquement si on a un monstre absolu (percentile >= 95%)
                    if P >= 0.95 and ACTION_RAISE in action_map:
                        min_raise, max_raise = action_map[ACTION_RAISE]
                        raise_amount = min(max_raise, max(min_raise, to_call * 3))
                        return ACTION_RAISE, raise_amount
                        
                    if ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                        
        return ACTION_FOLD, 0
