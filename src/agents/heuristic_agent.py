import numpy as np
from typing import Tuple, List
from src.agents.base_agent import BaseAgent
from src.engine.game_state import GameState, ACTION_FOLD, ACTION_CALL, ACTION_RAISE, ROUND_PREFLOP
from src.engine.card import make_card, card_to_str
from src.engine.hand_evaluator import evaluate_7_cards

class HeuristicAgent(BaseAgent):
    """
    Un agent de poker basé sur des règles heuristiques simples.
    Il simule le comportement typique d'un joueur récréatif de micro-limites (Loose-Passive ou "Fish") :
    - Préflop : Limpe beaucoup, paye les petites relances avec des mains spéculatives, relance ses premiums.
    - Postflop : Adopte une stratégie "fit-or-fold" (ne joue que s'il touche une paire ou un tirage).
    """
    def __init__(self, name: str = "RecPlayer"):
        super().__init__(name)
        
    def get_preflop_score(self, cards: np.ndarray) -> float:
        """
        Calcule un score de force de la main de départ préflop (style formule de Chen simplifiée).
        Retourne un score entre -2.0 et 20.0.
        """
        r1 = (cards[0] >> 8) & 0xF
        r2 = (cards[1] >> 8) & 0xF
        s1 = (cards[0] >> 12) & 0xF
        s2 = (cards[1] >> 12) & 0xF
        
        high = max(r1, r2)
        low = min(r1, r2)
        
        # Attribution de points selon la valeur de la carte haute
        # 2=1.0, 3=1.5, ..., 10=5.0, J=6.0, Q=7.0, K=8.0, A=10.0
        rank_to_pts = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
        score = rank_to_pts[high]
        
        # Si c'est une paire
        if r1 == r2:
            score = max(5.0, score * 2.0)
            if r1 == 12: # AA
                score = 20.0
        else:
            # Si les cartes sont assorties (Suited)
            if s1 == s2:
                score += 2.0
                
            # Pénalité selon l'écart (Gap) entre les deux cartes
            gap = high - low - 1
            if gap == 0:
                score += 1.0  # Connecteurs (ex: 8-9)
            elif gap == 1:
                score -= 1.0  # 1 gap (ex: 8-10)
            elif gap == 2:
                score -= 2.0  # 2 gaps (ex: 8-J)
            elif gap == 3:
                score -= 4.0  # 3 gaps
            else:
                score -= 5.0  # Plus de 3 gaps
                
        return score

    def has_flush_draw(self, hole_cards: np.ndarray, board: np.ndarray) -> bool:
        """
        Détermine si le joueur possède un tirage couleur (4 cartes de la même couleur).
        """
        suit_counts = np.zeros(16, dtype=np.int32)
        # Compter les couleurs en main
        for c in hole_cards:
            suit_counts[(c >> 12) & 0xF] += 1
        # Compter les couleurs sur le board
        for c in board:
            if c != 0:
                suit_counts[(c >> 12) & 0xF] += 1
                
        return np.any(suit_counts >= 4)

    def select_action(self, game_state: GameState) -> Tuple[int, int]:
        valid_actions = game_state.get_valid_actions()
        if not valid_actions:
            return ACTION_FOLD, 0
            
        p = game_state.players[game_state.current_player_idx]
        to_call = game_state.highest_bet - p.current_bet
        
        # Dictionnaire pour un accès rapide aux actions
        action_map = {act[0]: (act[1], act[2]) for act in valid_actions}
        
        # --- PHASE PRÉFLOP ---
        if game_state.current_round == ROUND_PREFLOP:
            pf_score = self.get_preflop_score(p.cards)
            
            # 1. Mains Premium (AA, KK, QQ, JJ, AKs, etc.)
            if pf_score >= 11.0:
                if ACTION_RAISE in action_map:
                    # Relancer ou sur-relancer à 3x la mise adverse
                    min_raise, max_raise = action_map[ACTION_RAISE]
                    raise_amount = min(max_raise, max(min_raise, to_call * 3))
                    return ACTION_RAISE, raise_amount
                elif ACTION_CALL in action_map:
                    return ACTION_CALL, action_map[ACTION_CALL][0]
                    
            # 2. Mains Fortes (TT, 99, AQo, KQs, etc.)
            elif pf_score >= 8.0:
                # Si la relance demandée est raisonnable (<= 4 BB)
                if to_call <= game_state.bb * 4:
                    if ACTION_RAISE in action_map and to_call == 0:
                        # Ouvrir à 3 BB
                        min_raise, max_raise = action_map[ACTION_RAISE]
                        raise_amount = min(max_raise, max(min_raise, game_state.bb * 3))
                        return ACTION_RAISE, raise_amount
                    elif ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                # Sinon fold
                return ACTION_FOLD, 0
                
            # 3. Mains Spéculatives (Paires faibles, connecteurs assortis comme 78s)
            elif pf_score >= 5.0:
                # On limpe ou on paye seulement si c'est pas trop cher (<= 2 BB)
                if to_call <= game_state.bb * 2:
                    if ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                return ACTION_FOLD, 0
                
            # 4. Poubelle
            else:
                if to_call == 0 and ACTION_CALL in action_map:
                    # Checker si c'est gratuit (ex: en Grosse Blinde)
                    return ACTION_CALL, 0
                return ACTION_FOLD, 0
                
        # --- PHASE POSTFLOP (Flop, Turn, River) ---
        else:
            # Évaluation de notre jeu de 7 cartes
            cards_7 = np.zeros(7, dtype=np.int32)
            cards_7[0:2] = p.cards
            cards_7[2:7] = game_state.board
            
            score = evaluate_7_cards(cards_7)
            category = score >> 20
            
            # Détection de tirage couleur
            has_draw = self.has_flush_draw(p.cards, game_state.board)
            
            # 1. Monstre (Brelan, Quinte, Couleur ou mieux)
            if category >= 4:
                if ACTION_RAISE in action_map:
                    # Relancer à hauteur du pot (pour valoriser)
                    min_raise, max_raise = action_map[ACTION_RAISE]
                    pot_size = game_state.pot + to_call
                    raise_amount = min(max_raise, max(min_raise, pot_size))
                    return ACTION_RAISE, raise_amount
                elif ACTION_CALL in action_map:
                    return ACTION_CALL, action_map[ACTION_CALL][0]
                    
            # 2. Main Faite Correcte (Double Paire ou Top Paire)
            elif category == 3 or (category == 2 and ((score >> 16) & 0xF) >= 9): # Paire de Valet ou mieux
                # Si quelqu'un mise gros (plus de la moitié du pot), on se contente de payer
                if to_call > game_state.pot * 0.6:
                    if ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                    return ACTION_FOLD, 0
                # Sinon, si c'est à nous de miser, on fait une petite mise (1/3 du pot)
                elif ACTION_RAISE in action_map and to_call == 0:
                    min_raise, max_raise = action_map[ACTION_RAISE]
                    bet_amount = min(max_raise, max(min_raise, int(game_state.pot * 0.33)))
                    return ACTION_RAISE, bet_amount
                elif ACTION_CALL in action_map:
                    return ACTION_CALL, action_map[ACTION_CALL][0]
                    
            # 3. Main Faible ou Simple Tirage
            elif (category == 2) or (category == 1 and has_draw):
                # On paye seulement si la mise adverse est petite (<= 33% du pot)
                if to_call <= game_state.pot * 0.33:
                    if ACTION_CALL in action_map:
                        return ACTION_CALL, action_map[ACTION_CALL][0]
                return ACTION_FOLD, 0
                
            # 4. Rien du tout (Hauteur pure sans tirage)
            else:
                if to_call == 0 and ACTION_CALL in action_map:
                    return ACTION_CALL, 0 # Check
                return ACTION_FOLD, 0
                
        return ACTION_FOLD, 0
