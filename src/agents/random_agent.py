import random
from typing import Tuple
from src.agents.base_agent import BaseAgent
from src.engine.game_state import GameState, ACTION_FOLD, ACTION_CALL, ACTION_RAISE

class RandomAgent(BaseAgent):
    """
    Un agent de poker simple qui choisit ses actions de manière uniforme 
    parmi les actions valides autorisées.
    """
    def __init__(self, name: str = "RandomPlayer"):
        super().__init__(name)
        
    def select_action(self, game_state: GameState) -> Tuple[int, int]:
        valid_actions = game_state.get_valid_actions()
        if not valid_actions:
            return ACTION_FOLD, 0
            
        # Choisir une action aléatoire
        action = random.choice(valid_actions)
        action_type, min_amt, max_amt = action
        
        if action_type == ACTION_RAISE:
            # Pour une relance, on tire un montant aléatoire entre min et max.
            # On favorise statistiquement les petites relances (min_amt) pour simuler un jeu plus réaliste
            if random.random() < 0.7:
                amount = min_amt
            else:
                amount = random.randint(min_amt, max_amt)
            return ACTION_RAISE, amount
            
        # Pour Fold ou Call, le montant est fixe
        return action_type, min_amt
