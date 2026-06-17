from src.engine.game_state import GameState
from typing import Tuple

class BaseAgent:
    """
    Interface de base pour tous les agents de poker.
    """
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        
    def select_action(self, game_state: GameState) -> Tuple[int, int]:
        """
        Prend en entrée l'état actuel du jeu et retourne un tuple d'action :
        (action_type, amount_to_add_to_pot)
        
        Note : 'amount_to_add_to_pot' est la quantité de jetons supplémentaire 
        que le joueur doit prélever de son tapis (et non la taille totale de sa mise).
        """
        raise NotImplementedError("Chaque agent doit implémenter la méthode select_action.")
