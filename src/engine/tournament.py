import numpy as np
from typing import List, Dict, Tuple
from src.engine.game_state import GameState, PlayerState
from src.engine.hand_evaluator import evaluate_7_cards

# Niveaux de blindes standards pour nos simulations
# Format : (Small Blind, Big Blind, Ante)
DEFAULT_BLIND_LEVELS = [
    (100, 200, 200),
    (150, 300, 300),
    (200, 400, 400),
    (300, 600, 600),
    (400, 800, 800),
    (500, 1000, 1000),
    (600, 1200, 1200),
    (800, 1600, 1600),
    (1000, 2000, 2000),
    (1500, 3000, 3000),
    (2000, 4000, 4000),
    (3000, 6000, 6000),
    (4000, 8000, 8000),
    (5000, 10000, 10000),
]

class TournamentTable:
    """
    Simule une table unique (Single Table Tournament - STT) de type Sit & Go ou la table finale
    d'un tournoi micro Winamax, avec gestion des blindes progressives et du format KO Bounty.
    """
    def __init__(self, num_players: int = 8, start_chips: int = 20000, 
                 blind_levels: List[Tuple[int, int, int]] = None, 
                 hands_per_level: int = 10, bounty_mode: bool = False,
                 initial_bounty: float = 1.0):
        
        self.num_players = num_players
        self.start_chips = start_chips
        self.blind_levels = blind_levels if blind_levels is not None else DEFAULT_BLIND_LEVELS
        self.hands_per_level = hands_per_level
        self.bounty_mode = bounty_mode
        self.initial_bounty = initial_bounty
        
        # Initialisation de la table de jeu
        self.game = GameState(num_players=num_players, start_chips=start_chips, 
                              sb=self.blind_levels[0][0], bb=self.blind_levels[0][1], 
                              use_bba=True)
                              
        self.current_level_idx = 0
        self.hand_counter = 0
        self.dealer_button = 0
        
        # Suivi des primes KO
        # head_bounties[i] est la prime visible sur la tête du joueur i
        # won_bounties[i] est l'argent réel de primes empoché par le joueur i
        self.head_bounties = np.full(num_players, initial_bounty, dtype=np.float32)
        self.won_bounties = np.zeros(num_players, dtype=np.float32)
        
        # Historique des éliminations : liste de tuples (siège, rang_sortie)
        self.eliminations: List[Tuple[int, int]] = []
        
    def check_eliminations(self) -> List[int]:
        """
        Détecte les joueurs éliminés (tapis = 0) et gère l'attribution des primes KO.
        Retourne la liste des sièges éliminés lors de cette main.
        """
        eliminated_this_hand = []
        
        # Pour le format KO, nous devons identifier qui a éliminé qui.
        # En poker de tournoi, si un joueur est éliminé, le gagnant de ses jetons remporte sa prime.
        # En cas de side pots, c'est le vainqueur du pot contenant la mise finale de l'éliminé qui prend le bounty.
        # Pour faire simple et robuste : nous attribuons le bounty au joueur actif qui a le meilleur jeu
        # lors du showdown, ou à celui qui a provoqué le fold final.
        
        # On commence par trouver qui a des jetons et qui n'en a plus.
        for i, p in enumerate(self.game.players):
            # Si le joueur n'a plus de jetons et n'a pas encore été enregistré comme éliminé
            if p.chips == 0 and i not in [e[0] for e in self.eliminations]:
                eliminated_this_hand.append(i)
                
        if len(eliminated_this_hand) == 0:
            return []
            
        # Trouver les gagnants de la main (ceux qui ont augmenté leur tapis lors de cette main)
        # Note : ceci est une approximation très robuste pour identifier les bénéficiaires des jetons.
        potential_killers = []
        max_gain = 0
        for i, p in enumerate(self.game.players):
            if p.chips > 0 and p.total_contribution > 0:
                # Si le joueur est actif à la fin
                potential_killers.append(i)
                
        # Attribuer les primes KO si mode bounty activé
        if self.bounty_mode and len(potential_killers) > 0:
            # S'il y a des éliminés, on répartit leurs primes aux survivants qui ont gagné des jetons.
            # En toute rigueur, le bounty va à celui qui remporte le pot correspondant.
            # Pour notre moteur, si plusieurs personnes éliminent quelqu'un (ex: split pot), on partage.
            for elim_seat in eliminated_this_hand:
                killer_seat = self.determine_killer(elim_seat, potential_killers)
                if killer_seat != -1:
                    bounty_value = self.head_bounties[elim_seat]
                    
                    # 50% de la prime va directement dans la poche du tueur (won_bounties)
                    self.won_bounties[killer_seat] += bounty_value * 0.5
                    # 50% s'ajoute à la propre tête du tueur
                    self.head_bounties[killer_seat] += bounty_value * 0.5
                    
                    self.head_bounties[elim_seat] = 0.0
                    
        # Enregistrer les éliminations dans l'ordre de sortie
        # Le rang de sortie est déterminé par le nombre de joueurs restants
        remaining_before = self.num_players - len(self.eliminations)
        for elim_seat in eliminated_this_hand:
            self.eliminations.append((elim_seat, remaining_before))
            self.game.players[elim_seat].in_play = False
            
        return eliminated_this_hand
        
    def determine_killer(self, elim_seat: int, potential_killers: List[int]) -> int:
        """
        Détermine quel joueur a éliminé le siège 'elim_seat'.
        Retourne le siège du tueur, ou -1 si non trouvé.
        """
        # Dans un vrai showdown, c'est celui qui a la meilleure main parmi ceux qui ont payé le all-in.
        # Pour simplifier, nous choisissons le joueur parmi les survivants de la main qui a le tapis le plus élevé
        # ou qui a gagné le plus (souvent l'agresseur ou le vainqueur du pot principal).
        if len(potential_killers) == 1:
            return potential_killers[0]
            
        # Par défaut, on attribue au joueur actif ayant la meilleure main au showdown
        # (on recrée l'évaluation rapide)
        best_killer = -1
        best_score = -1
        
        # Récupérer les cartes du board
        board_cards = self.game.board
        
        for killer_seat in potential_killers:
            p = self.game.players[killer_seat]
            cards_7 = np.zeros(7, dtype=np.int32)
            cards_7[0:2] = p.cards
            cards_7[2:7] = board_cards
            
            # Évaluation rapide de la main
            score = evaluate_7_cards(cards_7)
            if score > best_score:
                best_score = score
                best_killer = killer_seat
                
        return best_killer if best_killer != -1 else potential_killers[0]

    def next_hand(self) -> bool:
        """
        Passe à la main suivante. Gère l'augmentation des blindes et vérifie si le tournoi est fini.
        Retourne True si le tournoi continue, False si un vainqueur unique est trouvé.
        """
        # 1. Vérifier et enregistrer les éliminations de la main précédente
        self.check_eliminations()
        
        # 2. Vérifier s'il reste plus d'un joueur avec des jetons
        survivors = [p for p in self.game.players if p.chips > 0]
        if len(survivors) <= 1:
            # Fin du tournoi !
            if len(survivors) == 1:
                winner_seat = survivors[0].seat_id
                # Le vainqueur gagne sa propre prime KO restante en cash
                if self.bounty_mode:
                    self.won_bounties[winner_seat] += self.head_bounties[winner_seat]
            return False
            
        # 3. Incrémenter le compteur de mains
        self.hand_counter += 1
        
        # 4. Gérer le changement de niveau de blindes
        if self.hand_counter % self.hands_per_level == 0:
            if self.current_level_idx < len(self.blind_levels) - 1:
                self.current_level_idx += 1
                sb, bb, ante = self.blind_levels[self.current_level_idx]
                self.game.sb = sb
                self.game.bb = bb
                
        # 5. Déplacer le bouton de dealer au prochain joueur actif
        self.dealer_button = (self.dealer_button + 1) % self.num_players
        while not self.game.players[self.dealer_button].chips > 0:
            self.dealer_button = (self.dealer_button + 1) % self.num_players
            
        # 6. Lancer la nouvelle donne
        self.game.reset_hand(self.dealer_button)
        return True

    def get_rankings(self) -> List[Dict]:
        """
        Retourne le classement final du tournoi sous forme de liste de dictionnaires.
        """
        rankings = []
        
        # Ajouter le vainqueur (qui n'est pas dans la liste des éliminations)
        survivors = [p for p in self.game.players if p.chips > 0]
        if len(survivors) == 1:
            winner = survivors[0]
            rankings.append({
                "rank": 1,
                "seat": winner.seat_id,
                "chips": winner.chips,
                "bounties_won": self.won_bounties[winner.seat_id]
            })
            
        # Ajouter les éliminés dans l'ordre inverse de leur sortie
        for seat, remaining in reversed(self.eliminations):
            rankings.append({
                "rank": remaining,
                "seat": seat,
                "chips": 0,
                "bounties_won": self.won_bounties[seat]
            })
            
        return rankings
