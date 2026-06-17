import numpy as np
from typing import List, Tuple
from src.engine.card import get_deck, card_to_str
from src.engine.hand_evaluator import evaluate_7_cards, score_to_str

# Constants for Actions
ACTION_FOLD = 0
ACTION_CALL = 1  # includes Check
ACTION_RAISE = 2

# Constants for Rounds
ROUND_PREFLOP = 0
ROUND_FLOP = 1
ROUND_TURN = 2
ROUND_RIVER = 3
ROUND_SHOWDOWN = 4

class PlayerState:
    def __init__(self, seat_id: int, initial_chips: int):
        self.seat_id = seat_id
        self.chips = initial_chips
        self.current_bet = 0      # Bet in the current round
        self.total_contribution = 0 # Total chips put in the pot during the whole hand
        self.cards = np.zeros(2, dtype=np.int32)
        self.in_play = True       # False if folded
        self.is_all_in = False
        self.has_acted_this_round = False
        
    def clone(self):
        c = PlayerState(self.seat_id, self.chips)
        c.current_bet = self.current_bet
        c.total_contribution = self.total_contribution
        c.cards = self.cards.copy()
        c.in_play = self.in_play
        c.is_all_in = self.is_all_in
        c.has_acted_this_round = self.has_acted_this_round
        return c
        
    def __repr__(self):
        card_strs = [card_to_str(c) for c in self.cards] if self.in_play else ["Folded"]
        return f"Seat {self.seat_id} ({card_strs[0]},{card_strs[1]}) - Stack: {self.chips} - Bet: {self.current_bet}"

class GameState:
    def __init__(self, num_players: int = 6, start_chips: int = 20000, sb: int = 800, bb: int = 1600, use_bba: bool = False, ante: int = 200):
        self.num_players = num_players
        self.start_chips = start_chips
        self.sb = sb
        self.bb = bb
        self.use_bba = use_bba
        self.ante = ante
        
        self.players = [PlayerState(i, start_chips) for i in range(num_players)]
        self.deck = get_deck()
        self.deck_idx = 0
        self.board = np.zeros(5, dtype=np.int32)
        self.pot = 0
        self.current_round = ROUND_PREFLOP
        self.dealer_idx = 0
        self.current_player_idx = 0
        self.last_aggressor_idx = -1
        self.min_raise = bb
        self.highest_bet = 0  # Highest bet in current round
        
    def clone(self):
        c = GameState(self.num_players, self.start_chips, self.sb, self.bb, self.use_bba, self.ante)
        c.players = [p.clone() for p in self.players]
        c.deck = self.deck.copy()
        c.deck_idx = self.deck_idx
        c.board = self.board.copy()
        c.pot = self.pot
        c.current_round = self.current_round
        c.dealer_idx = self.dealer_idx
        c.current_player_idx = self.current_player_idx
        c.last_aggressor_idx = self.last_aggressor_idx
        c.min_raise = self.min_raise
        c.highest_bet = self.highest_bet
        return c

    def reset_hand(self, dealer_idx: int, randomize_stacks: bool = False):
        """
        Prépare une nouvelle donne : mélange le deck, distribue les cartes, et pose les blindes.
        """
        self.dealer_idx = dealer_idx
        # Mélange du deck
        np.random.shuffle(self.deck)
        self.deck_idx = 0
        self.pot = 0
        self.last_winners = []
        self.current_round = ROUND_PREFLOP
        self.highest_bet = 0
        self.min_raise = self.bb
        self.last_aggressor_idx = -1
        self.board.fill(0)
        
        for p in self.players:
            if randomize_stacks:
                # PKO Space KO : Profondeur entre 15 et 25 BB
                p.chips = np.random.randint(15 * self.bb, 25 * self.bb + 1)
                p.in_play = True
            else:
                # Comportement normal pour les tests et tournois : on préserve les jetons
                p.in_play = (p.chips > 0)
            p.is_all_in = False
            p.current_bet = 0
            p.total_contribution = 0
            p.has_acted_this_round = False
            p.cards.fill(0)
            
        # Distribuer les cartes
        for p in self.players:
            if p.in_play:
                p.cards[0] = self.deck[self.deck_idx]
                p.cards[1] = self.deck[self.deck_idx + 1]
                self.deck_idx += 2
                
        # Poster les blindes et l'ante (Big Blind Ante si activé)
        self.post_blinds_and_antes()
        self.highest_bet = self.bb
        self.min_raise = self.bb
        
        # Le premier à agir préflop est le joueur après la BB (UTG)
        # En 8-max : Dealer + 3 (Dealer=0, SB=1, BB=2, UTG=3)
        self.current_player_idx = (self.dealer_idx + 3) % self.num_players
        while not self.players[self.current_player_idx].in_play:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            
        self.last_aggressor_idx = (self.dealer_idx + 2) % self.num_players # BB est considéré comme la dernière mise
        
    def teleport_to_round(self, target_round: int):
        """
        Téléporte la partie directement à un tour donné (Flop, Turn, ou River).
        Simule l'action préflop pour forcer l'entraînement post-flop.
        """
        if target_round <= ROUND_PREFLOP or target_round > ROUND_RIVER:
            return
            
        active_count = np.random.randint(2, min(5, self.num_players + 1))
        
        first_active = np.random.randint(0, self.num_players)
        active_indices = []
        for i in range(self.num_players):
            idx = (first_active + i) % self.num_players
            if self.players[idx].chips > 0 and len(active_indices) < active_count:
                active_indices.append(idx)
                
        for i in range(self.num_players):
            if i not in active_indices:
                self.players[i].in_play = False
                self.players[i].cards.fill(0)
                
        pot_bbs = np.random.uniform(4.0, 25.0)
        total_simulated_pot = int(pot_bbs * self.bb)
        contribution_per_player = total_simulated_pot // active_count
        
        self.pot = 0
        for idx in active_indices:
            p = self.players[idx]
            # On retire l'argent des blindes déjà prélevées pour ne pas les compter en double
            remaining_to_pay = max(0, contribution_per_player - p.total_contribution)
            actual_contribution = min(remaining_to_pay, p.chips)
            p.chips -= actual_contribution
            self.pot += actual_contribution + p.total_contribution
            p.total_contribution += actual_contribution
            p.current_bet = 0
            if p.chips == 0:
                p.is_all_in = True
                
        # Les autres joueurs voient leur mise récupérée par le pot (déjà géré si on veut, mais ici on simplifie le pot total)
        # Mais attention aux blindes des joueurs qui se couchent ! 
        # Pour faire simple, self.pot reflète la réalité mathématique des joueurs actifs.
                
        self.highest_bet = 0
        self.min_raise = self.bb
        self.last_aggressor_idx = -1
        
        for p in self.players:
            p.has_acted_this_round = False
            
        if target_round >= ROUND_FLOP:
            self.board[0] = self.deck[self.deck_idx]
            self.board[1] = self.deck[self.deck_idx + 1]
            self.board[2] = self.deck[self.deck_idx + 2]
            self.deck_idx += 3
            self.current_round = ROUND_FLOP
            
        if target_round >= ROUND_TURN:
            self.board[3] = self.deck[self.deck_idx]
            self.deck_idx += 1
            self.current_round = ROUND_TURN
            
        if target_round >= ROUND_RIVER:
            self.board[4] = self.deck[self.deck_idx]
            self.deck_idx += 1
            self.current_round = ROUND_RIVER
            
        self.current_player_idx = (self.dealer_idx + 1) % self.num_players
        while not self.players[self.current_player_idx].in_play or self.players[self.current_player_idx].is_all_in:
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players
            
        # Vérif : si plus d'un joueur actif n'est pas all-in
        active_not_all_in = sum(1 for p in self.players if p.in_play and not p.is_all_in)
        if active_not_all_in <= 1:
            self.transition_to_next_round()

    def post_blinds_and_antes(self):
        """
        Gère le prélèvement des antes (Big Blind Ante) et des blindes.
        """
        active_players = sum(1 for p in self.players if p.in_play)
        if active_players < 2:
            return
            
        # 1. Identification des sièges SB et BB
        if active_players == 2:
            # En heads-up, le dealer est SB, l'autre joueur actif est BB
            sb_seat = self.dealer_idx
            bb_seat = (self.dealer_idx + 1) % self.num_players
            while not self.players[bb_seat].in_play:
                bb_seat = (bb_seat + 1) % self.num_players
        else:
            # À 3 joueurs ou plus, SB est après le dealer, BB est après la SB
            sb_seat = (self.dealer_idx + 1) % self.num_players
            while not self.players[sb_seat].in_play:
                sb_seat = (sb_seat + 1) % self.num_players
                
            bb_seat = (sb_seat + 1) % self.num_players
            while not self.players[bb_seat].in_play:
                bb_seat = (bb_seat + 1) % self.num_players
                
        p_sb = self.players[sb_seat]
        p_bb = self.players[bb_seat]
        
        # 2. Prélèvement de l'Ante (par joueur)
        if self.ante > 0:
            for p in self.players:
                if p.in_play:
                    actual_ante = min(p.chips, self.ante)
                    p.chips -= actual_ante
                    self.pot += actual_ante
                    if p.chips == 0:
                        p.is_all_in = True
                
        # 3. Prélèvement Small Blind
        actual_sb = min(p_sb.chips, self.sb)
        p_sb.chips -= actual_sb
        p_sb.current_bet = actual_sb
        p_sb.total_contribution = actual_sb
        if p_sb.chips == 0:
            p_sb.is_all_in = True
            
        # 4. Prélèvement Big Blind
        actual_bb = min(p_bb.chips, self.bb)
        p_bb.chips -= actual_bb
        p_bb.current_bet = actual_bb
        p_bb.total_contribution = actual_bb
        if p_bb.chips == 0:
            p_bb.is_all_in = True

    def get_valid_actions(self) -> List[Tuple[int, int, int]]:
        """
        Retourne la liste des actions valides pour le joueur actif.
        Chaque action est un tuple (action_type, min_amount, max_amount).
        """
        p = self.players[self.current_player_idx]
        
        if not p.in_play or p.is_all_in:
            return []
            
        actions = []
        
        # 1. Fold est toujours possible si le joueur fait face à un bet supérieur au sien
        if p.current_bet < self.highest_bet:
            actions.append((ACTION_FOLD, 0, 0))
        else:
            # Si aucune mise n'a été faite, le joueur peut Check (qui correspond à un Call à 0 jetons de plus)
            actions.append((ACTION_CALL, 0, 0))
            
        # 2. Call / Check
        to_call = self.highest_bet - p.current_bet
        if to_call > 0:
            call_amount = min(p.chips, to_call)
            actions.append((ACTION_CALL, call_amount, call_amount))
            
        # 3. Raise / Bet
        if p.chips > to_call:
            min_bet_total = self.highest_bet + self.min_raise
            # Si la relance minimale dépasse le tapis du joueur, il peut seulement faire tapis (All-in)
            actual_min_raise = min(p.chips, min_bet_total - p.current_bet)
            actual_max_raise = p.chips  # All-in
            
            if actual_min_raise > to_call:
                actions.append((ACTION_RAISE, actual_min_raise, actual_max_raise))
                
        return actions

    def execute_action(self, action_type: int, amount: int = 0):
        """
        Applique l'action d'un joueur et met à jour le game state.
        """
        p = self.players[self.current_player_idx]
        p.has_acted_this_round = True
        
        if action_type == ACTION_FOLD:
            p.in_play = False
            
        elif action_type == ACTION_CALL:
            # Le montant 'amount' est la somme ajoutée au pot
            p.chips -= amount
            p.current_bet += amount
            p.total_contribution += amount
            if p.chips == 0:
                p.is_all_in = True
                
        elif action_type == ACTION_RAISE:
            # Le montant 'amount' est la somme ajoutée au pot
            added_amount = amount
            p.chips -= added_amount
            p.current_bet += added_amount
            p.total_contribution += added_amount
            
            # Mise à jour du pas de relance minimum
            new_total_bet = p.current_bet
            raise_size = new_total_bet - self.highest_bet
            if raise_size > self.min_raise:
                self.min_raise = raise_size
                
            self.highest_bet = new_total_bet
            self.last_aggressor_idx = self.current_player_idx
            
            if p.chips == 0:
                p.is_all_in = True
                
        # Passer au joueur suivant
        self.move_to_next_player()

    def move_to_next_player(self):
        """
        Sélectionne le prochain joueur actif et gère la transition entre les rounds de mise.
        """
        # Vérifier s'il ne reste qu'un seul joueur actif
        active_players = [p for p in self.players if p.in_play]
        if len(active_players) <= 1:
            self.end_hand()
            return
            
        # Trouver le prochain joueur
        next_idx = (self.current_player_idx + 1) % self.num_players
        loops = 0
        while loops < self.num_players:
            p = self.players[next_idx]
            # Un joueur doit agir s'il est en jeu, pas all-in, et s'il n'a pas encore complété sa mise
            if p.in_play and not p.is_all_in:
                # Calculer le siège de la Grosse Blinde
                bb_seat = (self.dealer_idx + 2) % self.num_players
                while not self.players[bb_seat].in_play:
                    bb_seat = (bb_seat + 1) % self.num_players
                
                # Cas préflop particulier si tout le monde a callé la BB sans relancer
                if (self.current_round == ROUND_PREFLOP and 
                    self.highest_bet == self.bb and 
                    next_idx == bb_seat and 
                    p.current_bet == self.bb and
                    self.last_aggressor_idx == bb_seat):
                    # La BB a l'option de Check ou Relancer si elle n'a pas encore agi
                    self.current_player_idx = next_idx
                    # On change le last aggressor pour que le tour se termine si la BB check
                    self.last_aggressor_idx = -1
                    return
                
                # Si on est revenu à l'agresseur initial et que tout le monde a checké/callé, le round est fini
                if next_idx == self.last_aggressor_idx and p.current_bet == self.highest_bet:
                    break
                # Si le round de mise est complété
                if p.current_bet == self.highest_bet and p.has_acted_this_round:
                    break
                
                self.current_player_idx = next_idx
                return
                
            next_idx = (next_idx + 1) % self.num_players
            loops += 1
            
        # Si on arrive ici, le round de mise est terminé
        self.transition_to_next_round()

    def transition_to_next_round(self):
        """
        Passe au tour suivant (Flop, Turn, River) et distribue les cartes communes.
        """
        # Collecter les mises courantes de ce round dans le pot général
        for p in self.players:
            self.pot += p.current_bet
            p.current_bet = 0
            p.has_acted_this_round = False
            
        self.highest_bet = 0
        self.min_raise = self.bb
        self.last_aggressor_idx = -1
        
        # Vérifier si tous les joueurs restants (sauf un au maximum) sont All-in
        active_not_all_in = sum(1 for p in self.players if p.in_play and not p.is_all_in)
        
        if self.current_round == ROUND_PREFLOP:
            # Distribution du flop
            self.board[0] = self.deck[self.deck_idx]
            self.board[1] = self.deck[self.deck_idx + 1]
            self.board[2] = self.deck[self.deck_idx + 2]
            self.deck_idx += 3
            self.current_round = ROUND_FLOP
            
        elif self.current_round == ROUND_FLOP:
            # Distribution de la turn
            self.board[3] = self.deck[self.deck_idx]
            self.deck_idx += 1
            self.current_round = ROUND_TURN
            
        elif self.current_round == ROUND_TURN:
            # Distribution de la river
            self.board[4] = self.deck[self.deck_idx]
            self.deck_idx += 1
            self.current_round = ROUND_RIVER
            
        elif self.current_round == ROUND_RIVER:
            self.current_round = ROUND_SHOWDOWN
            self.end_hand()
            return
            
        # Si tout le monde est all-in sauf 1 joueur (ou tout le monde est all-in), on distribue le reste des cartes jusqu'au showdown
        if active_not_all_in <= 1:
            while self.current_round != ROUND_SHOWDOWN:
                if self.current_round == ROUND_FLOP:
                    self.board[3] = self.deck[self.deck_idx]
                    self.deck_idx += 1
                    self.current_round = ROUND_TURN
                elif self.current_round == ROUND_TURN:
                    self.board[4] = self.deck[self.deck_idx]
                    self.deck_idx += 1
                    self.current_round = ROUND_RIVER
                elif self.current_round == ROUND_RIVER:
                    self.current_round = ROUND_SHOWDOWN
            self.end_hand()
            return
            
        # Le premier à agir postflop est le premier joueur actif après le Dealer (SB)
        self.current_player_idx = (self.dealer_idx + 1) % self.num_players
        while not (self.players[self.current_player_idx].in_play and not self.players[self.current_player_idx].is_all_in):
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def end_hand(self):
        """
        Gère le showdown (ou la victoire par abandon) et distribue les jetons du pot.
        """
        self.last_winners = []
        
        # Collecter les dernières mises dans le pot
        for p in self.players:
            self.pot += p.current_bet
            p.current_bet = 0
            
        active_players = [p for p in self.players if p.in_play]
        
        # Cas 1 : Tout le monde a foldé sauf un joueur
        if len(active_players) == 1:
            winner = active_players[0]
            amount = self.pot
            winner.chips += self.pot
            self.last_winners.append({"id": winner.seat_id, "amount": amount, "reason": "Tous les autres ont abandonné"})
            self.pot = 0
            self.current_round = ROUND_SHOWDOWN
            return
            
        # Cas 2 : Showdown classique avec calcul des side pots
        # Nous utilisons notre algorithme d'attribution équitable basé sur les contributions réelles des joueurs.
        
        # Récupérer la force de main de chaque joueur actif
        player_scores = {}
        for p in active_players:
            cards_7 = np.zeros(7, dtype=np.int32)
            cards_7[0:2] = p.cards
            cards_7[2:7] = self.board
            player_scores[p.seat_id] = evaluate_7_cards(cards_7)
            
        # Dictionnaire des contributions (pour préserver l'état de total_contribution)
        contributions = {p.seat_id: p.total_contribution for p in self.players if p.total_contribution > 0}
        
        # Le surplus (les antes) est de l'argent mort qui va dans le pot principal (le premier pot partiel)
        dead_money = self.pot - sum(contributions.values())
        first_level = True
        
        # Algorithme d'attribution incrémental
        while len(contributions) > 0:
            # Trouver la contribution minimale en cours
            min_contrib = min(contributions.values())
            
            # Les joueurs éligibles pour ce pot partiel sont les joueurs encore actifs
            # qui ont participé à ce niveau de pot
            eligible_winners = [p for p in active_players if p.seat_id in contributions]
            
            # Construire le pot partiel
            partial_pot = 0
            for seat_id in list(contributions.keys()):
                partial_pot += min_contrib
                contributions[seat_id] -= min_contrib
                
            # Ajouter l'argent mort (ante) au pot principal (premier niveau)
            if first_level:
                partial_pot += dead_money
                first_level = False
                
            if len(eligible_winners) > 0:
                # Trouver la meilleure main parmi les gagnants éligibles
                best_score = max(player_scores[p.seat_id] for p in eligible_winners)
                winners = [p for p in eligible_winners if player_scores[p.seat_id] == best_score]
                
                # Partager le pot partiel équitablement (avec troncature des jetons)
                split_amount = partial_pot // len(winners)
                remainder = partial_pot % len(winners)
                
                for w in winners:
                    amt = split_amount + (remainder if w == winners[0] else 0)
                    w.chips += amt
                    self.last_winners.append({"id": w.seat_id, "amount": amt, "reason": score_to_str(best_score)})
            
            # Retirer de la liste les joueurs dont la contribution restante est de 0
            contributions = {seat_id: contrib for seat_id, contrib in contributions.items() if contrib > 0}
            
        self.pot = 0
        self.current_round = ROUND_SHOWDOWN
