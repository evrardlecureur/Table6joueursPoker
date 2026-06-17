import numpy as np

# Gestion du décorateur JIT de Numba pour compilation à la volée.
# Si Numba n'est pas installé, on utilise un décorateur pass-through.
try:
    from numba import jit
except ImportError:
    def jit(nopython=True, cache=True):
        return lambda f: f

# Noms des catégories de mains pour affichage
HAND_CATEGORIES = {
    9: "Straight Flush",
    8: "Four of a Kind",
    7: "Full House",
    6: "Flush",
    5: "Straight",
    4: "Three of a Kind",
    3: "Two Pair",
    2: "One Pair",
    1: "High Card"
}

@jit(nopython=True, cache=True)
def evaluate_draws(cards: np.ndarray) -> np.ndarray:
    """
    Détecte les tirages (flush draw, oesd, gutshot) et retourne un vecteur [fd, oesd, gutshot, normalized_outs].
    cards: array de 5 à 7 entiers représentant les cartes.
    """
    rank_counts = np.zeros(14, dtype=np.int32) # 0-13, 13 = As (high), on copiera rank 0 (2) à 12 (A) => non, ranks de 0(2) à 12(A). On ajoute un pseudo-rank -1 pour l'As low.
    # Pour numba, on fait un tableau de 14. 0=As low, 1=2, 2=3 ... 13=As high.
    suit_counts = np.zeros(16, dtype=np.int32)
    
    for i in range(len(cards)):
        if cards[i] == 0: continue
        r = (cards[i] >> 8) & 0xF
        s = (cards[i] >> 12) & 0xF
        suit_counts[s] += 1
        rank_counts[r + 1] += 1 # r de 0 à 12 -> 1 à 13
        if r == 12:
            rank_counts[0] += 1 # As low
            
    # Flush draw
    fd = 0.0
    for s in range(16):
        if suit_counts[s] == 4:
            fd = 1.0
            break
            
    # Straight draws
    oesd = 0.0
    gutshot = 0.0
    
    # On cherche les fenêtres de 5 cartes
    # OESD: 4 cartes consécutives (sans trou), ex: 4-5-6-7. Sauf si c'est A-2-3-4 ou J-Q-K-A (les extrémités ne sont pas ouvertes des deux côtés).
    # Gutshot: 4 cartes dans une fenêtre de 5 avec 1 trou.
    for i in range(10): # i de 0 à 9
        window = rank_counts[i:i+5]
        # On s'assure que chaque carte est unique dans la fenêtre pour compter les rangs uniques
        unique_ranks = 0
        for j in range(5):
            if window[j] > 0:
                unique_ranks += 1
                
        if unique_ranks == 4:
            # Est-ce un OESD ou Gutshot ?
            # Si le trou est aux extrémités (i.e., la carte manquante est l'index 0 ou 4) -> 4 consécutives
            if window[0] == 0 or window[4] == 0:
                # 4 consécutives. Si c'est A-2-3-4 (i=0) ou J-Q-K-A (i=9), ce n'est pas "open-ended" (seulement 4 outs, donc gutshot)
                if i == 0 or i == 9:
                    gutshot = 1.0
                else:
                    oesd = 1.0
            else:
                # Le trou est à l'intérieur -> Gutshot
                gutshot = 1.0
                
    outs = 0.0
    if fd == 1.0: outs += 9.0
    if oesd == 1.0: outs += 8.0
    elif gutshot == 1.0: outs += 4.0
    
    if fd == 1.0 and oesd == 1.0: outs = 15.0 # Flush + OESD = 15 outs (2 communs)
    elif fd == 1.0 and gutshot == 1.0: outs = 12.0
    
    norm_outs = min(15.0, outs) / 15.0
    
    return np.array([fd, oesd, gutshot, norm_outs], dtype=np.float32)

@jit(nopython=True, cache=True)
def evaluate_5_cards(cards: np.ndarray) -> int:
    """
    Évalue une main de 5 cartes et retourne un score entier de 32 bits.
    Plus le score est élevé, meilleure est la main.
    
    Structure du score (24 bits utiles) :
    [Catégorie (4 bits)] [Tie-breaker 1 (4 bits)] [Tie-breaker 2 (4 bits)] [Tie-breaker 3 (4 bits)] [Tie-breaker 4 (4 bits)] [Tie-breaker 5 (4 bits)]
    """
    # 1. Comptage des occurrences de valeurs (ranks) et couleurs (suits)
    rank_counts = np.zeros(13, dtype=np.int32)
    suit_counts = np.zeros(16, dtype=np.int32) # Index de 1 à 8 (bitmask de couleur)
    
    for i in range(5):
        c = cards[i]
        r = (c >> 8) & 0xF
        s = (c >> 12) & 0xF
        rank_counts[r] += 1
        suit_counts[s] += 1
        
    # 2. Vérification de la couleur (flush)
    is_flush = False
    for s in range(16):
        if suit_counts[s] == 5:
            is_flush = True
            break
            
    # 3. Vérification de la quinte (straight)
    unique_ranks_mask = 0
    min_r = 13
    max_r = -1
    for i in range(5):
        r = (cards[i] >> 8) & 0xF
        unique_ranks_mask |= (1 << r)
        if r < min_r: min_r = r
        if r > max_r: max_r = r
        
    num_unique = 0
    for r in range(13):
        if (unique_ranks_mask & (1 << r)) != 0:
            num_unique += 1
            
    is_straight = False
    straight_high = -1
    if num_unique == 5:
        if (max_r - min_r) == 4:
            is_straight = True
            straight_high = max_r
        elif unique_ranks_mask == 0x100F:  # A, 2, 3, 4, 5 (Ranks: 12, 0, 1, 2, 3)
            is_straight = True
            straight_high = 3  # La quinte se termine au 5 (Rank 3)
            
    # 4. Identification du Straight Flush
    if is_flush and is_straight:
        return (9 << 20) | (straight_high << 16)
        
    # 5. Extraction des paires, brelans et carrés
    quad_rank = -1
    triple_rank = -1
    pair_ranks = np.zeros(2, dtype=np.int32)
    num_pairs = 0
    
    # Parcourir les valeurs du As (12) au 2 (0) pour trier automatiquement par force
    for r in range(12, -1, -1):
        cnt = rank_counts[r]
        if cnt == 4:
            quad_rank = r
        elif cnt == 3:
            triple_rank = r
        elif cnt == 2:
            pair_ranks[num_pairs] = r
            num_pairs += 1

    # 6. Classification et calcul du score
    # Four of a Kind (Carré)
    if quad_rank != -1:
        # Trouver le kicker
        kicker = -1
        for r in range(12, -1, -1):
            if rank_counts[r] == 1:
                kicker = r
                break
        return (8 << 20) | (quad_rank << 16) | (kicker << 12)
        
    # Full House
    if triple_rank != -1 and num_pairs > 0:
        return (7 << 20) | (triple_rank << 16) | (pair_ranks[0] << 12)
        
    # Flush (Couleur)
    if is_flush:
        # Extraire les 5 cartes triées par valeur
        sorted_ranks = np.zeros(5, dtype=np.int32)
        idx = 0
        for r in range(12, -1, -1):
            if rank_counts[r] > 0:
                sorted_ranks[idx] = r
                idx += 1
        return (6 << 20) | (sorted_ranks[0] << 16) | (sorted_ranks[1] << 12) | (sorted_ranks[2] << 8) | (sorted_ranks[3] << 4) | sorted_ranks[4]
        
    # Straight (Quinte)
    if is_straight:
        return (5 << 20) | (straight_high << 16)
        
    # Three of a Kind (Brelan)
    if triple_rank != -1:
        # Extraire les deux kickers
        kickers = np.zeros(2, dtype=np.int32)
        idx = 0
        for r in range(12, -1, -1):
            if rank_counts[r] == 1:
                kickers[idx] = r
                idx += 1
        return (4 << 20) | (triple_rank << 16) | (kickers[0] << 12) | (kickers[1] << 8)
        
    # Two Pair (Double Paire)
    if num_pairs >= 2:
        # Trouver le kicker restant
        kicker = -1
        for r in range(12, -1, -1):
            if rank_counts[r] == 1:
                kicker = r
                break
        return (3 << 20) | (pair_ranks[0] << 16) | (pair_ranks[1] << 12) | (kicker << 8)
        
    # One Pair (Paire)
    if num_pairs == 1:
        # Extraire les 3 kickers restant
        kickers = np.zeros(3, dtype=np.int32)
        idx = 0
        for r in range(12, -1, -1):
            if rank_counts[r] == 1:
                kickers[idx] = r
                idx += 1
        return (2 << 20) | (pair_ranks[0] << 16) | (kickers[0] << 12) | (kickers[1] << 8) | (kickers[2] << 4)
        
    # High Card (Hauteur)
    sorted_ranks = np.zeros(5, dtype=np.int32)
    idx = 0
    for r in range(12, -1, -1):
        if rank_counts[r] == 1:
            sorted_ranks[idx] = r
            idx += 1
    return (1 << 20) | (sorted_ranks[0] << 16) | (sorted_ranks[1] << 12) | (sorted_ranks[2] << 8) | (sorted_ranks[3] << 4) | sorted_ranks[4]

@jit(nopython=True, cache=True)
def evaluate_7_cards(cards: np.ndarray) -> int:
    """
    Évalue une main de 7 cartes en testant les 21 combinaisons de 5 cartes.
    Retourne le score maximal trouvé.
    """
    max_score = -1
    temp_cards = np.zeros(5, dtype=np.int32)
    
    # Liste codée en dur des indices des combinaisons de 5 parmi 7
    # Évite de faire des allocations dynamiques pendant le jeu (très rapide)
    combos = (
        (0, 1, 2, 3, 4), (0, 1, 2, 3, 5), (0, 1, 2, 3, 6),
        (0, 1, 2, 4, 5), (0, 1, 2, 4, 6), (0, 1, 2, 5, 6),
        (0, 1, 3, 4, 5), (0, 1, 3, 4, 6), (0, 1, 3, 5, 6),
        (0, 1, 4, 5, 6), (0, 2, 3, 4, 5), (0, 2, 3, 4, 6),
        (0, 2, 3, 5, 6), (0, 2, 4, 5, 6), (0, 3, 4, 5, 6),
        (1, 2, 3, 4, 5), (1, 2, 3, 4, 6), (1, 2, 3, 5, 6),
        (1, 2, 4, 5, 6), (1, 3, 4, 5, 6), (2, 3, 4, 5, 6)
    )
    
    for i in range(21):
        combo = combos[i]
        temp_cards[0] = cards[combo[0]]
        temp_cards[1] = cards[combo[1]]
        temp_cards[2] = cards[combo[2]]
        temp_cards[3] = cards[combo[3]]
        temp_cards[4] = cards[combo[4]]
        
        score = evaluate_5_cards(temp_cards)
        if score > max_score:
            max_score = score
            
    return max_score

def score_to_str(score: int) -> str:
    """
    Affiche un score d'évaluation de manière compréhensible.
    """
    category = score >> 20
    cat_name = HAND_CATEGORIES.get(category, "Inconnue")
    
    # Lecture des tie-breakers (valeurs de cartes 0-12)
    tb = []
    for i in range(5):
        shift = 16 - (i * 4)
        val = (score >> shift) & 0xF
        if val != 0 or i == 0:  # Toujours garder au moins le premier tie-breaker
            tb.append("23456789TJQKA"[val])
            
    return f"{cat_name} (Force: {', '.join(tb)})"
import numpy as np
from src.engine.hand_evaluator import evaluate_7_cards

# Si Numba n'est pas installé, on utilise un décorateur pass-through.
try:
    from numba import jit
except ImportError:
    def jit(nopython=True, cache=True):
        return lambda f: f

@jit(nopython=True, cache=True)
def jit_estimate_equity(hero_cards: np.ndarray, board_cards: np.ndarray, num_board: int, deck: np.ndarray, trials: int = 10) -> float:
    # 1. Identifier les cartes disponibles
    available = np.zeros(52, dtype=np.int32)
    num_avail = 0
    
    for i in range(52):
        c = deck[i]
        # Check if c is in hero_cards
        is_hero = False
        if c == hero_cards[0] or c == hero_cards[1]:
            is_hero = True
        
        # Check if c is in board
        is_board = False
        for j in range(num_board):
            if c == board_cards[j]:
                is_board = True
                break
                
        if not is_hero and not is_board:
            available[num_avail] = c
            num_avail += 1
            
    wins = 0.0
    hero_7 = np.zeros(7, dtype=np.int32)
    opp_7 = np.zeros(7, dtype=np.int32)
    
    hero_7[0] = hero_cards[0]
    hero_7[1] = hero_cards[1]
    
    for j in range(num_board):
        hero_7[2+j] = board_cards[j]
        opp_7[2+j] = board_cards[j]
        
    cards_needed = 5 - num_board
    
    for _ in range(trials):
        # Tirer 2 cartes pour l'adversaire + cards_needed pour le board
        # Utilisation de np.random.choice sans remplacement (manuel pour Numba)
        # on mélange juste les (2 + cards_needed) premiers éléments
        for i in range(2 + cards_needed):
            idx = np.random.randint(i, num_avail)
            # swap
            tmp = available[i]
            available[i] = available[idx]
            available[idx] = tmp
            
        opp_c1 = available[0]
        opp_c2 = available[1]
        
        opp_7[0] = opp_c1
        opp_7[1] = opp_c2
        
        for j in range(cards_needed):
            b_c = available[2+j]
            hero_7[2+num_board+j] = b_c
            opp_7[2+num_board+j] = b_c
            
        h_score = evaluate_7_cards(hero_7)
        o_score = evaluate_7_cards(opp_7)
        
        if h_score > o_score:
            wins += 1.0
        elif h_score == o_score:
            wins += 0.5
            
    return wins / trials
