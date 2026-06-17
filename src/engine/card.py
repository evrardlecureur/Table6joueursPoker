# Représentation binaire d'une carte (Encodage Cactus Kev / Treys)
# Chaque carte est représentée par un entier sur 32 bits :
#
#  xxxbbbbb bbbbbbbb ccccvvvv vvvvvvvv
#  xxx : non utilisé
#  b : représentation binaire de la valeur (bitmask)
#  c : couleur de la carte (bitmask : 1=Trèfle, 2=Carreau, 4=Cœur, 8=Pique)
#  v : valeur de la carte (2=0, 3=1, ..., A=12)
#
# Cet encodage permet d'évaluer les quintes et les couleurs par de simples opérations binaires (AND, OR, SHIFT).

import numpy as np

# Valeurs de cartes (indexées de 0 à 12)
STR_VALS = "23456789TJQKA"
VAL_MAP = {char: i for i, char in enumerate(STR_VALS)}

# Couleurs (Clubs/Trèfle=1, Diamonds/Carreau=2, Hearts/Cœur=4, Spades/Pique=8)
STR_SUITS = "cdhs"  # c = clubs (♣), d = diamonds (♦), h = hearts (♥), s = spades (♠)
SUIT_MAP = {
    'c': 1,  # 0001
    'd': 2,  # 0010
    'h': 4,  # 0100
    's': 8   # 1000
}

# Représentation visuelle des couleurs
SUIT_SYMBOLS = {
    'c': '♣',
    'd': '♦',
    'h': '♥',
    's': '♠'
}

# Liste des nombres premiers pour le calcul du produit unique (Cactus Kev)
# Associer un nombre premier unique à chaque valeur permet d'identifier
# une combinaison de 5 cartes par multiplication, indépendamment de l'ordre.
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

def make_card(card_str: str) -> int:
    """
    Crée une carte (entier 32-bit) à partir de sa chaîne de caractères (ex: 'Ah', 'Td').
    """
    if len(card_str) != 2:
        raise ValueError("La chaîne doit faire exactement 2 caractères (ex: 'Ah')")
    
    val_char = card_str[0].upper()
    suit_char = card_str[1].lower()
    
    if val_char not in VAL_MAP or suit_char not in SUIT_MAP:
        raise ValueError(f"Carte invalide: {card_str}")
        
    val = VAL_MAP[val_char]
    suit = SUIT_MAP[suit_char]
    prime = PRIMES[val]
    
    # Construction de l'entier 32 bits
    # Bit 0-7: valeur de la carte (0 à 12)
    # Bit 8-11: couleur (1, 2, 4, 8)
    # Bit 12-15: non utilisé
    # Bit 16-31: valeur premier (représentation binaire)
    card_int = (prime << 16) | (suit << 12) | (val << 8) | prime
    return card_int

def card_to_str(card_int: int) -> str:
    """
    Convertit un entier de carte en chaîne lisible (ex: 'A♥', '10♣').
    """
    val = (card_int >> 8) & 0xF
    suit = (card_int >> 12) & 0xF
    
    val_char = STR_VALS[val]
    
    # Récupérer le symbole de la couleur
    suit_char = '?'
    for s_char, s_val in SUIT_MAP.items():
        if s_val == suit:
            suit_char = SUIT_SYMBOLS[s_char]
            break
            
    return f"{val_char}{suit_char}"

def get_deck() -> np.ndarray:
    """
    Retourne un paquet complet de 52 cartes mélangé sous forme de tableau NumPy.
    """
    deck = []
    for val in STR_VALS:
        for suit in STR_SUITS:
            deck.append(make_card(val + suit))
    return np.array(deck, dtype=np.int32)
