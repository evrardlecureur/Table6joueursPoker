from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import threading

# Ajouter la racine du projet pour importer src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.game_state import GameState, ACTION_FOLD, ACTION_CALL, ACTION_RAISE
from src.engine.card import card_to_str
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.deep_cfr_agent import DeepCFRAgent

app_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(app_dir, 'static'))

game_lock = threading.Lock()
game = None
agents = []
hero_seat = 0
hand_started = False  # True seulement après que le joueur clique "Distribuer"


def init_game():
    global game, agents
    sb, bb, ante = 800, 1600, 200
    game = GameState(num_players=6, start_chips=20000, sb=sb, bb=bb, ante=ante)

    agents = [
        None,  # Hero (siège 0)
        HeuristicAgent("Vilain_Nit"),
        HeuristicAgent("Vilain_Station"),
        HeuristicAgent("Vilain_Maniac"),
        HeuristicAgent("Vilain_Reg"),
        HeuristicAgent("Vilain_Fish")
    ]


init_game()


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/state')
def get_state():
    with game_lock:
        if game is None or not hand_started:
            return jsonify({"status": "no_game"})

        players_data = []
        for i, p in enumerate(game.players):
            show_cards = (i == hero_seat) or (game.current_round == 4)
            c_str = [card_to_str(c) for c in p.cards] if show_cards else ["??", "??"]
            if not p.in_play and i != hero_seat:
                c_str = []

            players_data.append({
                "id": i,
                "name": "HERO (Toi)" if i == hero_seat else agents[i].name,
                "chips": p.chips,
                "bet": p.current_bet,
                "in_play": p.in_play,
                "is_all_in": p.is_all_in,
                "cards": c_str,
            })

        board = [card_to_str(c) for c in game.board if c != 0]

        valid_actions = []
        if game.current_player_idx == hero_seat and game.current_round != 4:
            valid_actions = game.get_valid_actions()

        # Pot total = pot collecté + mises en cours (comme sur Winamax)
        total_pot = game.pot + sum(p.current_bet for p in game.players)

        # Diagnostic serveur
        if game.current_player_idx == hero_seat and game.current_round != 4:
            print(f"--- HERO TURN --- Round: {game.current_round}, "
                  f"Highest Bet: {game.highest_bet}, Pot: {game.pot}", flush=True)
            for p in game.players:
                print(f"  P{p.seat_id} chips:{p.chips} bet:{p.current_bet} "
                      f"in_play:{p.in_play}", flush=True)
            print(f"  Valid actions: {valid_actions}", flush=True)

        # Enrichir last_winners avec les noms (le moteur ne connaît que les seat_id)
        winners_data = []
        for w in getattr(game, "last_winners", []):
            seat = w["id"]
            name = "HERO (Toi)" if seat == hero_seat else agents[seat].name
            winners_data.append({**w, "name": name})

        return jsonify({
            "status": "ok",
            "pot": total_pot,
            "round": game.current_round,
            "current_player": game.current_player_idx,
            "players": players_data,
            "board": board,
            "valid_actions": valid_actions,
            "bb": game.bb,
            "last_winners": winners_data,
        })


@app.route('/api/bot_action', methods=['POST'])
def bot_action():
    with game_lock:
        if game.current_round != 4:
            cp = game.current_player_idx
            # Si le pointeur est sur le hero mais qu'il a foldé, on le saute
            if cp == hero_seat and not game.players[hero_seat].in_play:
                # Forcer le moteur à passer au joueur suivant
                game.move_to_next_player()
                cp = game.current_player_idx

            if cp != hero_seat and game.players[cp].in_play:
                agent = agents[cp]
                action, amount = agent.select_action(game)
                print(f"  BOT {agent.name} (seat {cp}) -> action={action}, amount={amount}", flush=True)
                game.execute_action(action, amount)
    return get_state()


@app.route('/api/next_hand', methods=['POST'])
def next_hand():
    with game_lock:
        global hand_started
        hand_started = True
        bb = game.bb

        # Rebuy les joueurs éliminés pour garder la table à 6
        for i in range(6):
            if game.players[i].chips < game.bb:
                game.players[i].chips = 20 * bb

        # Le bouton dealer tourne d'un cran
        new_dealer = (game.dealer_idx + 1) % game.num_players
        game.reset_hand(dealer_idx=new_dealer, randomize_stacks=False)

        print(f"\n=== NOUVELLE MAIN === Dealer: P{new_dealer}, "
              f"Current: P{game.current_player_idx}", flush=True)

    return get_state()


@app.route('/api/action', methods=['POST'])
def play_action():
    data = request.json
    action = int(data.get('action'))
    amount = int(data.get('amount', 0))

    with game_lock:
        if game.current_player_idx == hero_seat:
            print(f"  HERO -> action={action}, amount={amount}", flush=True)
            game.execute_action(action, amount)

    return get_state()


if __name__ == '__main__':
    print("🚀 Serveur Winamax SPACE KO Arena lancé sur http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
