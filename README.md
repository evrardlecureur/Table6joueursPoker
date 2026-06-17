# Table 6 Joueurs Poker - Winamax SPACE KO Arena

Une web app de poker en ligne (Texas Hold'em 6-Max) intégrant un moteur de jeu 100% mathématique sans hardcoding, ainsi qu'une IA et des bots heuristiques pour simuler une vraie table.

## Fonctionnalités

* **Moteur de Poker Complet** : Gestion des règles du Texas Hold'em, calcul d'équité en temps réel, side pots, etc.
* **Bots Heuristiques** : 5 profils de joueurs (Nit, Station, Maniac, Reg, Fish) aux comportements réalistes.
* **Support IA (Deep CFR)** : Possibilité de brancher un réseau de neurones pré-entraîné.
* **Interface Web** : Visuels clairs inspirés de Winamax, logs détaillés des showdowns, affichage dynamique des jetons et Big Blinds.

## Installation

1. Assurez-vous d'avoir **Python 3.8+** installé.
2. Clonez ce dépôt.
3. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Les dépendances principales incluent Flask, numpy, et PyTorch si vous utilisez l'IA)*

## Lancer la Table de Poker

Pour démarrer l'application web et jouer contre les bots heuristiques :

```bash
python web_app/app.py
```

Le serveur démarrera en local. Ouvrez votre navigateur et rendez-vous sur :
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Commandes en jeu
- **Distribuer** : Démarre une nouvelle main.
- **Se coucher (Fold)** : Abandonne la main.
- **Suivre/Check (Call/Check)** : Suit la mise actuelle ou check.
- **Relancer (Raise)** : Utilise le slider pour définir le montant de la relance.
