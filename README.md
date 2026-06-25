# TwentyRL Arena — 2048 with AI

A 2048 game engine with five AI algorithms running side-by-side, built as a Flask web application. Each algorithm can play the game autonomously, and a real-time dashboard shows decision reasoning, board analysis, and algorithm comparisons.

## AI Algorithms

| Algorithm | Approach |
|-----------|----------|
| **Q-Learning** | Tabular RL agent trained over hundreds of episodes |
| **MCTS** | Monte Carlo Tree Search with UCB1 selection |
| **Hill Climbing** | Greedy local search with board heuristic |
| **Simulated Annealing** | Temperature-based search that escapes local optima |
| **Genetic Algorithm** | Evolves move sequences via crossover & mutation |

The Q-Learning agent is augmented with two extra pillars:
- **Logic Engine** — Propositional rules + Modus Ponens inference (e.g. "if max tile is in corner AND many empty cells → maintain corner")
- **Math Models** — Linear algebra metrics (Frobenius norm, monotonicity, smoothness) and probability analysis (Shannon entropy, survival probability) used for reward shaping

## Features

- Play manually or let any of the 5 AI agents play
- Watch Q-values update in real time
- Compare all 5 algorithms on the same board state
- Training dashboard with live charts (score, epsilon, Q-table size)
- Logic Engine visualization: fired rules, truth tables, resolution steps
- Mathematical analysis panel: eigenvalues, matrix rank, tile distribution
- Train a new agent from the UI without restarting the server
- Docker support for one-command deployment

## Quick Start

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

### Docker

```bash
docker compose up --build
```

Open [http://localhost:5000](http://localhost:5000)

## Train the Agent

**From the UI:** Open the *Technical Monitoring* tab → click *Start Training*.

**From the terminal:**

```bash
python train.py
```

This trains for 300 episodes by default and saves `agent_final.pkl` + `training_history.json`. The app loads the model automatically on next start.

## Run Tests

```bash
python test_suite.py
```

Covers game logic, Q-Learning, reward shaping, logic engine, math models, and MCTS.

## Project Structure

```
├── app.py               # Flask routes & API endpoints
├── game_2048.py         # Game engine (board, moves, scoring)
├── rl_agent.py          # Q-Learning agent (Logic + Math + MCTS augmented)
├── logic_engine.py      # Propositional logic, Modus Ponens, Resolution
├── math_models.py       # Linear algebra & probability models
├── mcts.py              # Monte Carlo Tree Search
├── optimization.py      # Hill Climbing, Simulated Annealing, Genetic Algorithm
├── train.py             # Offline training pipeline
├── test_suite.py        # Automated test suite
├── index.html           # Frontend (single-page app)
├── agent_final.pkl      # Pre-trained Q-table
├── training_history.json
├── Dockerfile
└── docker-compose.yml
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/new_game` | Start a new game |
| POST | `/api/move` | Manual player move |
| POST | `/api/ai_move` | Q-Learning move |
| POST | `/api/ai_move_full` | Move using all 3 pillars |
| POST | `/api/ai_move_algo` | Move using a chosen algorithm |
| POST | `/api/compare_algorithms` | Run all 5 algorithms on current state |
| POST | `/api/analyze` | Board state analysis |
| POST | `/api/logic_analysis` | Logic engine output |
| POST | `/api/math_analysis` | Linear algebra + probability output |
| POST | `/api/mcts_analysis` | MCTS tree statistics |
| POST | `/api/start_training` | Start background training |
| GET | `/api/training_state` | Poll training progress |
| GET | `/api/training_history` | Training metrics for charts |

## Tech Stack

- Python 3.12
- Flask 3.x
- NumPy
- Gunicorn (production)
- Vanilla JS + CSS (no frontend framework)
