"""
=====================================================
Training Pipeline - Train RL Agent
=====================================================
ML/RL Engineer: Training pipeline & evaluation
Software Project Management & Technical Monitoring
=====================================================
"""

import numpy as np
import time
import json
import os
from game_2048 import Game2048
from rl_agent import QLearningAgent, compute_reward


def train_agent(episodes=300, save_path='agent_final.pkl',
                history_path='training_history.json', verbose=True):
    """Train the RL agent and save model + training history"""
    game = Game2048()
    agent = QLearningAgent()

    if verbose:
        print("=" * 60)
        print(f"TRAINING — {episodes} episodes")
        print(f"  α={agent.lr}  γ={agent.gamma}  ε={agent.epsilon}")
        print("=" * 60)

    start_time = time.time()
    best_score = 0
    best_max_tile = 0

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        steps = 0

        while not game.game_over and steps < 5000:
            old_max_tile = game.get_max_tile()
            action = agent.choose_action(state)
            changed, score_gained, done = game.make_move(action)
            new_state = game.get_state()
            new_max_tile = game.get_max_tile()
            reward = compute_reward(game, score_gained, changed,
                                    old_max_tile, new_max_tile)
            agent.learn(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        agent.training_history['episode_rewards'].append(total_reward)
        agent.training_history['episode_scores'].append(game.score)
        agent.training_history['episode_max_tiles'].append(game.get_max_tile())
        agent.training_history['episode_moves'].append(steps)
        agent.training_history['epsilon_history'].append(agent.epsilon)
        agent.training_history['q_table_size'].append(len(agent.q_table))

        if game.score > best_score:
            best_score = game.score
        if game.get_max_tile() > best_max_tile:
            best_max_tile = game.get_max_tile()

        if verbose and (episode + 1) % 50 == 0:
            avg_score = np.mean(agent.training_history['episode_scores'][-50:])
            avg_tile = np.mean(agent.training_history['episode_max_tiles'][-50:])
            print(f"Ep {episode + 1:5d} | Avg Score: {avg_score:7.1f} | "
                  f"Avg MaxTile: {avg_tile:5.0f} | "
                  f"Best: {best_score} ({best_max_tile}) | "
                  f"ε: {agent.epsilon:.3f} | Q-size: {len(agent.q_table):,}")

    elapsed = time.time() - start_time

    if verbose:
        print("=" * 60)
        print(f"COMPLETED in {elapsed:.1f}s")
        print(f"  Best Score: {best_score} | Best Max Tile: {best_max_tile}")
        print(f"  Q-Table Size: {len(agent.q_table):,}")
        print("=" * 60)

    # Save artifacts
    agent.save_model(save_path)

    history_serializable = {
        k: [float(v) for v in vals]
        for k, vals in agent.training_history.items()
    }
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)

    return agent


def evaluate_agent(agent, num_games=10, verbose=True):
    """Evaluate trained agent (epsilon=0)"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    game = Game2048()
    results = {'scores': [], 'max_tiles': [], 'moves': [], 'wins': 0}

    for i in range(num_games):
        game.reset()
        state = game.get_state()
        steps = 0
        consecutive_invalid = 0

        while not game.game_over and steps < 5000:
            action = agent.choose_action(state)
            changed, _, _ = game.make_move(action)

            if not changed:
                consecutive_invalid += 1
                if consecutive_invalid >= 4:
                    game.game_over = True
                    break
                for alt in [0, 1, 2, 3]:
                    if alt != action:
                        changed, _, _ = game.make_move(alt)
                        if changed:
                            consecutive_invalid = 0
                            break
            else:
                consecutive_invalid = 0

            state = game.get_state()
            steps += 1

        results['scores'].append(game.score)
        results['max_tiles'].append(game.get_max_tile())
        results['moves'].append(steps)
        if game.get_max_tile() >= 2048:
            results['wins'] += 1

        if verbose:
            print(f"Game {i+1:2d}: Score={game.score:5d} | "
                  f"MaxTile={game.get_max_tile():4d} | Moves={steps}")

    agent.epsilon = original_epsilon
    return results


if __name__ == "__main__":
    agent = train_agent(episodes=300)
    print()
    eval_results = evaluate_agent(agent, num_games=5)
    print(f"\nAvg Score: {np.mean(eval_results['scores']):.1f}")
    print(f"Avg Max Tile: {np.mean(eval_results['max_tiles']):.1f}")
