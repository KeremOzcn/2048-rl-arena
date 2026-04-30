"""
=====================================================
RL Agent - Q-Learning + Logic + MCTS for 2048
=====================================================
ML/RL Engineer: Reinforcement Learning implementation
Principles of AI — Applied Group Project
=====================================================

State: 4x4 board (log2 transformed for compactness)
Action: 0=up, 1=down, 2=left, 3=right
Reward: Score gained + max-tile bonus + invalid-move penalty

This agent integrates three AI pillars:
  1. LOGIC   — LogicEngine for rule-based inference (Modus Ponens)
  2. MATH    — Linear Algebra norms & Probability expected values
  3. OPTIMIZATION — MCTS for look-ahead decision making
"""

import numpy as np
import random
import pickle
import os
from collections import defaultdict

from logic_engine import LogicEngine
from math_models import LinearAlgebraModels, ProbabilityModels
from mcts import MCTS


class QLearningAgent:
    """Tabular Q-Learning Agent augmented with Logic, Math, and MCTS."""

    def __init__(self, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.actions = [0, 1, 2, 3]

        # --- AI Pillars ---
        self.logic_engine = LogicEngine()
        self.mcts = MCTS(iterations=50, rollout_depth=10)
        self.la = LinearAlgebraModels()
        self.pm = ProbabilityModels()

        # Config: how much to weight each pillar
        self.logic_weight = 0.3    # Weight for logic bias
        self.mcts_weight = 0.5     # Weight for MCTS guidance
        self.use_mcts = False      # MCTS off during training (expensive)

        # Monitoring metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_scores': [],
            'episode_max_tiles': [],
            'episode_moves': [],
            'epsilon_history': [],
            'q_table_size': []
        }

    def _state_to_key(self, state):
        """4x4 board → log2 transformed tuple (hashable)"""
        log_state = np.zeros_like(state)
        log_state[state > 0] = np.log2(state[state > 0]).astype(int)
        # Use plain Python ints for consistent hashing across save/load
        return tuple(int(x) for x in log_state.flatten())

    def choose_action(self, state, valid_actions=None):
        """
        Epsilon-greedy policy augmented with Logic Engine bias.
        During exploitation, combines Q-values with logical inference.
        """
        if valid_actions is None:
            valid_actions = self.actions

        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitation: Q-values + Logic bias + (optional) MCTS
        state_key = self._state_to_key(state)
        if state_key in self.q_table:
            q_values = self.q_table[state_key].copy()
        else:
            q_values = np.zeros(4)

        # --- LOGIC PILLAR: Apply Modus Ponens inference ---
        logic_bias = self.logic_engine.get_action_bias(state)
        combined = q_values + logic_bias * self.logic_weight

        # --- MATH PILLAR: Use norm-based board quality as tiebreaker ---
        # No direct action modification; math is used in reward shaping
        # (see compute_reward function)

        # --- OPTIMIZATION PILLAR: MCTS look-ahead ---
        if self.use_mcts:
            try:
                mcts_action, mcts_stats = self.mcts.search(state, score=0)
                # Boost the MCTS-recommended action
                combined[mcts_action] += self.mcts_weight * 2.0
            except Exception:
                pass  # Fallback to Q+Logic only

        valid_q = {a: combined[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)

    def choose_action_with_mcts(self, state, score=0):
        """
        Decision making with all three pillars active.
        Used during inference/demo (not training).

        Returns:
            action (int), decision_info (dict)
        """
        state_key = self._state_to_key(state)

        # Q-values
        if state_key in self.q_table:
            q_values = self.q_table[state_key].copy()
        else:
            q_values = np.zeros(4)

        # Logic inference
        logic_result = self.logic_engine.evaluate_state(state)
        logic_bias = self.logic_engine.get_action_bias(state)

        # Math analysis
        math_analysis = {
            'frobenius_norm': round(self.la.frobenius_norm(state), 2),
            'board_entropy': round(self.pm.board_entropy(state), 4),
            'survival_prob': round(self.pm.survival_probability(state), 4),
        }

        # MCTS search
        mcts_action, mcts_stats = self.mcts.search(state, score)

        # Combine all signals
        combined = q_values.copy()
        combined += logic_bias * self.logic_weight
        combined[mcts_action] += self.mcts_weight * 2.0

        best_action = int(np.argmax(combined))
        actions = ['up', 'down', 'left', 'right']

        decision_info = {
            'q_values': {actions[i]: round(float(q_values[i]), 3) for i in range(4)},
            'logic_bias': {actions[i]: round(float(logic_bias[i]), 3) for i in range(4)},
            'combined_scores': {actions[i]: round(float(combined[i]), 3) for i in range(4)},
            'logic_strategy': logic_result['strategy'],
            'logic_rules_fired': logic_result['num_rules_fired'],
            'mcts_action': actions[mcts_action],
            'mcts_iterations': mcts_stats['iterations'],
            'mcts_elapsed_ms': mcts_stats['elapsed_ms'],
            'mcts_action_stats': mcts_stats.get('action_stats', {}),
            'math': math_analysis,
            'final_action': actions[best_action],
        }

        return best_action, decision_info

    def get_q_values(self, state):
        """Return Q-values for all actions in this state (read-only, doesn't modify table)"""
        state_key = self._state_to_key(state)
        if state_key in self.q_table:
            return self.q_table[state_key].tolist()
        return [0.0, 0.0, 0.0, 0.0]

    def learn(self, state, action, reward, next_state, done):
        """Q-Learning update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        current_q = self.q_table[state_key][action]
        target_q = reward if done else (reward + self.gamma * np.max(self.q_table[next_state_key]))
        self.q_table[state_key][action] += self.lr * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filepath):
        """Load Q-table from pickle file. Handles version/format mismatches gracefully."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                loaded = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not unpickle {filepath}: {e}")
            return False

        if not isinstance(loaded, dict):
            print(f"[WARN] Loaded model is not a dict (got {type(loaded).__name__})")
            return False

        # Normalize keys to plain Python int tuples (consistent hashing)
        normalized = {}
        for k, v in loaded.items():
            try:
                key = tuple(int(x) for x in k)
                # Ensure value is a numpy array of length 4
                arr = np.asarray(v, dtype=float)
                if arr.shape != (4,):
                    continue
                normalized[key] = arr
            except (TypeError, ValueError):
                continue

        if not normalized:
            print(f"[WARN] No valid entries found in {filepath}")
            return False

        self.q_table = defaultdict(lambda: np.zeros(4), normalized)
        return True

    def get_stats(self):
        return {
            'q_table_size': len(self.q_table),
            'epsilon': float(self.epsilon),
            'lr': self.lr,
            'gamma': self.gamma,
        }

    def get_logic_summary(self, state):
        """Get logic engine analysis for current state."""
        return self.logic_engine.evaluate_state(state)

    def get_math_summary(self, state):
        """Get mathematical analysis for current state."""
        la_results = self.la.full_analysis(state)
        prob_results = self.pm.full_analysis(state)
        return {
            'linear_algebra': la_results,
            'probability': prob_results,
        }


def compute_reward(game, score_gained, changed, old_max_tile, new_max_tile):
    """
    Custom reward shaping for 2048.

    Incorporates Math of AI concepts:
    - Logarithmic scoring (log₂)
    - Norm-based board quality assessment
    - Probability-based survival awareness
    """
    reward = 0

    # Base reward from score
    if score_gained > 0:
        reward += np.log2(score_gained + 1)

    # New max tile bonus
    if new_max_tile > old_max_tile:
        reward += np.log2(new_max_tile) * 2

    # Invalid move penalty
    if not changed:
        reward -= 5

    # Game over penalties/bonuses
    if game.game_over:
        if game.get_max_tile() < 512:
            reward -= 20
        elif game.get_max_tile() >= 2048:
            reward += 100

    # --- MATH PILLAR: Board quality via Linear Algebra ---
    state = game.get_state()

    # Smoothness bonus (uses matrix gradient analysis)
    smoothness = LinearAlgebraModels.smoothness_score(state)
    reward += smoothness * 0.05  # Small smoothness bonus

    # Monotonicity bonus
    mono = LinearAlgebraModels.monotonicity_score(state)
    reward += mono['total'] * 0.02

    # --- PROBABILITY PILLAR: Survival awareness ---
    survival_prob = ProbabilityModels.survival_probability(state)
    if survival_prob < 0.5:
        reward -= 3.0  # Penalize dangerous states

    # Entropy bonus (encourage diverse tiles early, not late)
    entropy = ProbabilityModels.board_entropy(state)
    empty_count = int(np.sum(state == 0))
    if empty_count > 8:
        reward += entropy * 0.1  # Early game: diversity good
    else:
        reward -= entropy * 0.05  # Late game: focus tiles

    return float(reward)


if __name__ == "__main__":
    agent = QLearningAgent()
    print(f"Agent created. Initial epsilon: {agent.epsilon}")
    print(f"Logic Engine: {len(agent.logic_engine.rules)} rules loaded")
    print(f"MCTS: {agent.mcts.iterations} iterations configured")
