"""
=====================================================
MCTS — Monte Carlo Tree Search for 2048
=====================================================
Optimization Specialist: MCTS implementation
Principles of AI — Applied Group Project
=====================================================

Implements the four phases of MCTS:
1. Selection   — UCB1-based tree traversal
2. Expansion   — Add new child nodes
3. Simulation  — Random rollout (playout)
4. Backpropagation — Update node statistics

The MCTS agent can be used standalone or as a
decision augmentation layer on top of Q-Learning.
"""

import numpy as np
import math
import random
import time
import copy


# ---------------------------------------------------------------
# 1. MCTS NODE
# ---------------------------------------------------------------

class MCTSNode:
    """A single node in the MCTS search tree."""

    def __init__(self, state, parent=None, action=None):
        """
        Args:
            state: 4x4 numpy board state
            parent: parent MCTSNode
            action: action (0-3) that led to this state
        """
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.children = {}          # action -> MCTSNode
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [0, 1, 2, 3]  # up, down, left, right
        random.shuffle(self.untried_actions)

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    @property
    def average_reward(self):
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, exploration_constant=1.414):
        """
        Upper Confidence Bound formula:
        UCB1 = average_reward + C * sqrt(ln(parent.visits) / visits)

        Balances exploitation (high average reward) with
        exploration (less-visited nodes).
        """
        if self.visits == 0:
            return float('inf')
        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant=1.414):
        """Select child with highest UCB1 value."""
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_constant)
        )


# ---------------------------------------------------------------
# 2. GAME SIMULATOR (lightweight copy for MCTS rollouts)
# ---------------------------------------------------------------

class GameSimulator:
    """Lightweight 2048 game simulator for MCTS rollouts."""

    def __init__(self, board, score=0):
        self.board = board.copy()
        self.score = score
        self.game_over = False

    def _compress_row(self, row):
        new_row = [v for v in row if v != 0]
        new_row += [0] * (4 - len(new_row))
        return new_row

    def _merge_row(self, row):
        score_gained = 0
        for i in range(3):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                score_gained += row[i]
                row[i + 1] = 0
        return row, score_gained

    def move(self, action):
        """Execute a move. Returns (changed, score_gained)."""
        old_board = self.board.copy()
        score_gained = 0

        if action == 2:    # left
            for i in range(4):
                row = self._compress_row(self.board[i].tolist())
                row, sg = self._merge_row(row)
                row = self._compress_row(row)
                self.board[i] = row
                score_gained += sg
        elif action == 3:  # right
            self.board = np.fliplr(self.board)
            for i in range(4):
                row = self._compress_row(self.board[i].tolist())
                row, sg = self._merge_row(row)
                row = self._compress_row(row)
                self.board[i] = row
                score_gained += sg
            self.board = np.fliplr(self.board)
        elif action == 0:  # up
            self.board = self.board.T
            for i in range(4):
                row = self._compress_row(self.board[i].tolist())
                row, sg = self._merge_row(row)
                row = self._compress_row(row)
                self.board[i] = row
                score_gained += sg
            self.board = self.board.T
        elif action == 1:  # down
            self.board = self.board.T
            self.board = np.fliplr(self.board)
            for i in range(4):
                row = self._compress_row(self.board[i].tolist())
                row, sg = self._merge_row(row)
                row = self._compress_row(row)
                self.board[i] = row
                score_gained += sg
            self.board = np.fliplr(self.board)
            self.board = self.board.T

        changed = not np.array_equal(old_board, self.board)
        if changed:
            self.score += score_gained
            self._add_random_tile()

        # Check game over
        if not np.any(self.board == 0):
            game_over = True
            for i in range(4):
                for j in range(3):
                    if self.board[i][j] == self.board[i][j+1]:
                        game_over = False
                        break
                    if self.board[j][i] == self.board[j+1][i]:
                        game_over = False
                        break
                if not game_over:
                    break
            self.game_over = game_over

        return changed, score_gained

    def _add_random_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            r, c = random.choice(empty)
            self.board[r, c] = 2 if random.random() < 0.9 else 4

    def get_valid_actions(self):
        """Return list of actions that actually change the board."""
        valid = []
        for a in range(4):
            sim = GameSimulator(self.board, self.score)
            changed, _ = sim.move(a)
            if changed:
                valid.append(a)
        return valid

    def clone(self):
        sim = GameSimulator(self.board, self.score)
        sim.game_over = self.game_over
        return sim


# ---------------------------------------------------------------
# 3. MCTS ALGORITHM
# ---------------------------------------------------------------

class MCTS:
    """
    Monte Carlo Tree Search for 2048.

    Phases:
    1. Selection   — Traverse tree using UCB1
    2. Expansion   — Add a new child node
    3. Simulation  — Random rollout to terminal or depth limit
    4. Backpropagation — Update visit counts and rewards
    """

    def __init__(self, iterations=100, rollout_depth=20,
                 exploration_constant=1.414, time_limit_ms=None):
        """
        Args:
            iterations: number of MCTS iterations per decision
            rollout_depth: max depth for random rollout
            exploration_constant: C in UCB1 formula
            time_limit_ms: optional time limit (overrides iterations)
        """
        self.iterations = iterations
        self.rollout_depth = rollout_depth
        self.exploration_constant = exploration_constant
        self.time_limit_ms = time_limit_ms
        self.last_search_stats = None

    def search(self, board, score=0):
        """
        Run MCTS from the given board state.

        Returns:
            best_action (int), search_stats (dict)
        """
        start_time = time.perf_counter()
        root = MCTSNode(board)

        iteration_count = 0

        while True:
            # Check stopping condition
            if self.time_limit_ms:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms >= self.time_limit_ms:
                    break
            else:
                if iteration_count >= self.iterations:
                    break

            # --- Phase 1: SELECTION ---
            node = root
            sim = GameSimulator(board, score)
            while node.is_fully_expanded and node.children:
                node = node.best_child(self.exploration_constant)
                if node.action is not None:
                    sim.move(node.action)

            # --- Phase 2: EXPANSION ---
            if not sim.game_over and node.untried_actions:
                action = node.untried_actions.pop()
                child_sim = sim.clone()
                changed, _ = child_sim.move(action)

                if changed:
                    child_node = MCTSNode(
                        child_sim.board, parent=node, action=action
                    )
                    node.children[action] = child_node
                    node = child_node
                    sim = child_sim

            # --- Phase 3: SIMULATION (Random Rollout) ---
            rollout_reward = self._rollout(sim)

            # --- Phase 4: BACKPROPAGATION ---
            while node is not None:
                node.visits += 1
                node.total_reward += rollout_reward
                node = node.parent

            iteration_count += 1

        # Select best action (most visited child)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if not root.children:
            # No valid moves found
            self.last_search_stats = {
                'iterations': iteration_count,
                'elapsed_ms': round(elapsed_ms, 2),
                'action_stats': {},
                'best_action': 0,
            }
            return 0, self.last_search_stats

        # Gather stats for all actions
        action_stats = {}
        for action, child in root.children.items():
            dirs = ['up', 'down', 'left', 'right']
            action_stats[dirs[action]] = {
                'visits': child.visits,
                'avg_reward': round(child.average_reward, 2),
                'ucb1': round(child.ucb1(self.exploration_constant), 3),
            }

        best_action = max(root.children.keys(),
                          key=lambda a: root.children[a].visits)

        self.last_search_stats = {
            'iterations': iteration_count,
            'elapsed_ms': round(elapsed_ms, 2),
            'action_stats': action_stats,
            'best_action': best_action,
            'best_action_name': ['up', 'down', 'left', 'right'][best_action],
        }

        return best_action, self.last_search_stats

    def _rollout(self, sim):
        """
        Random rollout (playout) from current state.
        Returns normalized reward based on score gained and max tile.
        """
        rollout_sim = sim.clone()
        initial_score = rollout_sim.score

        for _ in range(self.rollout_depth):
            if rollout_sim.game_over:
                break
            action = random.randint(0, 3)
            changed, _ = rollout_sim.move(action)
            if not changed:
                # Try other actions
                for alt in range(4):
                    changed, _ = rollout_sim.move(alt)
                    if changed:
                        break
                if not changed:
                    break

        # Reward: normalized score gain + max tile bonus
        score_gain = rollout_sim.score - initial_score
        max_tile = int(np.max(rollout_sim.board))
        empty_cells = int(np.sum(rollout_sim.board == 0))

        # Composite reward
        reward = (score_gain / 100.0 +
                  np.log2(max_tile + 1) * 0.5 +
                  empty_cells * 0.1)

        if rollout_sim.game_over:
            reward -= 2.0

        return reward

    def get_summary(self):
        """Return JSON-serializable summary of last search."""
        if self.last_search_stats is None:
            return {'status': 'no_search_yet'}
        return self.last_search_stats


# ---------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MCTS — Monte Carlo Tree Search Demo")
    print("=" * 60)

    board = np.array([
        [256, 128, 64, 32],
        [16,    8,  4,  2],
        [2,     4,  0,  0],
        [0,     0,  0,  0]
    ])

    mcts = MCTS(iterations=200, rollout_depth=15)
    best_action, stats = mcts.search(board, score=1000)

    dirs = ['Up', 'Down', 'Left', 'Right']
    print(f"\nBest action: {dirs[best_action]}")
    print(f"Iterations: {stats['iterations']}")
    print(f"Time: {stats['elapsed_ms']} ms")

    print("\nAction statistics:")
    for action_name, s in stats['action_stats'].items():
        print(f"  {action_name}: visits={s['visits']}, "
              f"avg_reward={s['avg_reward']:.2f}, UCB1={s['ucb1']:.3f}")
