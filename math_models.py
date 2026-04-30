"""
=====================================================
Math Models — Linear Algebra & Probability
=====================================================
Mathematical Modeler: AI foundation mathematics
Principles of AI — Applied Group Project
=====================================================

This module implements:
1. LINEAR ALGEBRA
   - Matrix representation of board states
   - Frobenius norm (board complexity measure)
   - L2 norm (state vector magnitude)
   - Cosine similarity (state comparison)
   - Eigenvalue analysis (board structure)
   - Matrix transformations

2. PROBABILITY
   - Tile probability distributions
   - Expected value calculations
   - Conditional probability for merges
   - State transition probabilities
   - Entropy of board state
"""

import numpy as np
from collections import Counter


# ===============================================================
# 1. LINEAR ALGEBRA
# ===============================================================

class LinearAlgebraModels:
    """
    Linear algebra operations on 2048 board states.
    The board is treated as a 4×4 matrix for mathematical analysis.
    """

    @staticmethod
    def frobenius_norm(state):
        """
        Frobenius Norm: ||A||_F = sqrt(Σᵢ Σⱼ |aᵢⱼ|²)

        Measures overall magnitude/complexity of the board.
        Higher values = more advanced game state.
        """
        return float(np.linalg.norm(state, 'fro'))

    @staticmethod
    def l2_norm(state):
        """
        L2 Norm (Euclidean norm) of flattened state vector.
        ||x||₂ = sqrt(Σ xᵢ²)
        """
        return float(np.linalg.norm(state.flatten(), 2))

    @staticmethod
    def l1_norm(state):
        """
        L1 Norm (Manhattan norm) of flattened state vector.
        ||x||₁ = Σ |xᵢ|
        """
        return float(np.linalg.norm(state.flatten(), 1))

    @staticmethod
    def infinity_norm(state):
        """
        Infinity Norm (max absolute value).
        ||x||∞ = max |xᵢ|
        """
        return float(np.max(np.abs(state)))

    @staticmethod
    def log2_transform(state):
        """
        Log2 transform of board state.
        Maps tile values to their power-of-2 exponents.
        Used to create a more uniform representation.
        """
        log_state = np.zeros_like(state, dtype=float)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])
        return log_state

    @staticmethod
    def cosine_similarity(state_a, state_b):
        """
        Cosine Similarity between two board states:
        cos(θ) = (A · B) / (||A|| × ||B||)

        Values range from -1 (opposite) to 1 (identical).
        Used to measure how similar two game states are.
        """
        vec_a = state_a.flatten().astype(float)
        vec_b = state_b.flatten().astype(float)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def eigenvalue_analysis(state):
        """
        Eigenvalue analysis of the board state matrix.
        Eigenvalues λ satisfy: Av = λv

        Provides insight into the board's structural properties.
        - Dominant eigenvalue → overall tile magnitude
        - Eigenvalue spread → board diversity
        """
        state_float = state.astype(float)
        eigenvalues, eigenvectors = np.linalg.eig(state_float)

        # Sort by magnitude
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return {
            'eigenvalues': eigenvalues.real.tolist(),
            'dominant_eigenvalue': float(np.max(np.abs(eigenvalues))),
            'eigenvalue_spread': float(np.std(np.abs(eigenvalues))),
            'trace': float(np.trace(state_float)),  # sum of eigenvalues
            'determinant': float(np.linalg.det(state_float)),
        }

    @staticmethod
    def matrix_rank(state):
        """
        Matrix rank — number of linearly independent rows/columns.
        Higher rank = more diverse tile distribution.
        """
        return int(np.linalg.matrix_rank(state.astype(float)))

    @staticmethod
    def smoothness_score(state):
        """
        Smoothness: negative sum of absolute differences between adjacent tiles.
        Uses log2 values. Lower (closer to 0) = smoother board = better.

        This is a matrix-based analysis of spatial structure.
        """
        log_state = np.zeros_like(state, dtype=float)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])

        smoothness = 0.0
        for i in range(4):
            for j in range(4):
                if log_state[i, j] > 0:
                    if j < 3 and log_state[i, j+1] > 0:
                        smoothness -= abs(log_state[i, j] - log_state[i, j+1])
                    if i < 3 and log_state[i+1, j] > 0:
                        smoothness -= abs(log_state[i, j] - log_state[i+1, j])

        return float(smoothness)

    @staticmethod
    def monotonicity_score(state):
        """
        Monotonicity: measures how well tiles are ordered along rows/columns.
        Uses log2 values. Higher = more monotonic = better strategic position.

        This uses vector comparison (dot-product based analysis).
        """
        log_state = np.zeros_like(state, dtype=float)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])

        scores = [0.0, 0.0, 0.0, 0.0]  # up, down, left, right

        # Row monotonicity
        for i in range(4):
            for j in range(3):
                if log_state[i, j] > log_state[i, j+1]:
                    scores[2] += log_state[i, j] - log_state[i, j+1]  # left-decreasing
                else:
                    scores[3] += log_state[i, j+1] - log_state[i, j]  # right-decreasing

        # Column monotonicity
        for j in range(4):
            for i in range(3):
                if log_state[i, j] > log_state[i+1, j]:
                    scores[0] += log_state[i, j] - log_state[i+1, j]  # up-decreasing
                else:
                    scores[1] += log_state[i+1, j] - log_state[i, j]  # down-decreasing

        return {
            'total': float(max(scores[0], scores[1]) + max(scores[2], scores[3])),
            'up': float(scores[0]),
            'down': float(scores[1]),
            'left': float(scores[2]),
            'right': float(scores[3]),
        }

    @staticmethod
    def gradient_matrix(state):
        """
        Compute gradient (difference matrix) of the board.
        Shows how tile values change spatially.
        """
        log_state = np.zeros_like(state, dtype=float)
        mask = state > 0
        log_state[mask] = np.log2(state[mask])

        grad_x = np.diff(log_state, axis=1)   # 4x3 horizontal gradient
        grad_y = np.diff(log_state, axis=0)    # 3x4 vertical gradient

        return {
            'horizontal_gradient': grad_x.tolist(),
            'vertical_gradient': grad_y.tolist(),
            'gradient_magnitude': float(np.linalg.norm(
                np.concatenate([grad_x.flatten(), grad_y.flatten()])
            )),
        }

    @staticmethod
    def full_analysis(state):
        """Complete linear algebra analysis of a board state."""
        la = LinearAlgebraModels
        eigen = la.eigenvalue_analysis(state)
        mono = la.monotonicity_score(state)
        grad = la.gradient_matrix(state)

        return {
            'frobenius_norm': round(la.frobenius_norm(state), 2),
            'l2_norm': round(la.l2_norm(state), 2),
            'l1_norm': round(la.l1_norm(state), 2),
            'infinity_norm': round(la.infinity_norm(state), 2),
            'matrix_rank': la.matrix_rank(state),
            'smoothness': round(la.smoothness_score(state), 2),
            'monotonicity': round(mono['total'], 2),
            'dominant_eigenvalue': round(eigen['dominant_eigenvalue'], 2),
            'eigenvalue_spread': round(eigen['eigenvalue_spread'], 2),
            'trace': round(eigen['trace'], 2),
            'determinant': round(eigen['determinant'], 2),
            'gradient_magnitude': round(grad['gradient_magnitude'], 2),
        }


# ===============================================================
# 2. PROBABILITY
# ===============================================================

class ProbabilityModels:
    """
    Probability and statistical models for the 2048 game.
    """

    @staticmethod
    def tile_distribution(state):
        """
        Probability distribution of tile values on the board.
        P(tile = v) = count(v) / total_tiles

        Returns distribution as dict: {value: probability}
        """
        flat = state.flatten()
        non_zero = flat[flat > 0]
        if len(non_zero) == 0:
            return {}

        counter = Counter(non_zero.tolist())
        total = len(non_zero)

        distribution = {}
        for value, count in sorted(counter.items()):
            distribution[int(value)] = round(count / total, 4)

        return distribution

    @staticmethod
    def new_tile_probability():
        """
        Probability distribution for newly spawned tiles.
        P(new_tile = 2) = 0.9
        P(new_tile = 4) = 0.1
        """
        return {2: 0.9, 4: 0.1}

    @staticmethod
    def expected_new_tile_value():
        """
        Expected value of a new tile:
        E[X] = Σ xᵢ × P(xᵢ) = 2×0.9 + 4×0.1 = 2.2
        """
        return 2 * 0.9 + 4 * 0.1  # = 2.2

    @staticmethod
    def expected_score_per_merge(state):
        """
        Expected score gained from a random merge:
        E[merge_score] = Σ (2×v) × P(merge of value v)

        Where P(merge of v) is proportional to adjacent pairs of value v.
        """
        merge_candidates = []

        # Find all adjacent equal pairs
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j] == state[i, j+1]:
                    merge_candidates.append(int(state[i, j]) * 2)

        for i in range(3):
            for j in range(4):
                if state[i, j] > 0 and state[i, j] == state[i+1, j]:
                    merge_candidates.append(int(state[i, j]) * 2)

        if not merge_candidates:
            return 0.0

        # Expected value assuming uniform probability of each merge
        return float(np.mean(merge_candidates))

    @staticmethod
    def expected_value_of_action(state, action, num_simulations=50):
        """
        Estimate E[Score | action] via Monte Carlo sampling.
        E[Score | a] ≈ (1/N) Σ score_after_action_i

        Args:
            state: current board state
            action: 0-3 (up, down, left, right)
            num_simulations: number of MC samples

        Returns:
            expected_score_gain (float)
        """
        from mcts import GameSimulator

        total_gain = 0.0
        valid_sims = 0

        for _ in range(num_simulations):
            sim = GameSimulator(state, 0)
            changed, score_gained = sim.move(action)
            if changed:
                total_gain += score_gained
                valid_sims += 1

        if valid_sims == 0:
            return -1.0  # invalid move

        return total_gain / valid_sims

    @staticmethod
    def expected_values_all_actions(state, num_simulations=30):
        """
        Calculate expected value for all 4 actions.
        E[Score | a] for a ∈ {up, down, left, right}

        Returns:
            dict: {action_name: expected_value}
        """
        actions = ['up', 'down', 'left', 'right']
        ev = {}
        for i, name in enumerate(actions):
            ev[name] = round(
                ProbabilityModels.expected_value_of_action(state, i, num_simulations),
                2
            )
        return ev

    @staticmethod
    def board_entropy(state):
        """
        Shannon Entropy of the board state:
        H(X) = -Σ P(xᵢ) × log₂(P(xᵢ))

        Higher entropy = more diverse tile distribution = more uncertain.
        Lower entropy = dominated by few tile values = more predictable.
        """
        flat = state.flatten()
        non_zero = flat[flat > 0]
        if len(non_zero) == 0:
            return 0.0

        counter = Counter(non_zero.tolist())
        total = len(non_zero)
        entropy = 0.0

        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return float(entropy)

    @staticmethod
    def survival_probability(state):
        """
        Estimate P(survival) — probability of having at least one valid move.
        Based on empty cell count and merge potential.

        P(survive) = 1 - P(no_empty) × P(no_merges)
        """
        empty = int(np.sum(state == 0))
        total_cells = 16

        # P(no empty after next tile)
        if empty > 1:
            p_no_empty = 0.0
        elif empty == 1:
            p_no_empty = 1.0
        else:
            p_no_empty = 1.0

        # Count merge possibilities
        merges = 0
        for i in range(4):
            for j in range(3):
                if state[i, j] > 0 and state[i, j] == state[i, j+1]:
                    merges += 1
            for j in range(3):
                if state[j, i] > 0 and state[j, i] == state[j+1, i]:
                    merges += 1

        # P(no merges) — if many merges possible, unlikely to have no merges
        p_no_merges = 1.0 if merges == 0 else 0.0

        # P(survive) = 1 - P(game_over)
        p_game_over = p_no_empty * p_no_merges
        return 1.0 - p_game_over

    @staticmethod
    def conditional_merge_probability(state, tile_value):
        """
        P(merge | tile = v) — probability that a tile of value v
        can be merged given the current board state.

        Counts adjacent equal tiles for the given value.
        """
        positions = list(zip(*np.where(state == tile_value)))
        if not positions:
            return 0.0

        mergeable = 0
        for r, c in positions:
            if r > 0 and state[r-1, c] == tile_value:
                mergeable += 1
            if r < 3 and state[r+1, c] == tile_value:
                mergeable += 1
            if c > 0 and state[r, c-1] == tile_value:
                mergeable += 1
            if c < 3 and state[r, c+1] == tile_value:
                mergeable += 1

        # P = mergeable_neighbors / total_possible_neighbors
        total_neighbors = len(positions) * 4  # max 4 neighbors each
        return mergeable / total_neighbors if total_neighbors > 0 else 0.0

    @staticmethod
    def full_analysis(state):
        """Complete probability analysis of a board state."""
        pm = ProbabilityModels
        dist = pm.tile_distribution(state)

        return {
            'tile_distribution': dist,
            'expected_new_tile': pm.expected_new_tile_value(),
            'expected_merge_score': round(pm.expected_score_per_merge(state), 2),
            'board_entropy': round(pm.board_entropy(state), 4),
            'survival_probability': round(pm.survival_probability(state), 4),
        }


# ===============================================================
# 3. COMBINED MATH ANALYSIS
# ===============================================================

def full_math_analysis(state):
    """
    Complete mathematical analysis combining linear algebra and probability.
    Returns JSON-serializable dict.
    """
    la_results = LinearAlgebraModels.full_analysis(state)
    prob_results = ProbabilityModels.full_analysis(state)

    return {
        'linear_algebra': la_results,
        'probability': prob_results,
    }


# ---------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MATH MODELS — Linear Algebra & Probability Demo")
    print("=" * 60)

    test_state = np.array([
        [512, 128, 64, 32],
        [16,    8,  4,  2],
        [2,     4,  8,  0],
        [0,     0,  0,  0]
    ])

    # Linear Algebra
    la = LinearAlgebraModels
    print("\n--- Linear Algebra ---")
    print(f"  Frobenius Norm:     {la.frobenius_norm(test_state):.2f}")
    print(f"  L2 Norm:            {la.l2_norm(test_state):.2f}")
    print(f"  L1 Norm:            {la.l1_norm(test_state):.2f}")
    print(f"  Infinity Norm:      {la.infinity_norm(test_state):.2f}")
    print(f"  Matrix Rank:        {la.matrix_rank(test_state)}")
    print(f"  Smoothness:         {la.smoothness_score(test_state):.2f}")

    eigen = la.eigenvalue_analysis(test_state)
    print(f"  Eigenvalues:        {[round(e, 2) for e in eigen['eigenvalues']]}")
    print(f"  Dominant Eigenval:  {eigen['dominant_eigenvalue']:.2f}")
    print(f"  Trace:              {eigen['trace']:.2f}")
    print(f"  Determinant:        {eigen['determinant']:.2f}")

    mono = la.monotonicity_score(test_state)
    print(f"  Monotonicity:       {mono['total']:.2f}")

    # Cosine similarity
    other_state = np.array([
        [256, 64, 32, 16],
        [8,    4,  2,  0],
        [0,    0,  0,  0],
        [0,    0,  0,  0]
    ])
    sim = la.cosine_similarity(test_state, other_state)
    print(f"  Cosine Similarity:  {sim:.4f}")

    # Probability
    pm = ProbabilityModels
    print("\n--- Probability ---")
    dist = pm.tile_distribution(test_state)
    print(f"  Tile Distribution:  {dist}")
    print(f"  Expected New Tile:  {pm.expected_new_tile_value()}")
    print(f"  Expected Merge Score: {pm.expected_score_per_merge(test_state):.2f}")
    print(f"  Board Entropy:      {pm.board_entropy(test_state):.4f}")
    print(f"  Survival Prob:      {pm.survival_probability(test_state):.4f}")
    print(f"  P(merge|tile=4):    {pm.conditional_merge_probability(test_state, 4):.4f}")

    # Combined
    print("\n--- Full Analysis ---")
    full = full_math_analysis(test_state)
    for category, data in full.items():
        print(f"\n  {category}:")
        for key, val in data.items():
            print(f"    {key}: {val}")
