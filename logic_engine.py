"""
=====================================================
Logic Engine — Propositional & First-Order Logic
=====================================================
Logic Engineer: Logical inference for AI decisions
Principles of AI — Applied Group Project
=====================================================

This module implements:
- Propositional Logic (connectives, truth tables)
- Modus Ponens inference rule
- Resolution inference rule
- Game-specific strategy rules for the 2048 agent
- First-Order Logic predicates (objects, properties, relations)

The agent uses these logical rules alongside Q-values
to make more informed decisions.
"""

import numpy as np
from itertools import product


# ---------------------------------------------------------------
# 1. PROPOSITIONAL LOGIC — CONNECTIVES & TRUTH TABLES
# ---------------------------------------------------------------

class Proposition:
    """Represents a propositional logic variable or compound expression."""

    def __init__(self, name, evaluate_fn=None):
        self.name = name
        self.evaluate_fn = evaluate_fn  # function(board_state) -> bool

    def evaluate(self, state):
        """Evaluate this proposition given a game state (4x4 numpy array)."""
        if self.evaluate_fn is not None:
            return bool(self.evaluate_fn(state))
        return False

    def __repr__(self):
        return self.name


def AND(a, b):
    """Logical conjunction: A ∧ B"""
    return a and b


def OR(a, b):
    """Logical disjunction: A ∨ B"""
    return a or b


def NOT(a):
    """Logical negation: ¬A"""
    return not a


def IMPLIES(a, b):
    """Logical implication: A → B  (equivalent to ¬A ∨ B)"""
    return (not a) or b


def BICONDITIONAL(a, b):
    """Logical biconditional: A ↔ B"""
    return a == b


def generate_truth_table(propositions, expression_fn, expression_name="Result"):
    """
    Generate a truth table for given propositions and a compound expression.

    Args:
        propositions: list of proposition names (strings)
        expression_fn: function that takes bool values and returns bool
        expression_name: label for the result column

    Returns:
        dict with 'headers', 'rows' (list of dicts with values and result)
    """
    n = len(propositions)
    rows = []
    for combo in product([True, False], repeat=n):
        assignment = {propositions[i]: combo[i] for i in range(n)}
        result = expression_fn(*combo)
        rows.append({**assignment, expression_name: result})

    return {
        'headers': propositions + [expression_name],
        'rows': rows
    }


# ---------------------------------------------------------------
# 2. MODUS PONENS & RESOLUTION
# ---------------------------------------------------------------

class Rule:
    """
    Represents an inference rule: IF premise THEN conclusion.
    Used for Modus Ponens: If P and (P → Q), then Q.
    """

    def __init__(self, name, premises, conclusion, priority=1):
        """
        Args:
            name: Human-readable rule name
            premises: list of Proposition objects
            conclusion: string describing the action/conclusion
            priority: higher = more important (used for conflict resolution)
        """
        self.name = name
        self.premises = premises
        self.conclusion = conclusion
        self.priority = priority
        self.fired = False

    def evaluate(self, state):
        """Check if all premises are satisfied (Modus Ponens)."""
        results = {}
        all_true = True
        for prop in self.premises:
            val = prop.evaluate(state)
            results[prop.name] = val
            if not val:
                all_true = False
        self.fired = all_true
        return all_true, results

    def __repr__(self):
        premises_str = " ∧ ".join(p.name for p in self.premises)
        return f"Rule({self.name}): {premises_str} → {self.conclusion}"


def modus_ponens(rules, state):
    """
    Apply Modus Ponens across all rules:
    For each rule: IF all premises are True THEN conclude.

    Returns:
        list of (rule, premise_results) for fired rules, sorted by priority
    """
    fired_rules = []
    for rule in rules:
        satisfied, results = rule.evaluate(state)
        if satisfied:
            fired_rules.append({
                'rule_name': rule.name,
                'premises': results,
                'conclusion': rule.conclusion,
                'priority': rule.priority,
            })
    # Sort by priority (highest first)
    fired_rules.sort(key=lambda r: -r['priority'])
    return fired_rules


def resolution(clause_a, clause_b):
    """
    Resolution inference rule:
    Given two clauses (sets of literals), find complementary literals
    and produce the resolvent.

    Each clause is a set of strings, where '~X' means NOT X.
    Example: {'P', '~Q'} and {'Q', 'R'} → {'P', 'R'}

    Returns:
        resolvent set or None if no resolution possible
    """
    for literal in clause_a:
        complement = literal[1:] if literal.startswith('~') else '~' + literal
        if complement in clause_b:
            resolvent = (clause_a | clause_b) - {literal, complement}
            return resolvent
    return None


def resolution_chain(clauses):
    """
    Apply resolution iteratively to a set of clauses.
    Returns all derived resolvents (proof by contradiction).
    """
    clauses = [frozenset(c) for c in clauses]
    new_clauses = set()
    derived = []

    for i in range(len(clauses)):
        for j in range(i + 1, len(clauses)):
            result = resolution(set(clauses[i]), set(clauses[j]))
            if result is not None:
                fs = frozenset(result)
                if fs not in set(clauses) and fs not in new_clauses:
                    new_clauses.add(fs)
                    derived.append({
                        'clause_a': set(clauses[i]),
                        'clause_b': set(clauses[j]),
                        'resolvent': result,
                        'is_empty': len(result) == 0,
                    })
    return derived


# ---------------------------------------------------------------
# 3. GAME-SPECIFIC PROPOSITIONS (2048)
# ---------------------------------------------------------------

# Board property predicates — First-Order Logic style
# ∀x(Tile(x) ∧ InCorner(x) ∧ IsMax(x) → GoodPosition(x))

P_MAX_IN_CORNER = Proposition(
    "MaxTileInCorner",
    lambda s: int(np.max(s)) == s[0, 0] or int(np.max(s)) == s[0, 3] or
              int(np.max(s)) == s[3, 0] or int(np.max(s)) == s[3, 3]
)

P_MAX_ON_EDGE = Proposition(
    "MaxTileOnEdge",
    lambda s: int(np.max(s)) in [int(s[0, j]) for j in range(4)] +
              [int(s[3, j]) for j in range(4)] +
              [int(s[i, 0]) for i in range(4)] +
              [int(s[i, 3]) for i in range(4)]
)

P_MANY_EMPTY = Proposition(
    "ManyEmptyCells",
    lambda s: int(np.sum(s == 0)) >= 6
)

P_FEW_EMPTY = Proposition(
    "FewEmptyCells",
    lambda s: int(np.sum(s == 0)) <= 3
)

P_VERY_FEW_EMPTY = Proposition(
    "VeryFewEmptyCells",
    lambda s: int(np.sum(s == 0)) <= 1
)

P_HIGH_SCORE_POTENTIAL = Proposition(
    "HighScorePotential",
    lambda s: int(np.max(s)) >= 256
)

P_MONOTONIC_ROW = Proposition(
    "MonotonicFirstRow",
    lambda s: all(s[0, j] >= s[0, j+1] for j in range(3) if s[0, j] > 0) or
              all(s[0, j] <= s[0, j+1] for j in range(3))
)

P_ADJACENT_MERGE = Proposition(
    "AdjacentMergePossible",
    lambda s: any(
        s[i, j] == s[i, j+1] and s[i, j] > 0
        for i in range(4) for j in range(3)
    ) or any(
        s[i, j] == s[i+1, j] and s[i, j] > 0
        for i in range(3) for j in range(4)
    )
)

P_LARGE_TILES_CLUSTERED = Proposition(
    "LargeTilesClustered",
    lambda s: _check_clustering(s)
)

P_BOARD_NEARLY_FULL = Proposition(
    "BoardNearlyFull",
    lambda s: int(np.sum(s == 0)) <= 2
)


def _check_clustering(state):
    """Check if large tiles (>=64) are clustered together."""
    large = np.argwhere(state >= 64)
    if len(large) < 2:
        return True
    # Check if large tiles are within 2-Manhattan distance of each other
    for i in range(len(large)):
        for j in range(i + 1, len(large)):
            dist = abs(large[i][0] - large[j][0]) + abs(large[i][1] - large[j][1])
            if dist > 2:
                return False
    return True


# ---------------------------------------------------------------
# 4. STRATEGY RULES (Modus Ponens applied to 2048)
# ---------------------------------------------------------------

# Rule 1: Corner Strategy
# IF MaxTileInCorner ∧ ManyEmptyCells → "maintain_corner" (keep max in corner)
RULE_CORNER_STRATEGY = Rule(
    name="Corner Strategy",
    premises=[P_MAX_IN_CORNER, P_MANY_EMPTY],
    conclusion="maintain_corner",
    priority=8
)

# Rule 2: Merge Priority
# IF AdjacentMergePossible ∧ FewEmptyCells → "prioritize_merge"
RULE_MERGE_PRIORITY = Rule(
    name="Merge Priority",
    premises=[P_ADJACENT_MERGE, P_FEW_EMPTY],
    conclusion="prioritize_merge",
    priority=9
)

# Rule 3: Survival Mode
# IF VeryFewEmptyCells ∧ AdjacentMergePossible → "survival_merge"
RULE_SURVIVAL = Rule(
    name="Survival Mode",
    premises=[P_VERY_FEW_EMPTY, P_ADJACENT_MERGE],
    conclusion="survival_merge",
    priority=10
)

# Rule 4: Edge Protection
# IF MaxTileOnEdge ∧ HighScorePotential → "protect_edge"
RULE_EDGE_PROTECT = Rule(
    name="Edge Protection",
    premises=[P_MAX_ON_EDGE, P_HIGH_SCORE_POTENTIAL],
    conclusion="protect_edge",
    priority=7
)

# Rule 5: Monotonic Building
# IF MonotonicFirstRow ∧ MaxTileInCorner → "build_monotonic"
RULE_MONOTONIC = Rule(
    name="Monotonic Building",
    premises=[P_MONOTONIC_ROW, P_MAX_IN_CORNER],
    conclusion="build_monotonic",
    priority=6
)

# Rule 6: Clustering
# IF LargeTilesClustered ∧ HighScorePotential → "maintain_cluster"
RULE_CLUSTER = Rule(
    name="Cluster Maintenance",
    premises=[P_LARGE_TILES_CLUSTERED, P_HIGH_SCORE_POTENTIAL],
    conclusion="maintain_cluster",
    priority=5
)

# Rule 7: Emergency (Board Nearly Full, no merges) → must find a merge or lose
RULE_EMERGENCY = Rule(
    name="Emergency",
    premises=[P_BOARD_NEARLY_FULL, P_ADJACENT_MERGE],
    conclusion="emergency_merge",
    priority=10
)

# All rules collected
ALL_RULES = [
    RULE_CORNER_STRATEGY,
    RULE_MERGE_PRIORITY,
    RULE_SURVIVAL,
    RULE_EDGE_PROTECT,
    RULE_MONOTONIC,
    RULE_CLUSTER,
    RULE_EMERGENCY,
]


# ---------------------------------------------------------------
# 5. LOGIC ENGINE — Main Interface
# ---------------------------------------------------------------

class LogicEngine:
    """
    Central logic engine that evaluates game state using
    propositional logic and inference rules.
    """

    def __init__(self):
        self.rules = ALL_RULES
        self.propositions = [
            P_MAX_IN_CORNER, P_MAX_ON_EDGE, P_MANY_EMPTY, P_FEW_EMPTY,
            P_VERY_FEW_EMPTY, P_HIGH_SCORE_POTENTIAL, P_MONOTONIC_ROW,
            P_ADJACENT_MERGE, P_LARGE_TILES_CLUSTERED, P_BOARD_NEARLY_FULL,
        ]
        self.last_inference = None

    def evaluate_state(self, state):
        """
        Evaluate all propositions and apply Modus Ponens inference.

        Returns:
            dict with proposition values, fired rules, and strategy recommendation
        """
        # Evaluate all propositions
        prop_values = {}
        for prop in self.propositions:
            prop_values[prop.name] = prop.evaluate(state)

        # Apply Modus Ponens
        fired_rules = modus_ponens(self.rules, state)

        # Determine strategy
        strategy = "explore"  # default
        if fired_rules:
            strategy = fired_rules[0]['conclusion']  # highest priority fired rule

        self.last_inference = {
            'propositions': prop_values,
            'fired_rules': fired_rules,
            'strategy': strategy,
            'num_rules_fired': len(fired_rules),
        }

        return self.last_inference

    def get_action_bias(self, state):
        """
        Returns Q-value bias adjustments based on logical inference.
        This is used to augment Q-learning decisions with logical reasoning.

        Returns:
            numpy array of shape (4,) — bias for [up, down, left, right]
        """
        inference = self.evaluate_state(state)
        bias = np.zeros(4)  # [up, down, left, right]
        strategy = inference['strategy']

        max_pos = np.unravel_index(np.argmax(state), state.shape)

        if strategy == "maintain_corner":
            # Bias toward moves that keep max tile in corner
            if max_pos == (0, 0):
                bias[0] += 1.0  # up
                bias[2] += 1.0  # left
            elif max_pos == (0, 3):
                bias[0] += 1.0  # up
                bias[3] += 1.0  # right
            elif max_pos == (3, 0):
                bias[1] += 1.0  # down
                bias[2] += 1.0  # left
            elif max_pos == (3, 3):
                bias[1] += 1.0  # down
                bias[3] += 1.0  # right

        elif strategy == "prioritize_merge" or strategy == "survival_merge" or strategy == "emergency_merge":
            # Check which direction has the most merges
            merge_counts = _count_merges_per_direction(state)
            bias += merge_counts * 2.0

        elif strategy == "protect_edge":
            # Keep max on edge - avoid moves that pull it inward
            if max_pos[0] == 0:
                bias[0] += 1.5  # favor up
            elif max_pos[0] == 3:
                bias[1] += 1.5  # favor down
            if max_pos[1] == 0:
                bias[2] += 1.5  # favor left
            elif max_pos[1] == 3:
                bias[3] += 1.5  # favor right

        elif strategy == "build_monotonic":
            # Favor moves that maintain monotonicity
            bias[2] += 0.5  # left tends to maintain monotonic rows
            bias[0] += 0.5  # up tends to maintain monotonic columns

        elif strategy == "maintain_cluster":
            # Favor moves that don't scatter large tiles
            merge_counts = _count_merges_per_direction(state)
            bias += merge_counts * 1.0

        return bias

    def get_demonstration_truth_table(self):
        """
        Generate a truth table for a key game strategy rule.
        Demonstrates: (MaxInCorner ∧ ManyEmpty) → MaintainCorner
        Which is: ¬(MaxInCorner ∧ ManyEmpty) ∨ MaintainCorner
        = ¬MaxInCorner ∨ ¬ManyEmpty ∨ MaintainCorner
        """
        return generate_truth_table(
            ['MaxInCorner', 'ManyEmpty'],
            lambda a, b: IMPLIES(AND(a, b), True),
            'MaintainCorner'
        )

    def get_resolution_demo(self):
        """
        Demonstrate Resolution inference:
        Clause 1: {¬FewEmpty, PrioritizeMerge}    (FewEmpty → PrioritizeMerge)
        Clause 2: {FewEmpty}                        (It IS FewEmpty)
        Resolvent: {PrioritizeMerge}               (Therefore: PrioritizeMerge)
        """
        clauses = [
            {'~FewEmpty', 'PrioritizeMerge'},
            {'FewEmpty'},
        ]
        return resolution_chain(clauses)

    def get_summary(self):
        """Return JSON-serializable summary of last inference."""
        if self.last_inference is None:
            return {'status': 'no_inference_yet'}
        return {
            'propositions': self.last_inference['propositions'],
            'fired_rules': self.last_inference['fired_rules'],
            'strategy': self.last_inference['strategy'],
            'num_rules_fired': self.last_inference['num_rules_fired'],
        }


def _count_merges_per_direction(state):
    """Count potential merges for each direction [up, down, left, right]."""
    counts = np.zeros(4)

    # Left merges
    for i in range(4):
        for j in range(3):
            if state[i, j] > 0 and state[i, j] == state[i, j+1]:
                counts[2] += 1  # left
                counts[3] += 1  # right

    # Up merges
    for i in range(3):
        for j in range(4):
            if state[i, j] > 0 and state[i, j] == state[i+1, j]:
                counts[0] += 1  # up
                counts[1] += 1  # down

    return counts


# ---------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LOGIC ENGINE — Demonstration")
    print("=" * 60)

    # Create a sample board state
    test_state = np.array([
        [512, 128, 64, 32],
        [16,   8,  4,  2],
        [2,    4,  8,  0],
        [0,    0,  0,  0]
    ])

    engine = LogicEngine()

    # 1. Evaluate state
    print("\n--- Proposition Evaluation ---")
    result = engine.evaluate_state(test_state)
    for name, val in result['propositions'].items():
        print(f"  {name}: {val}")

    # 2. Fired rules (Modus Ponens)
    print(f"\n--- Modus Ponens Inference ({result['num_rules_fired']} rules fired) ---")
    for rule in result['fired_rules']:
        print(f"  [{rule['priority']}] {rule['rule_name']} → {rule['conclusion']}")
        for prem, val in rule['premises'].items():
            print(f"      {prem} = {val}")

    print(f"\n  Strategy: {result['strategy']}")

    # 3. Truth Table
    print("\n--- Truth Table: (A ∧ B) → C ---")
    tt = engine.get_demonstration_truth_table()
    print(f"  {tt['headers']}")
    for row in tt['rows']:
        print(f"  {row}")

    # 4. Resolution
    print("\n--- Resolution Demo ---")
    res = engine.get_resolution_demo()
    for step in res:
        print(f"  {step['clause_a']} ⊕ {step['clause_b']} = {step['resolvent']}")

    # 5. Action bias
    print("\n--- Action Bias ---")
    bias = engine.get_action_bias(test_state)
    dirs = ['Up', 'Down', 'Left', 'Right']
    for i, d in enumerate(dirs):
        print(f"  {d}: {bias[i]:+.2f}")
