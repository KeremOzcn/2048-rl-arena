"""
=====================================================
Optimization Algorithms — Hill Climbing, Simulated
Annealing, Genetic Algorithm
=====================================================
Optimization Specialist: Heuristic search techniques
Principles of AI — Applied Group Project
=====================================================
"""

import numpy as np
import random
import math
import copy
from mcts import GameSimulator


# ---------------------------------------------------------------
# 1. HILL CLIMBING
# ---------------------------------------------------------------

class HillClimbing:
    """
    Hill Climbing for 2048: Evaluate all 4 moves,
    pick the one with the best heuristic score.
    Greedy local search — always moves uphill.
    """

    def __init__(self, simulations_per_action=10, depth=3):
        self.simulations_per_action = simulations_per_action
        self.depth = depth
        self.last_stats = None

    def _heuristic(self, board):
        """Board quality heuristic combining multiple factors."""
        empty = float(np.sum(board == 0))
        max_tile = float(np.max(board))
        # Smoothness
        smooth = 0.0
        for i in range(4):
            for j in range(3):
                if board[i, j] > 0 and board[i, j + 1] > 0:
                    smooth -= abs(np.log2(board[i, j] + 1) - np.log2(board[i, j + 1] + 1))
                if board[j, i] > 0 and board[j + 1, i] > 0:
                    smooth -= abs(np.log2(board[j, i] + 1) - np.log2(board[j + 1, i] + 1))
        # Corner bonus
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        corner_bonus = 2.0 if max_tile in corners else 0.0
        # Monotonicity
        mono = 0.0
        for i in range(4):
            inc = dec = 0
            for j in range(3):
                if board[i, j] >= board[i, j + 1]:
                    dec += 1
                if board[i, j] <= board[i, j + 1]:
                    inc += 1
            mono += max(inc, dec)
        return empty * 2.7 + np.log2(max_tile + 1) * 1.0 + smooth * 0.1 + corner_bonus + mono * 0.5

    def search(self, board, score=0):
        """Evaluate each action and pick the best (hill climbing)."""
        action_scores = {}
        dirs = ['up', 'down', 'left', 'right']

        for action in range(4):
            total = 0.0
            valid = 0
            for _ in range(self.simulations_per_action):
                sim = GameSimulator(board, score)
                changed, sg = sim.move(action)
                if not changed:
                    continue
                # Look ahead
                h = self._heuristic(sim.board)
                for d in range(self.depth - 1):
                    if sim.game_over:
                        break
                    a = random.randint(0, 3)
                    sim.move(a)
                h_final = self._heuristic(sim.board)
                total += (h + h_final) / 2 + sg * 0.01
                valid += 1
            action_scores[action] = total / max(valid, 1)

        best = max(action_scores, key=action_scores.get)
        self.last_stats = {
            'action_scores': {dirs[a]: round(s, 2) for a, s in action_scores.items()},
            'best_action': dirs[best],
            'algorithm': 'Hill Climbing',
        }
        return best, self.last_stats


# ---------------------------------------------------------------
# 2. SIMULATED ANNEALING
# ---------------------------------------------------------------

class SimulatedAnnealing:
    """
    Simulated Annealing for 2048:
    Like Hill Climbing but sometimes accepts worse moves
    with probability e^(-ΔE/T), where T decreases over time.
    """

    def __init__(self, initial_temp=100.0, cooling_rate=0.92,
                 simulations=10, depth=3):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.simulations = simulations
        self.depth = depth
        self.temperature = initial_temp
        self.move_count = 0
        self.last_stats = None

    def _heuristic(self, board):
        empty = float(np.sum(board == 0))
        max_tile = float(np.max(board))
        smooth = 0.0
        for i in range(4):
            for j in range(3):
                if board[i, j] > 0 and board[i, j + 1] > 0:
                    smooth -= abs(np.log2(board[i, j] + 1) - np.log2(board[i, j + 1] + 1))
        corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
        corner = 2.0 if max_tile in corners else 0.0
        return empty * 2.7 + np.log2(max_tile + 1) + smooth * 0.1 + corner

    def _acceptance_probability(self, current_e, new_e, temp):
        """P(accept) = 1 if new > current, else e^((new-current)/T)"""
        if new_e >= current_e:
            return 1.0
        if temp <= 0.001:
            return 0.0
        return math.exp((new_e - current_e) / temp)

    def search(self, board, score=0):
        dirs = ['up', 'down', 'left', 'right']
        action_scores = {}
        current_h = self._heuristic(board)

        for action in range(4):
            total = 0.0
            valid = 0
            for _ in range(self.simulations):
                sim = GameSimulator(board, score)
                changed, sg = sim.move(action)
                if not changed:
                    continue
                new_h = self._heuristic(sim.board) + sg * 0.01
                ap = self._acceptance_probability(current_h, new_h, self.temperature)
                total += new_h * ap
                valid += 1
            action_scores[action] = total / max(valid, 1)

        best = max(action_scores, key=action_scores.get)

        # SA: with probability, accept a non-best action
        if self.temperature > 1.0 and random.random() < 0.15:
            candidates = [a for a in range(4) if action_scores[a] > 0]
            if len(candidates) > 1:
                alt = random.choice(candidates)
                delta = action_scores[best] - action_scores[alt]
                if delta > 0:
                    p = self._acceptance_probability(action_scores[best], action_scores[alt], self.temperature)
                    if random.random() < p:
                        best = alt

        self.temperature = max(0.01, self.temperature * self.cooling_rate)
        self.move_count += 1

        self.last_stats = {
            'action_scores': {dirs[a]: round(s, 2) for a, s in action_scores.items()},
            'best_action': dirs[best],
            'temperature': round(self.temperature, 3),
            'cooling_rate': self.cooling_rate,
            'acceptance_formula': 'e^(ΔE/T)',
            'algorithm': 'Simulated Annealing',
        }
        return best, self.last_stats

    def reset(self):
        self.temperature = self.initial_temp
        self.move_count = 0


# ---------------------------------------------------------------
# 3. GENETIC ALGORITHM
# ---------------------------------------------------------------

class GeneticAlgorithm:
    """
    Genetic Algorithm for 2048:
    Evolves a population of move sequences (chromosomes).
    Uses crossover and mutation to find the best strategy.
    """

    def __init__(self, population_size=20, chromosome_length=6,
                 generations=8, mutation_rate=0.2, crossover_rate=0.7):
        self.pop_size = population_size
        self.chrom_len = chromosome_length
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.last_stats = None

    def _random_chromosome(self):
        return [random.randint(0, 3) for _ in range(self.chrom_len)]

    def _fitness(self, chromosome, board, score):
        """Simulate the move sequence, return fitness score."""
        sim = GameSimulator(board, score)
        total_score = 0
        for action in chromosome:
            if sim.game_over:
                break
            changed, sg = sim.move(action)
            if changed:
                total_score += sg
            else:
                for alt in range(4):
                    changed, sg = sim.move(alt)
                    if changed:
                        total_score += sg
                        break
        empty = float(np.sum(sim.board == 0))
        max_t = float(np.max(sim.board))
        return total_score + empty * 10 + np.log2(max_t + 1) * 5

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        point = random.randint(1, self.chrom_len - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, chromosome):
        result = chromosome[:]
        for i in range(len(result)):
            if random.random() < self.mutation_rate:
                result[i] = random.randint(0, 3)
        return result

    def _tournament_select(self, population, fitnesses, k=3):
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def search(self, board, score=0):
        dirs = ['up', 'down', 'left', 'right']
        population = [self._random_chromosome() for _ in range(self.pop_size)]
        best_chromosome = None
        best_fitness = -float('inf')
        gen_stats = []

        for gen in range(self.generations):
            fitnesses = [self._fitness(c, board, score) for c in population]
            gen_best = max(fitnesses)
            gen_avg = sum(fitnesses) / len(fitnesses)
            gen_stats.append({'gen': gen + 1, 'best': round(gen_best, 1), 'avg': round(gen_avg, 1)})

            for i, f in enumerate(fitnesses):
                if f > best_fitness:
                    best_fitness = f
                    best_chromosome = population[i][:]

            # Selection + crossover + mutation
            new_pop = [best_chromosome[:]]  # Elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            population = new_pop

        best_action = best_chromosome[0] if best_chromosome else 0
        # Count first-move votes
        first_moves = [0, 0, 0, 0]
        final_fitnesses = [self._fitness(c, board, score) for c in population]
        for i, c in enumerate(population):
            first_moves[c[0]] += final_fitnesses[i]
        voted_action = max(range(4), key=lambda a: first_moves[a])

        self.last_stats = {
            'action_scores': {dirs[a]: round(first_moves[a], 1) for a in range(4)},
            'best_action': dirs[voted_action],
            'best_fitness': round(best_fitness, 1),
            'generations': self.generations,
            'population_size': self.pop_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'generation_stats': gen_stats,
            'algorithm': 'Genetic Algorithm',
        }
        return voted_action, self.last_stats
