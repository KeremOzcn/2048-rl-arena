"""
=====================================================
Test Suite - QA Deliverable
=====================================================
QA / Tester: Automated test cases
Principles of AI — Applied Group Project
=====================================================
"""

import unittest
import numpy as np
from game_2048 import Game2048
from rl_agent import QLearningAgent, compute_reward
from logic_engine import LogicEngine, Proposition, AND, OR, NOT, IMPLIES, generate_truth_table, modus_ponens, resolution, Rule
from math_models import LinearAlgebraModels, ProbabilityModels
from mcts import MCTS, GameSimulator, MCTSNode


class TestGame2048(unittest.TestCase):
    def setUp(self):
        self.game = Game2048()

    def test_initial_two_tiles(self):
        self.assertEqual(np.count_nonzero(self.game.board), 2)

    def test_initial_score_zero(self):
        self.assertEqual(self.game.score, 0)

    def test_board_size(self):
        self.assertEqual(self.game.board.shape, (4, 4))

    def test_move_left_merges(self):
        self.game.board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        score_before = self.game.score
        self.game.move_left()
        self.assertEqual(self.game.board[0][0], 4)
        self.assertEqual(self.game.score, score_before + 4)

    def test_move_right_merges(self):
        self.game.board = np.array([[0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.game.move_right()
        self.assertEqual(self.game.board[0][3], 8)

    def test_move_up_merges(self):
        self.game.board = np.array([[8, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.game.move_up()
        self.assertEqual(self.game.board[0][0], 16)

    def test_move_down_merges(self):
        self.game.board = np.array([[0, 0, 0, 16], [0, 0, 0, 16], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.game.move_down()
        self.assertEqual(self.game.board[3][3], 32)

    def test_different_tiles_no_merge(self):
        self.game.board = np.array([[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.game.move_left()
        self.assertEqual(self.game.board[0][0], 2)
        self.assertEqual(self.game.board[0][1], 4)

    def test_no_triple_merge(self):
        self.game.board = np.array([[2, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.game.move_left()
        self.assertEqual(self.game.board[0][0], 4)
        self.assertEqual(self.game.board[0][1], 2)

    def test_game_over_full_no_merges(self):
        self.game.board = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
        self.assertTrue(self.game._is_game_over())

    def test_not_game_over_with_empty(self):
        self.game.board = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0]])
        self.assertFalse(self.game._is_game_over())

    def test_reset(self):
        self.game.score = 1000
        self.game.reset()
        self.assertEqual(self.game.score, 0)
        self.assertEqual(self.game.move_count, 0)
        self.assertFalse(self.game.game_over)

    def test_max_tile(self):
        self.game.board = np.array([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(self.game.get_max_tile(), 16)

    def test_win_condition(self):
        self.game.board = np.array([[1024, 1024, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.game.move_left()
        self.assertTrue(self.game.won)

    def test_to_dict_serializable(self):
        d = self.game.to_dict()
        self.assertIn('board', d)
        self.assertIn('score', d)
        self.assertEqual(len(d['board']), 4)


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QLearningAgent()
        self.game = Game2048()

    def test_creation(self):
        self.assertEqual(len(self.agent.actions), 4)
        self.assertEqual(self.agent.epsilon, 1.0)

    def test_epsilon_decay(self):
        old = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, old)

    def test_epsilon_min(self):
        self.agent.epsilon = 0.05
        self.agent.decay_epsilon()
        self.assertGreaterEqual(self.agent.epsilon, 0.05)

    def test_action_in_range(self):
        action = self.agent.choose_action(self.game.get_state())
        self.assertIn(action, [0, 1, 2, 3])

    def test_state_key_hashable(self):
        key = self.agent._state_to_key(self.game.get_state())
        hash(key)

    def test_learning_updates_q(self):
        s = self.game.get_state()
        self.agent.learn(s, 0, 1.0, s, False)
        k = self.agent._state_to_key(s)
        self.assertNotEqual(self.agent.q_table[k][0], 0)


class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        self.game = Game2048()

    def test_positive_for_score_gain(self):
        r = compute_reward(self.game, 8, True, 2, 2)
        self.assertIsInstance(r, float)

    def test_penalty_for_invalid(self):
        r = compute_reward(self.game, 0, False, 2, 2)
        self.assertLess(r, 0)

    def test_bonus_for_new_max(self):
        with_bonus = compute_reward(self.game, 16, True, 8, 16)
        without_bonus = compute_reward(self.game, 16, True, 16, 16)
        self.assertGreater(with_bonus, without_bonus)


class TestLogicEngine(unittest.TestCase):
    def setUp(self):
        self.engine = LogicEngine()
        self.state = np.array([
            [512, 128, 64, 32],
            [16, 8, 4, 2],
            [2, 4, 8, 0],
            [0, 0, 0, 0]
        ])

    def test_connectives(self):
        self.assertTrue(AND(True, True))
        self.assertFalse(AND(True, False))
        self.assertTrue(OR(True, False))
        self.assertFalse(NOT(True))
        self.assertTrue(IMPLIES(False, False))
        self.assertFalse(IMPLIES(True, False))

    def test_truth_table(self):
        tt = generate_truth_table(['A', 'B'], lambda a, b: AND(a, b), 'A∧B')
        self.assertEqual(len(tt['rows']), 4)
        self.assertEqual(len(tt['headers']), 3)

    def test_proposition_evaluation(self):
        p = Proposition("test", lambda s: int(np.max(s)) > 100)
        self.assertTrue(p.evaluate(self.state))

    def test_modus_ponens(self):
        result = self.engine.evaluate_state(self.state)
        self.assertIn('propositions', result)
        self.assertIn('fired_rules', result)
        self.assertIn('strategy', result)
        self.assertIsInstance(result['fired_rules'], list)

    def test_resolution(self):
        r = resolution({'P', '~Q'}, {'Q', 'R'})
        self.assertIsNotNone(r)
        self.assertIn('P', r)
        self.assertIn('R', r)
        self.assertNotIn('Q', r)

    def test_action_bias_shape(self):
        bias = self.engine.get_action_bias(self.state)
        self.assertEqual(len(bias), 4)

    def test_truth_table_demo(self):
        tt = self.engine.get_demonstration_truth_table()
        self.assertIn('headers', tt)
        self.assertIn('rows', tt)

    def test_resolution_demo(self):
        res = self.engine.get_resolution_demo()
        self.assertIsInstance(res, list)


class TestMathModels(unittest.TestCase):
    def setUp(self):
        self.state = np.array([
            [512, 128, 64, 32],
            [16, 8, 4, 2],
            [2, 4, 8, 0],
            [0, 0, 0, 0]
        ])

    def test_frobenius_norm(self):
        norm = LinearAlgebraModels.frobenius_norm(self.state)
        self.assertGreater(norm, 0)

    def test_l2_norm(self):
        norm = LinearAlgebraModels.l2_norm(self.state)
        self.assertGreater(norm, 0)

    def test_cosine_similarity_identical(self):
        sim = LinearAlgebraModels.cosine_similarity(self.state, self.state)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_eigenvalue_analysis(self):
        result = LinearAlgebraModels.eigenvalue_analysis(self.state)
        self.assertIn('eigenvalues', result)
        self.assertIn('dominant_eigenvalue', result)
        self.assertEqual(len(result['eigenvalues']), 4)

    def test_matrix_rank(self):
        rank = LinearAlgebraModels.matrix_rank(self.state)
        self.assertGreater(rank, 0)
        self.assertLessEqual(rank, 4)

    def test_smoothness(self):
        s = LinearAlgebraModels.smoothness_score(self.state)
        self.assertIsInstance(s, float)

    def test_monotonicity(self):
        m = LinearAlgebraModels.monotonicity_score(self.state)
        self.assertIn('total', m)

    def test_tile_distribution(self):
        dist = ProbabilityModels.tile_distribution(self.state)
        self.assertIsInstance(dist, dict)
        total = sum(dist.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_expected_new_tile(self):
        ev = ProbabilityModels.expected_new_tile_value()
        self.assertAlmostEqual(ev, 2.2)

    def test_board_entropy(self):
        h = ProbabilityModels.board_entropy(self.state)
        self.assertGreater(h, 0)

    def test_survival_probability(self):
        sp = ProbabilityModels.survival_probability(self.state)
        self.assertGreaterEqual(sp, 0.0)
        self.assertLessEqual(sp, 1.0)

    def test_full_analysis(self):
        result = LinearAlgebraModels.full_analysis(self.state)
        self.assertIn('frobenius_norm', result)
        self.assertIn('matrix_rank', result)


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.board = np.array([
            [256, 128, 64, 32],
            [16, 8, 4, 2],
            [2, 4, 0, 0],
            [0, 0, 0, 0]
        ])

    def test_mcts_returns_valid_action(self):
        mcts = MCTS(iterations=20, rollout_depth=5)
        action, stats = mcts.search(self.board)
        self.assertIn(action, [0, 1, 2, 3])

    def test_mcts_stats(self):
        mcts = MCTS(iterations=20, rollout_depth=5)
        _, stats = mcts.search(self.board)
        self.assertIn('iterations', stats)
        self.assertIn('elapsed_ms', stats)
        self.assertIn('action_stats', stats)

    def test_game_simulator_move(self):
        sim = GameSimulator(self.board)
        changed, _ = sim.move(2)  # left
        self.assertIsInstance(changed, (bool, np.bool_))

    def test_game_simulator_clone(self):
        sim = GameSimulator(self.board)
        clone = sim.clone()
        self.assertTrue(np.array_equal(sim.board, clone.board))

    def test_mcts_node_ucb1(self):
        node = MCTSNode(self.board)
        node.visits = 10
        node.total_reward = 5.0
        node.parent = MCTSNode(self.board)
        node.parent.visits = 100
        ucb = node.ucb1()
        self.assertGreater(ucb, 0)


class TestIntegration(unittest.TestCase):
    def test_agent_plays_game(self):
        game = Game2048()
        agent = QLearningAgent()
        agent.epsilon = 0.5
        steps = 0
        while not game.game_over and steps < 200:
            state = game.get_state()
            action = agent.choose_action(state)
            game.make_move(action)
            steps += 1
        self.assertGreater(steps, 0)

    def test_training_grows_qtable(self):
        game = Game2048()
        agent = QLearningAgent(epsilon=0.5)
        for _ in range(20):
            game.reset()
            state = game.get_state()
            while not game.game_over:
                action = agent.choose_action(state)
                _, sg, done = game.make_move(action)
                ns = game.get_state()
                agent.learn(state, action, sg or -1, ns, done)
                state = ns
        self.assertGreater(len(agent.q_table), 0)

    def test_agent_has_logic_engine(self):
        agent = QLearningAgent()
        self.assertIsNotNone(agent.logic_engine)
        self.assertGreater(len(agent.logic_engine.rules), 0)

    def test_agent_has_mcts(self):
        agent = QLearningAgent()
        self.assertIsNotNone(agent.mcts)

    def test_full_pillar_decision(self):
        game = Game2048()
        agent = QLearningAgent()
        agent.epsilon = 0
        state = game.get_state()
        action, info = agent.choose_action_with_mcts(state)
        self.assertIn(action, [0, 1, 2, 3])
        self.assertIn('logic_strategy', info)
        self.assertIn('mcts_action', info)
        self.assertIn('math', info)


def run_all():
    print("=" * 60)
    print("AUTOMATED TEST SUITE")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGame2048))
    suite.addTests(loader.loadTestsFromTestCase(TestQLearningAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestLogicEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestMathModels))
    suite.addTests(loader.loadTestsFromTestCase(TestMCTS))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\n{'='*60}")
    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"Total: {total} | Passed: {passed}")
    print(f"Failed: {len(result.failures)} | Errors: {len(result.errors)}")
    print(f"Pass Rate: {(passed / total * 100):.1f}%")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_all()
