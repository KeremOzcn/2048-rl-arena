"""
=====================================================
Game Engine - 2048 Game Logic
=====================================================
Lead Developer: Game mechanics implementation
Software Project Management & Technical Monitoring
=====================================================
"""

import numpy as np
import random


class Game2048:
    """
    2048 Oyun Engine
    - 4x4 grid
    - Tile'lar yukarı/aşağı/sol/sağ kaydırılır
    - Aynı sayılı tile'lar birleşir
    - Hedef: 2048 tile'a ulaşmak
    """

    SIZE = 4

    def __init__(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.score = 0
        self.move_count = 0
        self.game_over = False
        self.won = False
        self._add_random_tile()
        self._add_random_tile()

    def _add_random_tile(self):
        """Boş hücreye yeni tile ekler (%90 ihtimalle 2, %10 ihtimalle 4)"""
        empty_cells = [(i, j) for i in range(self.SIZE)
                       for j in range(self.SIZE) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def _compress_row(self, row):
        """Sıfır olmayanları sola toplar"""
        new_row = [v for v in row if v != 0]
        new_row += [0] * (self.SIZE - len(new_row))
        return new_row

    def _merge_row(self, row):
        """Yan yana aynı tile'ları birleştirir"""
        for i in range(self.SIZE - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                self.score += row[i]
                row[i + 1] = 0
                if row[i] == 2048:
                    self.won = True
        return row

    def move_left(self):
        changed = False
        for i in range(self.SIZE):
            original = self.board[i].copy()
            row = self._compress_row(self.board[i].tolist())
            row = self._merge_row(row)
            row = self._compress_row(row)
            self.board[i] = row
            if not np.array_equal(original, self.board[i]):
                changed = True
        return changed

    def move_right(self):
        self.board = np.fliplr(self.board)
        changed = self.move_left()
        self.board = np.fliplr(self.board)
        return changed

    def move_up(self):
        self.board = self.board.T
        changed = self.move_left()
        self.board = self.board.T
        return changed

    def move_down(self):
        self.board = self.board.T
        self.board = np.fliplr(self.board)
        changed = self.move_left()
        self.board = np.fliplr(self.board)
        self.board = self.board.T
        return changed

    def make_move(self, action):
        """
        action: 0=up, 1=down, 2=left, 3=right
        Returns: (changed, score_gained, game_over)
        """
        score_before = self.score
        moves = {0: self.move_up, 1: self.move_down,
                 2: self.move_left, 3: self.move_right}

        changed = moves[action]()
        score_gained = self.score - score_before

        if changed:
            self._add_random_tile()
            self.move_count += 1

        if self._is_game_over():
            self.game_over = True

        return changed, score_gained, self.game_over

    def make_move_by_name(self, direction):
        """Direction string ile hareket yapar"""
        mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        return self.make_move(mapping[direction])

    def _is_game_over(self):
        """Boş hücre yok ve birleştirilebilecek tile yoksa oyun biter"""
        if np.any(self.board == 0):
            return False
        for i in range(self.SIZE):
            for j in range(self.SIZE - 1):
                if self.board[i][j] == self.board[i][j + 1]:
                    return False
                if self.board[j][i] == self.board[j + 1][i]:
                    return False
        return True

    def get_state(self):
        return self.board.copy()

    def get_max_tile(self):
        return int(np.max(self.board))

    def get_empty_count(self):
        return int(np.sum(self.board == 0))

    def reset(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.score = 0
        self.move_count = 0
        self.game_over = False
        self.won = False
        self._add_random_tile()
        self._add_random_tile()
        return self.get_state()

    def to_dict(self):
        """Web API için JSON-serializable dict"""
        return {
            'board': self.board.tolist(),
            'score': int(self.score),
            'move_count': int(self.move_count),
            'max_tile': self.get_max_tile(),
            'empty_count': self.get_empty_count(),
            'game_over': bool(self.game_over),
            'won': bool(self.won),
        }


if __name__ == "__main__":
    game = Game2048()
    print("Initial board:")
    print(game.board)
    print(f"Score: {game.score}")
