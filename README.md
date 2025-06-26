# Mate-in-Two (Qwen + RL Chess Puzzle Solver)
- Inspired by [TinyZero](https://github.com/Jiayi-Pan/TinyZero/) (countdown task)

## Ablations
- From TinyZero
    - Model size, base vs. instruct, RL algorithm
- Effect of curriculum training
    - Increasing elo vs. random ordering
- Board representation
    - FEN vs. piece-list vs. ascii
    - Qwen VL/2.5-VL + image of board (?)
- Reward shaping
    - Binary 0/1 vs. Stockfish score delta (centipawns)