# Mate-in-Two (Qwen + RL Chess Puzzle Solver)
- Inspired by [TinyZero](https://github.com/Jiayi-Pan/TinyZero/) (countdown task)

## Instructions
- Setup Stockfish
    - [Download link](https://stockfishchess.org/download/)
    - On Mac, allow app in Systems Settings -> Privacy & Security
    - Set `STOCKFISH_PATH` environment variable to the Stockfish executable

## Ablations
- From TinyZero
    - [Twitter thread](https://x.com/jiayi_pirate/status/1882839370505621655)
    - Model size, base vs. instruct, RL algorithm
- Effect of curriculum training
    - Increasing elo vs. random ordering
- Board representation
    - FEN vs. piece-list vs. ascii
    - Qwen VL/2.5-VL + image of board (?)
- Reward shaping
    - Binary 0/1 vs. Stockfish score delta (centipawns)