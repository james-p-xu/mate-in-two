"""
GRPO loop for Mate-in-1: binary 0/1 reward if the model's UCI move is legal and gives checkmate in one ply (half-move).
"""

import random
import itertools
import pandas as pd
import torch
import chess
import wandb
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

wandb.init(project="mate1-grpo", name="qwen2.5-3B", config={
    "model": "Qwen2.5-3B-Instruct", "algo": "GRPO", "batch_size": 64,
    "group_size": 8, "accumulate_steps": 4, "lr": 2e-5
})

# load game states
df = pd.read_csv(Path(__file__).resolve().parent.parent / "data/mate_in_1_fen_by_rating.csv")
fens = df["FEN"].tolist()

# models
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

# simple reward: legal + immediate checkmate
def reward_fn(fen, move_str):
    board = chess.Board(fen)
    try: board.push_uci(move_str)
    except ValueError: return 0.0
    return 1.0 if board.is_checkmate() else 0.0

# hyperparameters
cfg = GRPOConfig(
    model_name=MODEL_ID,
    learning_rate=2e-5,
    batch_size=64,
    group_size=8,
    lora_r=16,
    lora_alpha=32,
    max_new_tokens=4,
    pad_token_id=tokenizer.pad_token_id,
    log_with = "wandb"
)

trainer = GRPOTrainer(cfg, model=model, tokenizer=tokenizer)

# infinite loader, shuffle per epoch
def fen_loader():
    while True:
        random.shuffle(fens)
        yield from fens

fen_iter = fen_loader()

for step in range(10000):
    # build batch of prompts
    prompts = []
    for _ in range(cfg.batch_size):
        fen = next(fen_iter)
        prompts.append(f"<|user|>FEN: {fen}\nMate in 1. Your best move?<|assistant|>")

    # generate candidates
    gen = trainer.batched_generate(prompts)
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

    # score candidates
    rewards = []
    cycle = itertools.chain.from_iterable([[fen]*cfg.group_size for fen in prompts])
    for fen, text in zip(cycle, texts):
        move = text.strip().split()[0]
        rewards.append(reward_fn(fen, move))

    # policy update
    trainer.step(prompts, gen, rewards)

    # compute mean reward + logging
    mean_r = sum(rewards) / len(rewards)
    wandb.log({"mean_reward": mean_r, "step": step+1})
    if (step+1) % 500 == 0:
        trainer.save_pretrained(f"ckpts/mate1_grpo_step{step+1}")
