"""Reasoning utilities and orchestrators.

Provides higher-level reasoning flows (chain-of-thought, step decomposition)
that orchestrate calls to the `ModelManager` for generating intermediate
thoughts and final conclusions.
"""

from __future__ import annotations

from typing import List
from models.model_manager import get_manager


def chain_of_thought(prompt: str, steps: int = 3) -> str:
    mgr = get_manager()
    # Very simple iterative reasoning: append step markers and ask the model
    thought = ""
    for i in range(steps):
        sub_prompt = f"Step {i+1}: Consider the following: {prompt}\nPrevious thoughts: {thought}\nWhat is the next reasoning step?"
        resp = mgr.generate_text(sub_prompt)
        thought += f"\nStep {i+1}: {resp}"
    return thought


def decompose_and_solve(input_text: str) -> str:
    mgr = get_manager()
    plan = mgr.generate_text(f"Decompose this problem into steps: {input_text}")
    # naive executor: ask model to solve following its plan
    solution = mgr.generate_text(f"Given plan:\n{plan}\nProvide a concise solution.")
    return f"Plan:\n{plan}\n\nSolution:\n{solution}"
