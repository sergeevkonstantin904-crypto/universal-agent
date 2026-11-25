
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
UNIVERSAL RL AGENT CORE (v1.0)
Абстрактный фреймворк для обучения агента. 
Применим для: управления ресурсами, рекомендательных систем, логистики, управления производством.
"""

import os
import sys
import asyncio
import json
import math
import datetime
import signal
import logging
import hashlib
import time
import secrets
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import List, Tuple, Dict, Any, AsyncIterator, Deque
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import aiosqlite
import aiofiles
from pydantic import BaseModel, Field, ValidationError


import uvloop
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiohttp import web, ClientSession, ClientTimeout
import aiohttp
from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry
from dotenv import load_dotenv


@dataclass
class Config:
    SEED: int = 42
    DEVICE: str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# RL Params
GAMMA: float = 0.99
LAMBDA: float = 0.95
CLIP_EPS: float = 0.2
LEARNING_RATE: float = 3e-4
GRAD_CLIP: float = 0.5

# Risk/Constraint Management
MAX_CUMULATIVE_REWARD_DRAWDOWN: float = 0.20
MAX_ACTION_INTENSITY: float = 0.05  # Макс. 5% от доступной емкости за шаг
K_FRACTION: float = 0.2

# System
BACKUP_INTERVAL: int = 300
BATCH_SIZE: int = 2048
PPO_EPOCHS: int = 4
BATCH_SIZE_PPO: int = 64

# AI Models
EMBEDDING_DIM: int = 384
GROUNDING_DIM: int = 32
INPUT_DIM: int = 7 + GROUNDING_DIM # 7 base features + 32 grounding

CONFIG_HASH: str = field(init=False, repr=False)

def __post_init__(self):
    config_dict = {k: v for k, v in asdict(self).items() if k != 'CONFIG_HASH'}
    self.CONFIG_HASH = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

    NUM_ACTIONS: int = 3
    CAPACITY_SCALE: float = 1000.0
    REWARD_SCALE: float = 10000.0

CONFIG = Config()


torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
G_GENERATOR = torch.Generator(device=CONFIG.DEVICE).manual_seed(CONFIG.SEED)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ContextData(BaseModel):
    features: Dict[str, float]
    action_capacity: float = Field(gt=0)
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class ActionResult:
    status: str 
    action_id: int 
    action_value: float 
    action_intensity: float
    outcome: int 
    net_reward: float
    logp_intensity: torch.Tensor
    logp_action: torch.Tensor
    value: torch.Tensor
    reward: float


def calculate_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float, device: str) -> torch.Tensor:
    advantages = torch.zeros_like(rewards, device=device)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        next_values = values[t + 1] if t < len(rewards) - 1 else torch.tensor(0.0, device=device)
        delta = rewards[t] + gamma * next_values - values[t]
        last_gae_lam = delta + gamma * lam * last_gae_lam
        advantages[t] = last_gae_lam
    return advantages


class ActorCritic(nn.Module):
    def __init__(self, input_dim=CONFIG.INPUT_DIM, num_actions=CONFIG.NUM_ACTIONS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU()
        )
        # Непрерывное действие (насколько сильно действовать)
        self.intensity_alpha = nn.Linear(64, 1)
        self.intensity_beta = nn.Linear(64, 1)
        # Дискретное действие (какое действие выбрать)
        self.action_logits = nn.Linear(64, num_actions) 
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        hidden = self.shared(x)
        alpha = torch.exp(self.intensity_alpha(hidden)) + 1.0
        beta = torch.exp(self.intensity_beta(hidden)) + 1.0
        return (alpha, beta, self.action_logits(hidden)), self.value(hidden)

class UniversalAgent:
    def __init__(self,
                 transition_fn=None,
                 reward_fn=None,
                 risk_fn=None,
                 state_encoder_fn=None,
                 num_actions: int = CONFIG.NUM_ACTIONS):
        self.cumulative_reward = 0.0
        self.peak_reward = 0.0
        self.episode_buffer = []
        self.device = CONFIG.DEVICE
        self.num_actions = num_actions
        # Внешние функции предметной области — агент не содержит доменной логики
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.risk_fn = risk_fn
        self.state_encoder_fn = state_encoder_fn

        self.agent = ActorCritic(input_dim=CONFIG.INPUT_DIM, num_actions=num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=CONFIG.LEARNING_RATE)

        # Для RAG (опционально)
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base').to(self.device)
        self.embedding_model.eval()
        self.grounding_adapter = nn.Linear(CONFIG.EMBEDDING_DIM, CONFIG.GROUNDING_DIM).to(self.device)

    def _risk_ok(self):
        if self.risk_fn is not None:
            return self.risk_fn({
                'cumulative_reward': self.cumulative_reward,
                'peak_reward': self.peak_reward,
                'episode_buffer_len': len(self.episode_buffer),
            })
        return True

    async def step(self, context: ContextData, generator) -> ActionResult:
        if not self._risk_ok(): 
            return ActionResult(
                "HALT", -1, 0, 0, 0, 0, 
                torch.tensor([0.0], device=self.device), 
                torch.tensor([0.0], device=self.device), 
                torch.tensor([0.0], device=self.device), 
                0.0
            )
        
        start = time.perf_counter()
        
        # --- State Vector Construction ---
        base_features = list(context.features.values())
        base_features.append(context.action_capacity / CONFIG.CAPACITY_SCALE) 
        base_features.append(self.cumulative_reward / CONFIG.REWARD_SCALE)
        
        base_vec = torch.FloatTensor(base_features).unsqueeze(0).to(self.device)
        state_vec = base_vec 
        
        (alpha, beta, action_logits), value = self.agent(state_vec)
        
        # --- Policy Sampling ---
        intensity_dist = Beta(alpha.squeeze(), beta.squeeze())
        raw_intensity = intensity_dist.sample(generator=generator)
        action_intensity = raw_intensity.clamp(1e-6, CONFIG.MAX_ACTION_INTENSITY).item()
        
        action_dist = Categorical(logits=action_logits.squeeze())
        action_id = action_dist.sample(generator=generator).item()
        
        # --- Environment Transition (через внешний переход) ---
        outcome = await self._environment_step(context, action_id, action_intensity)
        
        # --- Reward (через внешнюю функцию; по умолчанию — исход как награда) ---
        if self.reward_fn is not None:
            net_reward = self.reward_fn(context, action_id, action_intensity, outcome)
        else:
            net_reward = float(outcome)
        
        self.cumulative_reward += net_reward
        if self.cumulative_reward > self.peak_reward: 
            self.peak_reward = self.cumulative_reward
        
        reward_tensor = torch.tensor(net_reward, device=self.device)
        
        step_id = hashlib.sha256(f"{context.timestamp}|{action_id}|{action_intensity}".encode()).hexdigest()
        
        logp_intensity = intensity_dist.log_prob(raw_intensity).detach()
        logp_action = action_dist.log_prob(torch.tensor([action_id], device=self.device)).detach()
        
        result = ActionResult(
            status="OK", 
            action_id=action_id, 
            action_value=action_intensity,
            action_intensity=action_intensity, 
            outcome=outcome, 
            net_reward=net_reward,
            logp_intensity=logp_intensity, 
            logp_action=logp_action, 
            value=value.squeeze(),
            reward=reward_tensor
        )
        
        self.episode_buffer.append({
            'state': state_vec.squeeze(0).detach(), 
            'action_intensity': torch.tensor([action_intensity], device=self.device), 
            'action_id': torch.tensor([action_id], device=self.device),
            'logp_intensity': logp_intensity, 
            'logp_action': logp_action, 
            'value': value.squeeze().detach(),
            'reward': reward_tensor, 
            'done': True
        })
        
        if len(self.episode_buffer) >= CONFIG.BATCH_SIZE: 
            await self.update()
        
        return result

    async def _environment_step(self, context: ContextData, action_id: int, intensity: float) -> int:
        """
        Переход среды (Environment Step). Замените на свою логику предметной области.
        Контракт: принимает (context, action_id, intensity) -> int-метка исхода.
        По умолчанию — вызов внешней transition_fn; если её нет, метод не выполняется.
        """
        if self.transition_fn is not None:
            return self.transition_fn(context, action_id, intensity)
        raise NotImplementedError(
            "transition_fn не задана: передайте свою функцию перехода среды в UniversalAgent(transition_fn=...)"
        )


    async def update(self):
        if len(self.episode_buffer) < 2: 
            return
        
        states = torch.stack([b['state'] for b in self.episode_buffer])
        actions_intensity = torch.stack([b['action_intensity'] for b in self.episode_buffer])
        actions_id = torch.stack([b['action_id'] for b in self.episode_buffer])
        old_values = torch.stack([b['value'] for b in self.episode_buffer])
        rewards = torch.stack([b['reward'] for b in self.episode_buffer])
        old_logp_intensity = torch.stack([b['logp_intensity'] for b in self.episode_buffer])
        old_logp_action = torch.stack([b['logp_action'] for b in self.episode_buffer])
        
        advantages = calculate_gae(rewards, old_values, CONFIG.GAMMA, CONFIG.LAMBDA, self.device)
        # Нормализация преимуществ для стабильности обучения
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + old_values
        
        for epoch in range(CONFIG.PPO_EPOCHS):
            idx = torch.randperm(len(states))
            for i in range(0, len(states), CONFIG.BATCH_SIZE_PPO):
                batch = idx[i:i+CONFIG.BATCH_SIZE_PPO]
                (new_alpha, new_beta, new_logits), new_value = self.agent(states[batch])
                
                new_dist_intensity = Beta(new_alpha.squeeze(), new_beta.squeeze())
                new_logp_intensity = new_dist_intensity.log_prob(actions_intensity[batch])
                ratio_intensity = torch.exp(new_logp_intensity - old_logp_intensity[batch])
                
                new_dist_action = Categorical(logits=new_logits.squeeze())
                new_logp_action = new_dist_action.log_prob(actions_id[batch])
                ratio_action = torch.exp(new_logp_action - old_logp_action[batch])
                
                surr1_i = ratio_intensity * advantages[batch]
                surr2_i = torch.clamp(ratio_intensity, 1-CONFIG.CLIP_EPS, 1+CONFIG.CLIP_EPS) * advantages[batch]
                surr1_a = ratio_action * advantages[batch]
                surr2_a = torch.clamp(ratio_action, 1-CONFIG.CLIP_EPS, 1+CONFIG.CLIP_EPS) * advantages[batch]
                
                value_loss = 0.5 * F.mse_loss(new_value.squeeze(), returns[batch])
                entropy_bonus = (new_dist_intensity.entropy().mean() + new_dist_action.entropy().mean()) * 0.01
                
                loss = -torch.min(surr1_i, surr2_i).mean() - torch.min(surr1_a, surr2_a).mean() + value_loss - entropy_bonus
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), CONFIG.GRAD_CLIP)
                self.optimizer.step()
                
        self.episode_buffer.clear()

    async def shutdown(self):
        logging.info("🛑 Shutdown: Saving model...")
        torch.save({'agent': self.agent.state_dict(), 'reward': self.cumulative_reward}, "universal_agent_final.pt")

async def mock_context_stream() -> AsyncIterator[ContextData]:
    for i in range(500):
        yield ContextData(
            features={"f1": secrets.randbelow(100)/100, "f2": np.random.randn(), "f3": np.random.uniform(0, 1)},
            action_capacity=1000.0 + secrets.randbelow(5000),
            metadata={"step": i}
        )
        await asyncio.sleep(0.01)


async def main():
    agent = UniversalAgent()
    logging.info("🚀 Starting Universal Agent (Mode: Training)")

    try:
        async for context in mock_context_stream():
            result = await agent.step(context, G_GENERATOR)
            
            if result.status == "OK":
                logging.info(f"Action: {result.action_id} | Intensity: {result.action_intensity:.4f} | Reward: {result.net_reward:.2f} | Total: {agent.cumulative_reward:.2f}")
            elif result.status == "HALT":
                logging.critical("System halted by Risk Engine.")
                break
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
