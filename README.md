===== НАЧАЛО ФАЙЛА =====
# Universal RL Agent Core (v1.0)

Абстрактный PPO-агент на PyTorch. Домен подключается снаружи через 4 функции —
внутри класса нет ни одной строки, привязанной к предметной области.
Один и тот же файл обучает логистику, финтех, рекомендации и производство
без правки кода агента.

## Что внутри

- `ActorCritic` — общая сеть (shared trunk + Beta-распределение для интенсивности,
  Categorical для дискретного действия, value head). Число действий — параметр
  `num_actions`, не хардкод.
- `UniversalAgent.step` — сбор состояния, сэмплирование политики, вызов внешней
  `transition_fn`, внешняя `reward_fn`, запись в `episode_buffer`.
- `UniversalAgent.update` — PPO-обновление: GAE, нормализация преимуществ,
  энтропийный бонус, gradient clipping.
- `calculate_gae` — отдельная функция, без привязки к домену.
- Риск-проверка — внешняя `risk_fn` (по умолчанию `True`). Никакого drawdown
  внутри агента: это была доменная логика, вынесена наружу.

## 4 интерфейса предметной области

| Функция | Сигнатура | Что возвращает | По умолчанию |
|---|---|---|---|
| `transition_fn` | `(ContextData, action_id, intensity) -> int` | метка исхода среды | `NotImplementedError` |
| `reward_fn` | `(ContextData, action_id, intensity, outcome) -> float` | награда | `float(outcome)` |
| `risk_fn` | `(dict состояния агента) -> bool` | продолжать или HALT | `True` |
| `state_encoder_fn` | `(ContextData, agent) -> Tensor` | вектор состояния | встроенный base-features |

Агент не знает, что именно он делает. Формула награды, переход среды,
риск-критерий и кодирование состояния — снаружи.

## Запуск

```bash
pip install -r requirements.txt
python universal_agent.py
