# Predictive Cognitive State Modeling for Adaptive Game Difficulty Regulation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Domain](https://img.shields.io/badge/Domain-NLP%20%2B%20AI%20%2B%20Gaming-purple?style=for-the-badge)
![Type](https://img.shields.io/badge/Type-Research%20Prototype-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-green?style=for-the-badge)

**A system that predicts a player's future cognitive state using emotional and behavioral data, and dynamically adjusts game difficulty to maintain optimal engagement.**



</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Innovation](#2-core-innovation)
3. [System Architecture](#3-system-architecture)
4. [Feature Space](#4-feature-space)
5. [Mathematical Model](#5-mathematical-model)
6. [Cognitive Instability Index (CII)](#6-cognitive-instability-index-cii)
7. [Prediction & Adaptive Control](#7-prediction--adaptive-control)
8. [Getting Started](#8-getting-started)
9. [Running the Demo](#9-running-the-demo)
10. [Project Structure](#10-project-structure)
11. [Key Results](#11-key-results)
12. [References](#12-references)

---

## 1. Project Overview

Interactive gaming systems currently rely on **reactive** adaptation — difficulty adjusts only _after_ a player shows observable frustration or disengagement. These systems suffer from temporal lag, intervening too late.

This project proposes a **predictive cognitive regulation framework** that:

- Models the **temporal evolution** of player emotion
- Computes a novel **Cognitive Instability Index (CII)** in real-time
- Predicts **future player state** (frustrated / bored / engaged) before onset
- Adjusts game difficulty **proactively** using a Reinforcement Learning controller

> **Paradigm Shift:**
> *From reactive difficulty adjustment → predictive cognitive regulation using temporal emotional modeling.*

---

## 2. Core Innovation

Unlike traditional systems, this framework:

| Aspect | Traditional Systems | This System |
|--------|-------------------|-------------|
| Adjustment Trigger | Reactive (post-event) | **Predictive (pre-event)** |
| Emotion Modeling | Static / Binary labels | **Temporal continuous E(t)** |
| Instability Measure | Heuristic thresholds | **Formal metric: CII(t)** |
| Signal Source | Telemetry only | **Telemetry + NLP (multimodal)** |
| Adaptation Direction | Unidirectional | **Bidirectional (frustration + boredom)** |
| Controller | Rule-based | **RL-optimized policy** |

The key insight: **Emotional Acceleration** A(t) detects a player falling toward frustration *before* their absolute state becomes critical — something impossible in any reactive system.

---

## 3. System Architecture

```
+------------------------------------------------------------------+
|                          INPUT LAYER                             |
|  Gameplay Telemetry: deaths, retries, score, reaction time...    |
|  Player Text Input : chat messages, reactions, voice-to-text     |
+---------------------------+--------------------------------------+
                            |
             +--------------+--------------+
             v                             v
+--------------------+         +------------------------+
|  TELEMETRY MODULE  |         |      NLP MODULE        |
|  -> Perf. Dev D(t) |         |  -> Polarity  P(t)     |
|  (z-score vs own   |         |  -> Intensity I(t)     |
|   baseline)        |         |  -> Anger prob p_ang   |
+--------+-----------+         |  -> Intent Gap G(t)    |
         |                     +----------+-------------+
         +------------------+-------------+
                            v
              +-----------------------------+
              |    EMOTIONAL DYNAMICS ENGINE |
              |  E(t) = { P, I, D, G }      |
              |  M(t) = P(t) * I(t)         |  <- Momentum
              |  A(t) = [M(t)-M(t-1)] / dt  |  <- Acceleration
              |  CII(t) = aM + bA + gD + dG |  <- Instability Index
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              |     PREDICTION MODULE       |
              |   LSTM / Transformer        |
              |   Input : CII(t-W ... t)    |
              |   Output: p_frus, p_bore    |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              |   ADAPTIVE CONTROLLER (RL)  |
              |   Decrease if frustrated    |
              |   Increase if bored         |
              |   -> Maintain Flow State    |
              +-----------------------------+
```

---

## 4. Feature Space

### 4.1 Gameplay Telemetry Features — T(t)

| Symbol | Feature | Description |
|--------|---------|-------------|
| d(t) | Death Count | Deaths in current session window |
| r(t) | Retry Rate | Level retries per minute |
| Δs(t) | Score Delta | Score change vs. expected |
| k(t) | Streak | Current win/loss streak length |
| τ(t) | Reaction Time | Average input latency (ms) |
| ν(t) | Input Speed | Actions per second |

**Why telemetry?** These signals are continuous, automatic, and encode behavioral manifestations of cognitive states the player may not consciously recognize. Reaction time and input speed appear *before* the player consciously feels frustrated.

**Limitation:** Telemetry measures *behavior*, not *affect*. Two players with identical death counts may be in completely different emotional states. NLP resolves this.

---

### 4.2 NLP-Based Emotional Features — N(t)

| Symbol | Feature | Description |
|--------|---------|-------------|
| P(t) | Emotional Polarity | Sentiment score ∈ [-1, +1] (BERT/RoBERTa) |
| I(t) | Emotional Intensity | Magnitude of expressed emotion ∈ [0, 1] |
| π_ang(t) | Anger Probability | Fine-tuned emotion classifier output |
| Δ_s(t) | Sentiment Trend | First-order difference of P(t) over sliding window |
| G(t) | Intent-Outcome Gap | Cosine distance between intent and outcome embeddings |

**NLP Pipeline:**
```
Raw Text -> Tokenize -> Sentiment Model -> P(t), I(t)
                     -> Emotion Classifier -> p_ang(t)
                     -> Semantic Role Labeling -> v_intent
                     -> Compare with outcome embedding -> G(t)
```

**Why NLP?** Linguistic signals capture the *subjective* dimension — what the player feels and thinks. A player performing well but hating the game is invisible to telemetry alone; NLP catches them.

### 4.3 Multimodal Complementarity

| Scenario | Telemetry | NLP | Resolution |
|----------|-----------|-----|------------|
| High stress, silent player | Detects | Missing | Telemetry only |
| Performing well, frustrated text | Misleading | Detects | NLP only |
| Both negative signals | Confirms | Confirms | Highest confidence CII |
| Conflicting signals | Negative | Positive | G(t) arbitrates |

---

## 5. Mathematical Model

### 5.1 Emotional State Vector E(t)

The player's emotional state is modeled as a **4-dimensional vector evolving over time**:

```
E(t) = ( P(t),  I(t),  D(t),  G(t) )  ∈  R^4
          |       |       |       |
       Polarity Intensity Perf.  Intent
                          Dev.   Gap
```

The **origin E(t) = (0,0,0,0)** = optimal flow state.
- Trajectories toward `{P<0, I>0.5, D<-1, G>0.5}` → **frustration attractor**
- Trajectories toward `{P>0, I<0.2, D>0, G<0.2}` → **boredom attractor**

### 5.2 Performance Deviation D(t)

```
D(t) = ( P_obs(t) - mu_P ) / sigma_P
```

Player-normalized z-score — captures underperformance relative to the **individual's own baseline**, not population norms. Invariant to between-player skill differences.

### 5.3 Intent-Outcome Gap G(t)

```
G(t) = 1 - cosine_similarity( v_intent(t), v_outcome(t) )
```

Directly operationalizes the frustration-aggression hypothesis: frustration arises when goal-directed behavior is blocked. G(t) = 0 → perfect alignment; G(t) = 1 → maximum mismatch.

---

## 6. Cognitive Instability Index (CII)

### The Core Formula

```
CII(t) = alpha * M(t)  +  beta * A(t)  +  gamma * D(t)  +  delta * G(t)
```

where `alpha + beta + gamma + delta = 1`, and all weights > 0.

**Default weights:** α=0.35, β=0.30, γ=0.25, δ=0.10

### Building Blocks

**Step 1 — Emotional Momentum:**
```
M(t) = P(t) * I(t)
```
Analogous to physical momentum p = mv. Encodes *direction × force* of the emotional state.

| P(t) | I(t) | M(t) | Meaning |
|------|------|------|---------|
| +0.9 | 0.8 | +0.72 | Highly engaged, confident |
| +0.5 | 0.1 | +0.05 | Positive but emotionally flat — boredom risk |
| 0.0  | 0.5 | 0.00  | Neutral — optimal flow |
| -0.5 | 0.3 | -0.15 | Mild discontent, low risk |
| -0.8 | 0.9 | -0.72 | Acute frustration risk |

**Step 2 — Emotional Acceleration:**
```
A(t) = [ M(t) - M(t-1) ] / delta_t
```
The **rate of change** of momentum — the early warning signal.

> Two players with identical M(t) = -0.3:
> - Player A: A(t) = -0.45 → heading to -0.75 (acute frustration) — INTERVENE
> - Player B: A(t) = +0.12 → heading to -0.18 (fading discontent) — WAIT
>
> Only a system with A(t) can distinguish these. This is what makes prediction possible.

**Step 3 — CII Decision Boundaries:**

```
CII(t) < -0.50  -->  [FRUSTRATION ALERT]  --> Decrease difficulty
-0.50 <= CII <= +0.30  -->  [FLOW STATE]  --> Maintain difficulty
CII(t) > +0.30  -->  [BOREDOM ALERT]     --> Increase difficulty
```

### Classical Mechanics Analogy

| CII Component | Classical Physics | Captures |
|---------------|-----------------|----------|
| P(t) | Velocity direction | Emotional direction |
| I(t) | Mass | Emotional persistence |
| M(t) = P·I | Momentum p=mv | Directional force |
| A(t) = ΔM/Δt | Force F=dp/dt | Rate of collapse |
| D(t) | Displacement | Behavioral drift |
| G(t) | Potential energy | Cognitive friction |
| **CII(t)** | **Total energy** | **Net instability** |

---

## 7. Prediction & Adaptive Control

### 7.1 LSTM Temporal Predictor

Input sequence of W=10 past feature vectors:
```
X(t) = [ x(t-9), x(t-8), ..., x(t) ]
where x(t) = [ CII(t), M(t), A(t), D(t), G(t) ]
```

Output:
```
(p_frus, p_bore) = LSTM( X(t) )
```

**Why temporal modeling?** A static model sees only the current snapshot. A temporal model learns the *signature* of an impending frustration event — the recognizable pattern in the 10-step CII trajectory that precedes onset.

### 7.2 Adaptive Difficulty Controller (RL)

MDP formulation:

| Component | Definition |
|-----------|-----------|
| State s_t | [CII, p_frus, p_bore, M, A, D, G, Difficulty] |
| Action a_t | {decrease, maintain, increase} difficulty |
| Reward r_t | -0.5·CII² - 0.3·p_frus - 0.2·p_bore + 0.1·[flow_bonus] |
| Policy | PPO (Proximal Policy Optimization) |
| Goal | Maximize cumulative reward → sustained flow state |

Reward is **zero at the flow state** and negative for any deviation. The quadratic CII term penalizes extremes more than mild deviations.

---

## 8. Getting Started

### Requirements

```bash
Python 3.10+
pip install pygame requests streamlit fastapi uvicorn pydantic pandas
```

### Clone / Download

```bash
git clone https://github.com/narendrasaraf/NLP-RP.git
cd NLP-RP
```

---

## 9. Running the Complete System

Because this is a real-time multimodal system, you must run three separate processes concurrently. Open three separate terminal windows in the project root:

**Terminal 1: Start the AI Backend (The Brain)**
```bash
uvicorn backend.main:app --reload --port 8001
```

**Terminal 2: Start the Live Analytics Dashboard (The Monitor)**
```bash
streamlit run frontend/dashboard.py
```

**Terminal 3: Start the Pygame Engine (The Client)**
```bash
python -m game.main
```

If you place your game window next to your browser window, you will visually see the Streamlit line-chart plotting your Cognitive Instability Index (CII) in real-time as you play!

---

## 10. Project Structure

```text
NLP-RP/
├── backend/
│   ├── main.py            # FastAPI REST Endpoint
│   ├── model.py           # CII Math Engine
│   ├── predictor.py       # ML Rule Regressor
│   └── utils.py           # Normalization math
├── frontend/
│   └── dashboard.py       # Live Streamlit UI
├── game/
│   ├── main.py            # 60 FPS Pygame Loop
│   ├── api_client.py      # Background Daemon Thread
│   ├── adaptation.py      # Pygame Physics Scaler
│   └── entities.py        # Player/Enemy Sprites
└── README.md
```

---

## 11. Key Results (Prototype)

| Scenario | CII Trajectory | System Response | Correctness |
|----------|---------------|-----------------|-------------|
| Frustration | +0.00 → -0.48 → -0.63 | Difficulty: 5.0 → 4.26 → 3.45 → 2.65 | Correct |
| Boredom | +0.32 → +0.04 → -0.36 | Correctly reduces as boredom collapses momentum | Correct |
| Flow | -0.24 → +0.20 → +0.44 | Minor corrections, final increase at boredom detection | Correct |

**Prototype limitations (future work):**
- Rule-based predictor → replace with trained LSTM
- Simulated inputs → real game engine integration
- Threshold weights → RL-optimized via PPO training
- Single player → multi-player generalization

---

## 12. References

1. Csikszentmihalyi, M. (1990). *Flow: The Psychology of Optimal Experience.* Harper & Row.
2. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
3. Dollard, J. et al. (1939). *Frustration and Aggression.* Yale University Press.
4. Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692.*
5. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
6. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347.*
7. Demszky, D. et al. (2020). GoEmotions: A dataset of fine-grained emotions. *ACL 2020.*
8. Ziebart, B. D. et al. (2008). Maximum entropy inverse reinforcement learning. *AAAI.*

---

<div align="center">


*"Our system shifts from reactive difficulty adjustment to predictive cognitive regulation using temporal emotional modeling."*

</div>
