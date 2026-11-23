# $\color{Red}{\textsf{GateMind: Deep Reinforcement Learning for Airport Gate Assignment}}$

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-custom_env-0081A5?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![RL](https://img.shields.io/badge/Topic-Deep_Reinforcement_Learning-FF6F00?style=flat)

</div>



### Table of Contents

1. [Overview](#color1f6febtextsfoverview)
2. [The Problem](#color1f6febtextsfthe-problem)
3. [The Environment](#color1f6febtextsfthe-environment)
4. [MDP Formulation](#color1f6febtextsfmdp-formulation)
5. [Reward Function](#color1f6febtextsfreward-function)
6. [The Agents](#color1f6febtextsfthe-agents)
7. [Evaluation](#color1f6febtextsfevaluation)
8. [Results](#color1f6febtextsfresults)
9. [Roadmap](#color1f6febtextsfroadmap)
10. [References](#color1f6febtextsfreferences)

<br>


## $\color{#1F6FEB}{\textsf{The Motivation}}$

The **Airport Gate Assignment Problem (AGAP)** consists of **assigning each flight to a gate** while respecting operational constraints and **maximizing both passenger convenience** and **airport efficiency**. The problem has been studied since the 1970s, with traditional approaches mainly relying on **mathematical optimization**.


<div align="center">

<img src="./assets/agap.png" alt="Amsterdam Schiphol terminal map" width="80%">

</div>


### $\color{#E36209}{\textsf{Why Reinforcement Learning?}}$

A natural approach is a **greedy rule**: give each arriving flight the closest free gate that fits. This rule is short-sighted:

- A narrow-body aircraft is parked at a wide-body gate because it is the closest one available.
- Two hours later, a wide-body aircraft arrives and no compatible gate is free, so it goes to a remote stand.

A good assignment depends on **the flights that have not arrived yet**. This is a sequential decision problem, where today's choice changes tomorrow's options, which is exactly what RL is designed for.

<br>

## $\color{#1F6FEB}{\textsf{GateMind : Overview}}$

**GateMind** is a `Deep Reinforcement Learning` project that develops an agent able to **assign airport gates to arriving flights automatically**, using a simulated airport and synthetic flight schedules.

- Each time a flight arrives, the agent chooses a gate for it, or sends it to a remote stand when no suitable gate is available.
- Every assignment must respect the operational constraints: **aircraft size and gate compatibility**, **gate availability**, and **operational restrictions** of each pier.
- Among the valid choices, the agent learns to pick the most **efficient** one: avoiding delays and remote stands, limiting walking distance, and not wasting large gates on small aircraft.
- The project evaluates a **Deep Q-Network (DQN)** and two of its variants, against **random** and **greedy rule-based** agents used as reference points.

### $\color{#E36209}{\textsf{Research question}}$

> Can a DQN agent learn to assign gates to flights in a way that respects all operational constraints while making efficient use of the airport's resources?


<br>


## $\color{#1F6FEB}{\textsf{The Environment}}$

The environment is a **simplified airport inspired by [Amsterdam Schiphol Airport (AMS)](https://www.oneworld.com/airport-maps/ams)**. It keeps Schiphol's structure (several piers with different roles, contact gates and remote stands, size limits per pier) but uses a reduced number of gates and synthetic distances. It is not an exact model of the real airport.

<div align="center">

<img src="./assets/gates.png" alt="Amsterdam Schiphol terminal map" width="70%">

</div>

### $\color{#E36209}{\textsf{Gates}}$

<div align="center">

| Gate | Max aircraft size | Role | Distance to terminal | Gates in simulation |
| ---  | --- | --- | --- | ---: |
| B    | Narrow-body | Regional and short-haul flights, fast turnaround | Long | 4 |
| C    | Narrow-body | Short-haul flights | Medium | 4 |
| D    | Wide-body | Most flexible group, all aircraft types | Short | 6 |
| E    | Wide-body | Long-haul flights | Medium | 4 |
| F    | Wide-body | Long-haul international flights | Medium | 3 |
| G    | Wide-body | Long-haul flights, limited availability | Long | 2 |
| Remote stands | Any | Used when no gate is suitable, bus transfer needed | Longest | 1 |


</div>



### $\color{#E36209}{\textsf{Aircraft Categories}}$

<div align="center">

| Category | Example aircraft | Turnaround |
| --- | --- | --- |
| Regional | Embraer E190 | Short |
| Narrow-body | Airbus A320, Boeing 737 | Medium |
| Wide-body | Boeing 777, Airbus A350 | Long |

</div>

### $\color{#E36209}{\textsf{Flight Schedule Generator}}$

$\color{Red}{\textsf{No real schedule data was used}}$. Each episode is a **simulated day of flights** produced by a schedule generator:

- Arrival times follow daily patterns, with **morning and evening peaks**.
- Each flight gets an aircraft category, drawn from a realistic fleet mix.
- The turnaround time depends on the category, with some random variation.
- Each day is generated from a random seed, so the agent sees a different day every episode and must **generalize** instead of memorizing one schedule.

<br>


## $\color{#1F6FEB}{\textsf{MDP Formulation}}$

The problem is modeled as a **Markov Decision Process (MDP)**. Each decision step corresponds to **one arriving flight**: flights are processed in order of arrival, and the agent assigns them one by one.

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \mathcal{T}, \gamma)$$

<br>

```mermaid
flowchart LR
    A[DQN Agent] -- "action: gate or remote stand" --> E[Airport Environment]
    E -- "state: next flight + gate availability" --> A
    E -- "reward" --> A
```

<br>

<div align="center">

| Symbol | Element | Definition |
| --- | --- | --- |
| $\mathcal{S}$ | State space | Features of the incoming flight + for each gate: time until it becomes free and whether it fits the aircraft |
| $\mathcal{A}$ | Action space | $\lbrace \text{gate}_1, \dots, \text{gate}_N, \text{remote} \rbrace$: $N + 1$ discrete actions |
| $P(s' \mid s, a)$ | Transition function | Deterministic. The chosen gate becomes occupied until the flight departs, then the next flight arrives |
| $R(s, a, s')$ | Reward function | Defined in the next section |
| $\mathcal{T}$ | Terminal state | All flights of the day have been assigned |
| $\gamma$ | Discount factor | $\gamma = 0.99$ |

</div>

### $\color{#E36209}{\textsf{State}}$

The state is a vector given to the neural network:

- **Incoming flight:** aircraft category, arrival time in the day, expected turnaround time.
- **For each gate:** time remaining until the gate is free (0 if free), and a compatibility flag (1 if the aircraft fits the gate).

Using the *time until free* instead of a simple free/occupied flag keeps the state **Markov**: the agent knows everything needed to predict which gates will be available for the next flights.

### $\color{#E36209}{\textsf{Action Masking}}$

Some actions are always invalid: a gate that is occupied, or too small for the aircraft. Instead of letting the agent try them and punishing it, these actions are **masked**: their Q-values are set to $-\infty$, so the agent can never select them.

$$\mathcal{A}_{\text{valid}}(s) = \lbrace g : g \text{ is free at arrival and } \text{size}(f) \le \text{size}(g) \rbrace \cup \lbrace \text{remote} \rbrace$$

The remote stand is always valid, so the agent always has at least one possible action. As a result, **gate conflicts and size violations are impossible by design**, and the agent only learns *which valid gate is the best choice*.

<br>

## $\color{#1F6FEB}{\textsf{Reward Function}}$

The reward follows the two main goals of gate assignment: **passenger convenience** (board at the terminal, walk less) and **efficient use of gates** (keep large gates for large aircraft). Each goal becomes one term of the reward. Hard constraints are not in the reward, since the action mask already blocks them.

<div align="center">

| Event | Effect | Purpose |
| --- | --- | --- |
| Flight assigned to a contact gate | Positive | Keep flights at the terminal |
| Walking distance of the gate | Negative | Prefer gates close to the terminal |
| Small aircraft at a large gate | Negative | Keep large gates for large aircraft |
| Flight sent to a remote stand | Negative | Avoid bus transfers |

</div>

For a flight $f$ assigned to gate $g$:

$$
R =
\begin{cases}
r_{\text{contact}} - d(g) - m(f, g) & \text{if } g \text{ is a contact gate} \\
-r_{\text{remote}} & \text{if } g \text{ is a remote stand}
\end{cases}
$$

where:

- $r_{\text{contact}}$ is the reward for using a contact gate,
- $d(g)$ is the walking distance penalty of gate $g$,
- $m(f, g)$ is the size mismatch penalty (0 if the aircraft matches the gate size),
- $r_{\text{remote}}$ is the remote stand penalty.

The values are set in the configuration file so that a contact gate is always better than a remote stand.

<br>

<br>


## $\color{#1F6FEB}{\textsf{The Agents}}$

The project evaluates five agents. Each one adds a single change compared to the previous one, so the effect of every change can be measured.

<div align="center">

| # | Agent | Type | Looks ahead? |
| --- | --- | --- | --- |
| 1 | Random | Picks a random valid gate | No |
| 2 | Greedy | Picks the closest valid gate that fits | No |
| 3 | DQN | Neural Q-network, replay buffer, target network | Yes |
| 4 | Double DQN | DQN with decoupled action selection and evaluation | Yes |
| 5 | Dueling Double DQN | Double DQN with a value / advantage network | Yes |

</div>

All agents use the same action mask, so they all produce valid schedules. The difference is the **quality** of the schedule.

### $\color{#E36209}{\textsf{What Each Comparison Answers}}$

<div align="center">

| Comparison | Question |
| --- | --- |
| 1 vs 2 | How much does a simple rule improve over random choices? |
| 2 vs 3 | Does learning to look ahead beat the greedy rule? |
| 3 vs 4 | Does reducing overestimation improve the policy? |
| 4 vs 5 | Does the dueling architecture help when many gates are similar? |

</div>


<br>


## $\color{#1F6FEB}{\textsf{Evaluation}}$

Each agent is trained with 5 random seeds, then tested on the **same set of new simulated days**.

<div align="center">

| Metric | What it measures |
| --- | --- |
| Contact gate rate | Share of flights assigned to a contact gate (the rest go to remote stands) |
| Walking distance | Average walking distance per flight |
| Size mismatch | How often small aircraft use large gates |
| Total reward | Overall quality of the day's schedule |
| Gap to optimal | Difference with the best possible schedule |

</div>

We also plot the **learning curves** (reward over training) and a **Gantt chart** of the final schedule, showing which flight uses each gate over the day.

<div align="center">

<img src="./assets/figures/gantt_example.png" alt="Gantt chart of a daily gate schedule" width="90%">

</div>

<br>


## $\color{#1F6FEB}{\textsf{Project Structure}}$

```bash
gatemind/
│
├── configs/
│   ├── airport.yaml           # Gates: group, max size, distance, count + remote stands
│   └── default.yaml           # Reward values, DQN hyperparameters, seeds, episodes
│
│   # Main package
├── gatemind/
│   ├── data/
│   │   └── schedule_generator.py  # Generates one simulated day of flights
│   ├── env/
│   │   ├── airport.py             # Loads gates from airport.yaml, tracks when each gate is free
│   │   └── gate_env.py            # Custom Gymnasium env: state, action mask, reward, transitions
│   ├── agents/
│   │   ├── random_agent.py        # Random valid gate
│   │   ├── greedy_agent.py        # Closest valid gate
│   │   ├── networks.py            # Q-network and dueling Q-network (PyTorch)
│   │   ├── replay_buffer.py       # Stores and samples transitions
│   │   └── dqn_agent.py           # DQN, with options for Double and Dueling
│   ├── planning/
│   │   └── optimal.py             # Offline best schedule (solver), optional
│   └── utils/
│       ├── metrics.py             # Contact gate rate, walking distance, size mismatch...
│       └── plotting.py            # Learning curves, Gantt charts
│
├── scripts/
│   ├── train.py               # Trains the DQN agents over 5 seeds
│   ├── evaluate.py            # Runs all agents on the test days, builds the metrics table
│   └── plot_results.py        # Generates the figures
│
├── results/                   # Saved models, logs, figures
│
├── tests/
│   ├── test_generator.py      # Flights are valid (times, categories)
│   ├── test_env.py            # Masking, rewards, gate occupancy, episode end
│   └── test_agents.py         # Agents only pick valid actions
│
├── main.py                    # Entry point: python main.py train / evaluate / plot
├── requirements.txt
└── README.md
```

<br>

## $\color{#1F6FEB}{\textsf{Results}}$

Results will be added after training.

```bash
python main.py train
python main.py evaluate
python main.py plot
```

<br>


## $\color{#1F6FEB}{\textsf{Roadmap}}$

- [x] Airport configuration and flight schedule generator
- [ ] Gymnasium environment with action masking, tested with the random agent
- [ ] Greedy baseline
- [ ] DQN agent
- [ ] Double DQN and Dueling DQN
- [ ] Offline optimal reference
- [ ] Evaluation, learning curves and Gantt charts
- [ ] Optional: random flight delays (stochastic environment)

<br>


## $\color{#1F6FEB}{\textsf{References}}$

1. Sutton, R. S., and Barto, A. G. (2018). [*Reinforcement Learning: An Introduction* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html). MIT Press.
2. Bouras, A., Ghaleb, M. A., Suryahatmaja, U. S., and Salem, A. M. (2014). [The Airport Gate Assignment Problem: A Survey](https://doi.org/10.1155/2014/923859). *The Scientific World Journal*.
3. Daş, G. S., Gzara, F., and Stützle, T. (2020). [A review on airport gate assignment problems: Single versus multi objective approaches](https://ideas.repec.org/a/eee/jomega/v92y2020ics0305048319306000.html). *Omega*, 92.
4. Li, H., Wu, X., Ribeiro, M., Santos, B., and Zheng, P. (2025). [Deep reinforcement learning approach for real-time airport gate assignment](https://doi.org/10.1016/j.orp.2025.100338). *Operations Research Perspectives*, 14.
5. Mnih, V., et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature*, 518, 529–533.
6. van Hasselt, H., Guez, A., and Silver, D. (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). *AAAI Conference on Artificial Intelligence*.
7. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., and de Freitas, N. (2016). [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581). *ICML*.