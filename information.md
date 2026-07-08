# Secure Federated Learning — Project Information

A privacy-preserving, distributed federated learning system built incrementally over ~7 weeks (May 19 → Jul 7, 2026): from a single-machine CNN baseline to a fault-tolerant, hierarchical, asynchronous swarm of self-electing clusters — containerized with Docker and observable through Prometheus + Grafana.

**Core idea:** multiple nodes collaboratively train a shared CNN on distributed (non-IID) MNIST data. Raw data never leaves a node — only gradient deltas travel over the network.

---

## Table of Contents

1. [Timeline (from git history)](#timeline-from-git-history)
2. [System Architecture](#system-architecture)
3. [File Map — every important file and its job](#file-map--every-important-file-and-its-job)
4. [Phase 0 — Centralized Baseline](#phase-0--centralized-baseline)
5. [Phase 1 — Baseline Federated Learning](#phase-1--baseline-federated-learning)
6. [Phase 2 — Communication & Protocol Infrastructure](#phase-2--communication--protocol-infrastructure)
7. [Phase 3 — Leader Election & Logical Clocks](#phase-3--leader-election--logical-clocks)
8. [Phase 4 — Async Training & the Convergence Journey](#phase-4--async-training--the-convergence-journey)
9. [Phase 5 — Hierarchical Aggregation, Pipeline, Exactly-Once](#phase-5--hierarchical-aggregation-pipeline-exactly-once)
10. [Phase 7 — Fault Tolerance & Resiliency](#phase-7--fault-tolerance--resiliency)
11. [Phase 8 — Observability (Launcher + Pygame Dashboard)](#phase-8--observability-launcher--pygame-dashboard)
12. [Phase 9 — Containerization (Docker, Kubernetes Foundation)](#phase-9--containerization-docker-kubernetes-foundation)
13. [Phase 10 — Prometheus + Grafana Monitoring (in progress)](#phase-10--prometheus--grafana-monitoring-in-progress)
14. [Supporting Tooling & Config](#supporting-tooling--config)
15. [End-to-End Data Flow](#end-to-end-data-flow)
16. [Key Results & Lessons](#key-results--lessons)
17. [How to Run](#how-to-run)
18. [Tech Stack](#tech-stack)

---

## Timeline (from git history)

27 commits, grouped by phase. Dates come straight from `git log`.

| Phase | Dates | Commits (chronological) | What was achieved |
|---|---|---|---|
| **0 — Centralized baseline** | May 19–20 | `909fb80` initial structure · `789f942` architecture · `500784d` dependencies · `3dd3daa` config system + dataset · `8ac4d84` CNN model · `1be294f` training/evaluation engine · `bc67758` logger, checkpointing, inference pipeline | A working single-machine MNIST trainer with config, logging, checkpoints, and inference |
| **1 — Baseline FL** | May 22–25 | `048463d` server/client/communication · `2f7e945` client + local trainer · `90e9e44` basic FL infrastructure · `78a7996` client simulation · `3521665` heterogeneity + transparency | FedAvg over gradient deltas with non-IID client partitions |
| **2 — Communication infra** | Jun 5–15 | `e083b84` nodes folder · `a431429` node metadata + leader provision · `1c27c78` leader_service, client objects, model sync · `d7363f2` weight-transfer hierarchy + binary serialization · `a00e794` protocol versioning + registry · `c648ac7` repo-map scaffolding | Transport layer fully decoupled from training logic |
| **3 — Leader election** | Jun 17–22 | `679d821` Raft via PySyncObj (`consensus_state` replaces static role) · `ce633dc` training pipeline joined with leader logic · `fe61eb8` leader election + cluster coordination done | Clusters self-elect leaders at runtime; no pre-assigned roles |
| **4 — Async training** | Jun 22 | `57ca12a` async training, logical clocks for staleness, global aggregator | Synchronous barrier deleted; K-or-T aggregation window |
| **5 — Hierarchical + pipeline** | Jun 23–24 | `a7adf99` shared init so all clusters train in the same weight-space valley · `7aea934` event-driven pipeline | Multi-cluster tree aggregation, exactly-once semantics |
| **7 — Fault tolerance** | Jun 25 | `ca4ff10` failover checked + resiliency updates | Timeout, retry+jitter, circuit breaker; live leader-kill test |
| **8 — Observability** | Jul 2 | `f6798a0` dashboard and launcher | One-command swarm launch + live pygame dashboard with interactive failover |
| **9 — Docker** | Jul 7 | `d9e3fdd` Docker foundation built | Single image, registry-based discovery, compose swarm verified end-to-end |
| **10 — Monitoring** *(uncommitted, Jul 7)* | Jul 7 | working tree: `server/metrics.py`, `monitoring/`, `gen_compose.py`, `/metrics` endpoint | Prometheus + Grafana stack, auto-provisioned dashboard, compose generator |

*(There is no Phase 6 — the numbering intentionally skips it; upstream resiliency goals collapsed into Phase 7.)*

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Global Server (server/app.py)               │
│  /register  /get_model  /cluster_update  /backup_log         │
│  /registered_nodes  /metrics (Prometheus)                    │
│  FederatedCoordinator + GlobalAggregator                     │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP                    ▲ scrape /metrics
        ┌──────────────────┴───────────┐      ┌──────┴──────┐
        │         Cluster 1            │      │ Prometheus  │◄── Grafana
        │  ┌────────────────────────┐  │      └─────────────┘   (dashboard)
        │  │ Node (elected LEADER)  │  │
        │  │  LeaderService (HTTP)  │  │
        │  │  FederatedCoordinator  │  │   Discovery:
        │  │  Update Pipeline       │  │    · loopback → UDP broadcast :9999
        │  └──────────┬─────────────┘  │    · Docker/k8s → registry polling
        │      Raft   │ (PySyncObj)    │
        │  ┌──────────┴─────────────┐  │
        │  │ Node (FOLLOWER)        │  │
        │  │  FederatedClient       │  │
        │  │  Lamport + VectorClock │  │
        │  └────────────────────────┘  │
        └──────────────────────────────┘
```

- The **global server** holds the canonical model, registers nodes, merges cluster models (hierarchical FedAvg), evaluates accuracy, and exports Prometheus metrics.
- Each **cluster** is a group of nodes that discover each other (UDP or registry), run **Raft** to elect a leader, and do federated learning internally.
- The **leader** runs `LeaderService`: collects gradient deltas from followers, aggregates, pushes the cluster model up to the global server, and adopts the merged result back.
- **Followers** run `FederatedClient`: download the model from their leader, train locally on their non-IID partition, upload only the delta.

---

## File Map — every important file and its job

```
secure_federated_learning/
│
├── server/
│   ├── app.py               # Global server — registry (+TTL pruning), cross-cluster merge,
│   │                        #   evaluation, backup sink, Prometheus /metrics
│   ├── coordinator.py       # Intra-cluster brain: pipeline, async K-or-T window, staleness
│   │                        #   gate, exactly-once dedup, vector-clock lockstep
│   ├── aggregator.py        # federated_averaging() + staleness_weighted_averaging()
│   ├── global_aggregator.py # Hierarchical FedAvg — one slot per cluster, re-average on push
│   ├── pipeline.py          # Event-driven pipes-and-filters update pipeline
│   ├── leader_service.py    # FastAPI app run by the elected leader (/get_model, /send_update,
│   │                        #   /model_status)
│   ├── model_manager.py     # Loads/saves the CNN model
│   └── metrics.py           # Prometheus gauges/counters for the /metrics endpoint (Phase 10)
│
├── nodes/
│   ├── node.py              # Node dataclass + __main__ startup: discovery → Raft → role loop
│   ├── consensus.py         # ClusterConsensus — PySyncObj Raft wrapper, @replicated set_leader
│   ├── discovery.py         # Dual-mode discovery: UDP broadcast (loopback) OR registry polling
│   └── peer_table.py        # In-memory peer registry with state tracking + leader lookup
│
├── client/
│   ├── client.py            # FederatedClient — rank-based partition, train, clock-stamp, upload,
│   │                        #   wait_for_merge() causal lockstep
│   ├── local_trainer.py     # local_train() — N epochs, Adam, CrossEntropyLoss
│   └── backup_client_code.py# Preserved Phase-1 client (used by run_federated.py simulation)
│
├── communication/
│   ├── clock.py             # LamportClock + VectorClock implementations
│   ├── model_sync.py        # ModelSync — download/upload/status/backup/cluster-push over HTTP
│   ├── resiliency.py        # resilient_request(): timeout + retry/backoff/jitter + circuit breaker
│   ├── delta.py             # compute_delta() / apply_delta()
│   ├── serialization.py     # torch.save → base64 binary weight serialization
│   ├── protocol.py          # PROTOCOL_VERSION — every message carries it, mismatch rejected
│   ├── registry.py          # register_node() — POST node info to the global server
│   └── updates.py           # (reserved for a future update schema)
│
├── models/cnn.py            # 2-block CNN for MNIST (Conv→Pool ×2 → FC ×2)
├── training/trainer.py      # train() — one epoch                    ┐ shared by centralized
├── training/evaluator.py    # evaluate() — loss + accuracy           ┘ and federated paths
├── data/dataset.py          # MNIST download, DataLoaders, client partitioning
├── evalution/evaluate.py    # evaluate_model() used by the global server after each merge
│
├── utils/
│   ├── env.py               # THE config seam — all network settings resolved from env vars
│   │                        #   with loopback defaults (Phase 9)
│   ├── config.py            # YAML config loader (repo-relative path, CONFIG_PATH override)
│   ├── partition.py         # Non-IID data partitioning across clients
│   ├── logger.py / checkpoint.py / seed.py   # logging, model checkpoints, reproducibility
│
├── configs/config.yaml      # All knobs: seed, batch size, lr, federated rounds, async window
│
├── main.py                  # Phase 0: centralized (non-federated) training baseline
├── run_federated.py         # Phase 1: round-based FL simulation driver (spawns client procs)
├── inference.py             # Loads checkpoints/best_model.pth, predicts one MNIST sample
│
├── swarm_launcher.py        # Standalone: pick topology, spawn + supervise the whole swarm
├── dashboard.py             # Standalone: live pygame dashboard + click-to-kill failover
├── smoke_test.py            # One command, 20 offline checks of all Phase 4/5/7 logic
│
├── Dockerfile               # Single CPU-only image serving both roles (~1 GB)
├── docker-compose.yml       # AUTO-GENERATED by gen_compose.py — swarm + monitoring stack
├── gen_compose.py           # Topology → docker-compose.yml generator (Phase 10)
│
├── monitoring/
│   ├── prometheus.yml       # Scrape global-server:8000/metrics every 5s
│   └── grafana/             # Auto-provisioned datasource + FL dashboard JSON
│
├── tools/repo_map.py        # Generates the repo dependency graph
├── docs/repo_map/           # Generated: REPO_MAP.md + mermaid + JSON (34 files, ~1312 LOC)
└── backup_codes/            # Preserved earlier-phase server implementation
```

---

## Phase 0 — Centralized Baseline

*(Not covered in imp_records.md — this is where the project started.)*

Before any federation, a complete single-machine training stack was built to prove the model and data pipeline:

- **`main.py`** — the centralized entry point: loads `configs/config.yaml`, seeds everything (`utils/seed.py`), builds DataLoaders, trains the CNN with Adam + CrossEntropyLoss for `training.epochs`, evaluates each epoch, and saves the best model to `checkpoints/best_model.pth` via `utils/checkpoint.py`. Auto-selects CUDA when available.
- **`inference.py`** — loads the best checkpoint and runs a single-sample MNIST prediction (actual vs. predicted label) — a minimal serving proof.
- **`utils/config.py` + `configs/config.yaml`** — one YAML drives everything: dataset (batch 64), model (10 classes), training (lr 0.001), federated (`num_clients: 2`, `num_rounds: 20`, `local_epochs: 2`), and later the `async_training` block (`min_updates: 1`, `window_seconds: 15`, `max_staleness: 0`, `consistency_model: "eventual"`).
- **`utils/logger.py`**, **`utils/seed.py`** — structured logging and reproducibility from day one.

This baseline matters: `training/trainer.py` and `training/evaluator.py` written here are the *same* functions the federated system uses later — the learning code never forked.

## Phase 1 — Baseline Federated Learning

**Goal:** multiple clients train locally; a server averages their updates into one shared model.

- **`models/cnn.py`** — `Input(1×28×28) → Conv(32)→ReLU→Pool → Conv(64)→ReLU→Pool → FC(3136→128) → FC(128→10)`.
- **`server/aggregator.py`** — FedAvg: element-wise mean of client weight dicts. Applied to **gradient deltas**, not full weights.
- **`communication/delta.py`** — clients send only the *change*: `delta = local − global` (`compute_delta`), server applies `global += averaged_delta` (`apply_delta`). Payload shrinks proportionally to model similarity.
- **`client/local_trainer.py`** — configurable local epochs, Adam (lr 0.001), CrossEntropyLoss.
- **`data/dataset.py` + `utils/partition.py`** — **non-IID partitioning**: each client gets a biased subset of digit classes, simulating real heterogeneous data (this later becomes the root cause of the Phase 4 oscillation and the Phase 7 accuracy finding).
- **`run_federated.py`** *(not in imp_records.md)* — the original round-based simulation driver: for each of `num_rounds`, spawns `num_clients` client subprocesses (`client/backup_client_code.py`), waits for all, repeats. This synchronous barrier is exactly what Phase 4 later deletes.

## Phase 2 — Communication & Protocol Infrastructure

**Goal:** separate transport from learning logic so the system can grow without touching training code.

- **`communication/protocol.py`** — `PROTOCOL_VERSION` in every message; mismatches rejected immediately (no silent breakage across versions).
- **`communication/serialization.py`** — `torch.save()` → binary buffer → base64 for JSON transport; exact tensor dtypes/devices restored. Replaced lossy/slow nested-float-list encoding.
- **`communication/model_sync.py`** — `ModelSync` encapsulates all HTTP: `download_model()` (GET `/get_model` + version check), `upload_update()` (POST delta + clock metadata), `upload_cluster_update()` (push aggregate to global server).
- **`nodes/discovery.py`** — UDP broadcast discovery on port 9999: three daemon threads — `advertise()` (broadcast self every 5s), `browse()` (listen + update peer table), `cleanup()` (evict peers unseen >15s).
- **`nodes/peer_table.py`** — `node_id → Node` registry with state-change logging and `get_cluster_leader(cluster_id)`.
- **`communication/registry.py`** — `register_node()` POSTs full node metadata to the global server's `/register`, at startup and on every role change.

## Phase 3 — Leader Election & Logical Clocks

**Goal:** no pre-assigned leader — each cluster elects one at runtime and the aggregation role migrates to the winner.

### Raft consensus (`nodes/consensus.py`)
Uses **PySyncObj** to run Raft inside each cluster. `set_leader()` is `@replicated`, so *every* node in the Raft group learns who won — not just the leader. `update_role()` maps Raft state to `consensus_state = "leader" | "follower"`.

### Dynamic role loop (`nodes/node.py`)
Every node starts as follower; after discovery + Raft stabilization it polls every 2s:
- role changed → `register_node()` re-POSTs the new state to the global server
- `leader` → start `LeaderService` (FastAPI) in a daemon thread
- `follower` → start/restart `FederatedClient` in a daemon thread

A follower promoted mid-training drops its client thread and starts serving; a demoted leader becomes follower on the next poll.

### Rank-based partition assignment (`client/client.py`)
Each client sorts all follower node-IDs in the cluster and takes its index as its data-partition rank — so when leadership changes, no partition is wasted and no two nodes train on the same data.

### Logical clocks (`communication/clock.py`)
- **LamportClock** — scalar clock: `tick()` before send, `update(received)` on receive (`max(local, received)+1`). Guarantees: A happened-before B ⇒ `A.ts < B.ts`.
- **VectorClock** — per-node counter vector with `tick()`, `update()` (element-wise max merge), `happened_before()`, `is_concurrent()`. Exact causality: `A.happened_before(B)` ⟺ A causally precedes B.

**What the clocks buy:**

| Property | Mechanism |
|---|---|
| Deduplication of retried uploads | coordinator drops repeated `(node_id, lamport_ts)` pairs |
| Deterministic aggregation order | updates sorted by Lamport ts before FedAvg — same logical order every run, regardless of network arrival order |
| Causality / staleness visibility | each update's vector clock shows if a node missed rounds |
| Split-brain hint | two simultaneous "leaders" produce diverging timestamp timelines, visible in logs |

## Phase 4 — Async Training & the Convergence Journey

**Goal:** delete the synchronous barrier. Previously the coordinator aggregated *only* when exactly `num_clients` updates arrived — one dead follower stalled the cluster forever.

**Consistency model chosen consciously:** *eventual* consistency (`config.yaml`). Nodes may briefly hold different model versions; they converge over rounds.

### What was built (`server/coordinator.py`, `server/aggregator.py`)
1. **K-or-T aggregation window** — first update of a batch arms a `threading.Timer(window_seconds)`. Full batch before it fires → aggregate immediately; timer fires first → aggregate whatever arrived, dropping stragglers. **This permanently kills the deadlock.**
2. **Versioning** — `model_version` bumps per aggregation and is returned by `/get_model`; each update carries its `base_version`; `staleness = model_version − base_version` (the classic concurrent-write measure).
3. **Staleness-weighted averaging** — weight `wᵢ = 1/(1+stalenessᵢ)`, normalized, so stale updates contribute less.
4. **Thread safety** — one `threading.Lock` guards all shared coordinator state; network I/O happens *outside* the lock so a slow peer never blocks incoming updates.

### The convergence journey (the hard, non-obvious part)

| Attempt | Result | Root cause |
|---|---|---|
| Naive staleness weighting | Accuracy **flat at ~51%** | A delta is only valid against the exact base it was computed from — applying `local − W₀` on top of the moved-on `W₁` injects a misaligned displacement. **Delta compression + asynchrony are incompatible without a guard.** |
| Staleness gate (`max_staleness: 0`) | **Oscillated 48% ↔ 82%** | The gate often left one fresh update per window; single-worker aggregation on non-IID data replaces the good global model with one biased local model |
| **Vector-clock causal lockstep** | **Smooth climb to ~96%** | see below |

**The lockstep fix:** workers were *lapping* each other — a fast worker re-trained before its previous update merged, so someone was always stale. Fix: the coordinator keeps `model_vc = {node_id → highest lamport_ts merged into the model}` (a causal fingerprint). After uploading update `ts`, a worker calls `wait_for_merge(ts)` — polling the leader's `/model_status` until `model_vc[my_id] ≥ ts`, i.e. *"my contribution is provably in."* Only then does it download and start the next round. A plain version counter can't answer "is **my** update in?" — the vector clock can. This forces every aggregation to be a full multi-worker FedAvg, and it stays deadlock-free: if a partner dies, the window timer still fires and releases the waiter.

## Phase 5 — Hierarchical Aggregation, Pipeline, Exactly-Once

**Goal:** scale from one cluster to a *tree* of clusters, and harden the update path.

### Hierarchical FedAvg (`server/global_aggregator.py`)
Each cluster leader pushes its **full aggregated model** (not a delta) tagged with `cluster_id` + `model_version`. The `GlobalAggregator` keeps **one slot per cluster** and re-averages all slots on every push — naturally asynchronous, no barrier, and naturally idempotent (last-write-wins per slot). `--num_workers` became a CLI arg; a cluster of N auto-sets `num_workers = N−1` (the leader doesn't train).

### The cross-cluster convergence fix
Naive merging fluctuated 60–80% and *dropped* when a second cluster joined: each leader started from its own random init, so cluster models lived in **different weight-space basins** — averaging them lands on the ridge between (worse than either). Two changes lifted it to a stable **~99%**:
1. **Shared initialization** — a fresh leader downloads the canonical model from the global server instead of random-initializing. All clusters start in the same basin (commit `a7adf99`).
2. **Feedback loop** — `/cluster_update` *returns* the freshly-merged global model and the cluster adopts it before its next round. Clusters re-center every round and never drift.

Together: proper **HierFAVG** — workers→leader (reduce), leader↔global (reduce + broadcast down).

### Event-driven pipeline (`server/pipeline.py`)
`receive_update` refactored into explicit pipes-and-filters; an `UpdateEvent` flows through ordered stages, first drop short-circuits:

```
ProtocolFilter → ClusterRouter → IdempotencyAppliedFilter → ClockStage
   → StalenessFilter → IdempotencyInflightFilter → BackupCopier → merger (FedAvg)
```

The **BackupCopier** duplicates each accepted update to the global server's durable `/backup_log` (bounded per-cluster ring) so a leader crash mid-window doesn't lose in-flight updates.

### Exactly-once application
A retried `/send_update` must never be applied twice (double-counting a delta corrupts the model). Each update carries idempotency key `update_id = "{node_id}:{lamport_ts}"` — stable across retries because it derives from the logical clock. Two-tier dedup:
- **Applied tier (persistent):** `lamport_ts ≤ model_vc[node_id]` → already merged → drop. `model_vc` doubles as a bounded high-water mark (no unbounded seen-ID set).
- **In-flight tier (per window):** `_buffered_ids` catches duplicates buffered but not yet aggregated.

This is what makes the Phase 7 retries safe.

## Phase 7 — Fault Tolerance & Resiliency

**Goal:** one node dying never stalls the swarm. Analysis showed most of this was *already earned* — the Phase 4 window timer survives a dead worker, Phase 3 Raft re-elects a dead leader — so only one new module was needed.

### `communication/resiliency.py` — `resilient_request()`
Wraps every inter-node HTTP call with:
- **Timeout `(3.05, 30)`** — a dead peer fails in ~3s instead of hanging forever (hot-path calls previously had *none*).
- **Retry + exponential backoff + FULL jitter** — randomized so lockstep-synchronized workers don't all hammer a recovering leader in the same instant (retry amplification / self-DDoS).
- **Per-endpoint circuit breaker** — repeated failures → fail fast for a cooldown → half-open probe. Keyed by host+path so a failing `/backup_update` can't trip `/cluster_update`.

The client's round-level retry also moved from a fixed 3s sleep to exponential backoff with equal jitter. Retries are safe *only because* Phase 5 made writes idempotent.

### Deliberately reused / skipped
Leader failover (Raft already does it — verified live), failure detection (Raft heartbeats + discovery timeout; liveness decoupled from training so a slow trainer isn't evicted), rate limiting (the causal lockstep already prevents a node getting ahead), readiness probes (the window timer already skips absent nodes).

### The honest failover finding
Live test — kill the leader mid-training: **liveness passed** (Raft re-elected, new leader pulled shared-init, resiliency layer re-resolved, zero stall) but **accuracy cratered 96% → 51%**: in a 3-node/2-worker cluster the promoted worker stops training, leaving one non-IID worker (half the digit classes). **Rule discovered: to keep N training workers through a leader failure you need N+2 nodes** (leader + N workers + 1 standby). The system chooses liveness over accuracy when the pool shrinks — a topology property, not a bug.

## Phase 8 — Observability (Launcher + Pygame Dashboard)

**Goal:** debug the swarm without `print()`. Built as **standalone programs that import no project code** — observability can never destabilize the system.

- **`swarm_launcher.py`** — pick a topology (`--topology 3,2` or interactive), it spawns the global server + every node with auto-computed `--num_workers` and unique ports, captures output to `logs/launch/*.log`, publishes `logs/launch/processes.json` (node→PID). Ctrl+C stops the swarm.
- **`dashboard.py`** — read-only pygame dashboard (poller @1s, render @30fps) combining only existing outputs: `/registered_nodes` + `/backup_log`, the global-server log (accuracy/version history), and `processes.json` + `tasklist` (liveness). Renders top-bar SLIs, per-cluster panels (gold "L" leader, blue "W" workers), a live accuracy chart. **Interactive failover: click a node, press K** — it's killed via `taskkill`, greys out, and the gold leader marker visibly migrates as Raft re-elects.
- **Tracing without Jaeger** — `update_id` already appears in every captured log, so an update's journey (device → leader → global) is traceable by grepping the correlation ID across `logs/launch/*.log`.
- **`smoke_test.py`** — one command, in-process: clocks, all pipeline stages, async window, versioning, staleness gate, exactly-once dedup, cross-cluster aggregator, circuit breaker. 20 checks, exit 0 = green.

## Phase 9 — Containerization (Docker, Kubernetes Foundation)

**Goal:** package the swarm for Docker today and Kubernetes next — with **zero behavior change in loopback mode** (no env vars set = Phases 1–8 exactly, so the launcher/dashboard and the `loopback-simulation` git branch remain valid).

### The two assumptions Docker breaks

| Assumption | Why it breaks | Fix |
|---|---|---|
| Everything on `127.0.0.1` | each container has its own IP; loopback means "myself" | advertise a routable DNS name via `NODE_HOST`; all server URLs from env |
| Peers found by UDP broadcast | Docker bridge and k8s pod networks don't route broadcast | **registry-based discovery** — poll the global server |

(+ a hardcoded Windows config path, made repo-relative with a `CONFIG_PATH` override.)

### Key pieces
- **`utils/env.py`** — the single config seam; no other file reads `os.environ`. `global_server_url()`, `node_host()`, `api_port()`, `discovery_mode()` (`udp` default | `registry`), `registry_poll_interval()`. Same code runs loopback / compose / k8s.
- **`nodes/discovery.py`** — registry mode adds `registry_heartbeat()` (re-register periodically = liveness signal) and `registry_browse()` (poll `/registered_nodes`, reconcile the PeerTable — feeds Raft its partners and lets followers find the leader). UDP path untouched; stale-peer `cleanup()` shared by both modes.
- **`server/app.py`** — registry TTL pruning: `/registered_nodes` drops entries whose heartbeat is older than `REGISTRY_TTL` (default `0` = disabled, so loopback is unchanged; compose sets `20`). Fixes crashed leaders lingering forever.
- **`Dockerfile`** — **one image, both roles** (default CMD = global server; node services override). CPU-only torch keeps it ~1 GB vs ~5 GB; MNIST raw data baked in so nothing downloads at runtime.
- **`docker-compose.yml`** — `global-server` (healthcheck) + per-node services with `NODE_HOST=<service name>`, `DISCOVERY_MODE=registry`, `depends_on: service_healthy`. Deliberately shaped 1:1 for k8s: global-server → Deployment+Service; each cluster → StatefulSet + headless Service.

### Verified end-to-end
`docker compose up --build`: registry discovery (no UDP), Raft election across containers, follower→leader training over Docker DNS (`http://cluster1-node2:5001`), leader aggregation (`v1, v2, v3…`), hierarchical push + re-sync to the global model. Everything after election is **unchanged Phase 3–5 logic** — proof the containerization is non-invasive.

*Windows 11 Home note:* Docker Desktop requires the WSL2 backend (no Hyper-V on Home). If "virtualization not detected": enable Virtual Machine Platform + WSL, `bcdedit /set hypervisorlaunchtype auto`, then **Restart** (not Shut Down).

## Phase 10 — Prometheus + Grafana Monitoring (in progress)

*(Newest work — currently uncommitted; not yet in imp_records.md.)*

Phase 8's pygame dashboard was the right call for a no-Docker Windows setup; with Docker available, the industry-standard stack was added alongside it.

- **`server/metrics.py`** — Prometheus instrumentation via `prometheus_client`. Global-model gauges (`fl_test_accuracy_percent`, `fl_test_loss`, `fl_global_model_version`, `fl_clusters_active`), per-cluster labelled metrics (`fl_cluster_model_version`, `fl_cluster_updates_total` counter), and topology gauges (`fl_registered_nodes` by role). Design: **event metrics are set in request handlers as things happen; state gauges are recomputed at scrape time** via `refresh_from_registry()` from the live node registry.
- **`server/app.py`** — new `GET /metrics` endpoint (refreshes topology gauges under the registry lock, returns Prometheus text format); `/cluster_update` now exports accuracy/loss/versions/counters after every merge. The global server is the natural single metrics source — it already knows merged accuracy, versions, and (via the registry) who is alive.
- **`monitoring/prometheus.yml`** — scrapes `global-server:8000/metrics` every 5s (Docker DNS).
- **`monitoring/grafana/`** — **fully auto-provisioned**: datasource (Prometheus) + a ready-made "Federated Learning Overview" dashboard (global version, test accuracy stat, active clusters, leaders/followers, accuracy-over-time, loss-over-time, per-cluster version, cluster update rate). Anonymous viewer access enabled — open `http://localhost:3000` and the dashboard is already there.
- **`gen_compose.py`** — `docker-compose.yml` is no longer hand-edited: `python gen_compose.py 3,2` generates services for any topology (same convention as the launcher: `num_workers = N−1`, min 2 nodes/cluster, regions rotated per cluster), using YAML anchors to DRY the shared node config, and appends the Prometheus + Grafana services. `requirements.txt`/`Dockerfile` gained `prometheus_client`.

---

## Supporting Tooling & Config

*(Pieces not documented in imp_records.md.)*

- **`tools/repo_map.py` → `docs/repo_map/`** — generates a repository dependency map (REPO_MAP.md with a mermaid module-dependency graph, plus `.mmd` and `.json` exports). Snapshot at generation: 34 Python files, ~1,312 lines.
- **`backup_codes/` + `client/backup_client_code.py`** — earlier-phase server/client implementations preserved intact rather than deleted, keeping the evolution inspectable (the Phase-1 simulation `run_federated.py` still drives the backup client).
- **`configs/config.yaml`** — single source for all tunables; see Phase 0. The `async_training` block is the live control surface for Phase 4 behavior (`min_updates`, `window_seconds`, `max_staleness`, `consistency_model`).
- **`checkpoints/`, `logs/`** — best-model checkpoints (Phase 0 path) and per-process launch logs (Phase 8 path).

---

## End-to-End Data Flow

```
1. STARTUP     node.py: register_node() → DiscoveryService (UDP or registry)
               → Raft cluster forms from PeerTable partners
2. ELECTION    Raft elects a leader; consensus_state replicated to all peers;
               every role change re-registers with the global server
3. LEADER      starts LeaderService (/get_model, /send_update, /model_status);
               pulls shared-init model from the global server
4. FOLLOWER    loop: download model → compute rank → load non-IID partition
               → local_train (Adam) → compute_delta → lamport.tick(), vc.tick()
               → upload_update(delta, node_id, ts, vc, base_version)
               → wait_for_merge(ts)   # causal lockstep
5. AGGREGATE   leader pipeline: protocol → route → dedup(applied) → clock
               → staleness gate → dedup(in-flight) → backup-copy → K-or-T window
               → sort by lamport ts → FedAvg → apply_delta → model_version++
6. GLOBAL      leader POSTs full cluster model → GlobalAggregator re-averages
               per-cluster slots → evaluate_model() → accuracy logged +
               exported to Prometheus → merged model returned → cluster adopts it
7. OBSERVE     pygame dashboard (loopback) / Grafana via Prometheus (Docker)
```

---

## Key Results & Lessons

| Result | Where earned |
|---|---|
| Delta compression + asynchrony are incompatible without a staleness guard (accuracy flat at 51% otherwise) | Phase 4 |
| Single-worker aggregation on non-IID data causes oscillation (48%↔82%); vector-clock causal lockstep fixes it → **~96%** | Phase 4 |
| Independently-initialized clusters average onto a bad ridge; shared init + feedback loop → stable **~99%** | Phase 5 |
| Retries are only safe on top of idempotent writes (logical-clock-derived `update_id`, two-tier dedup) | Phase 5 → 7 |
| Surviving a leader failure with N training workers requires **N+2 nodes**; liveness vs. accuracy is a topology trade | Phase 7 |
| Killing UDP broadcast + loopback assumptions is the real k8s migration work; the rest is mechanical manifests | Phase 9 |

---

## How to Run

**Prerequisites:** `pip install -r requirements.txt` (torch, torchvision, fastapi, uvicorn, pysyncobj, requests, pyyaml, pygame, prometheus_client).

### Loopback (recommended for development)
```bash
python swarm_launcher.py --topology 3,2   # terminal 1 — whole swarm
python dashboard.py                       # terminal 2 — live dashboard (click node + K = kill)
python smoke_test.py                      # offline verification, 20 checks
```

### Docker + monitoring
```bash
python gen_compose.py 3          # generate compose for any topology
docker compose up --build -d     # swarm + Prometheus + Grafana
# Grafana:    http://localhost:3000   (FL dashboard pre-provisioned)
# Prometheus: http://localhost:9090
docker compose down
```

### Manual / other entry points
```bash
python -m server.app                                   # global server alone
python -m nodes.node --cluster_id cluster1 --port 4001 \
       --region us-east --latency 10 --client_id 1 --num_workers 2
python main.py                                         # centralized baseline training
python inference.py                                    # single-sample prediction from checkpoint
curl http://localhost:8000/registered_nodes            # who's alive + roles
curl http://localhost:8000/metrics                     # Prometheus text format
```

---

## Tech Stack

| Library / Tool | Purpose |
|---|---|
| PyTorch + torchvision | CNN, local training, tensor serialization, MNIST |
| FastAPI + Uvicorn | Leader service + global server HTTP APIs |
| PySyncObj | Raft consensus for leader election |
| Requests | Inter-node HTTP client |
| Python sockets / threading | UDP discovery; concurrent roles; Timer-based async window |
| PyYAML | Configuration |
| pygame | Standalone live dashboard (loopback path) |
| Docker + Compose | Single-image containerization, k8s-shaped topology |
| prometheus_client + Prometheus + Grafana | Metrics export, scraping, auto-provisioned dashboards |
