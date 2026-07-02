"""
Swarm launcher (standalone) — pick the topology, it spawns the whole swarm.

This program does NOT import or modify any existing project code. It only shells
out to the same commands you would type by hand:
    uvicorn server.app:app --port 8000
    python -m nodes.node --cluster_id ... --port ... --num_workers ...
and captures each process's output to logs/launch/*.log (so a dashboard can read
them later without touching the running system).

Usage:
    python swarm_launcher.py                  # interactive: asks clusters + nodes
    python swarm_launcher.py --topology 3,2   # cluster1=3 nodes, cluster2=2 nodes
    python swarm_launcher.py --topology 3 --no-server   # don't start global server

Convention: in a cluster of N nodes, 1 is elected leader (aggregates) and N-1 train,
so --num_workers is set to N-1 automatically. Minimum 2 nodes per cluster.

Press Ctrl+C to stop the whole swarm.
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "logs", "launch")
PYTHON = sys.executable          # use the same interpreter/env the launcher runs in
SERVER_PORT = 8000
RAFT_PORT_BASE = 4001            # api_port = raft_port + 1000 (handled by node.py)
REGIONS = ["us-east", "eu-west", "ap-south", "us-west", "eu-north"]


def parse_topology(args):
    """Return a list of node-counts, one per cluster. e.g. [3, 2]."""
    if args.topology:
        counts = [int(x) for x in args.topology.split(",") if x.strip()]
    else:
        n = int(input("How many clusters? ").strip())
        counts = []
        for i in range(n):
            c = int(input(f"  Cluster {i + 1}: how many nodes? ").strip())
            counts.append(c)

    for i, c in enumerate(counts):
        if c < 2:
            raise SystemExit(
                f"Cluster {i + 1} has {c} node(s). Need >= 2 (1 leader + >=1 worker)."
            )
    return counts


def build_plan(counts):
    """Turn node-counts into concrete launch specs (cluster_id, port, client_id, workers)."""
    plan = []
    port = RAFT_PORT_BASE
    for ci, node_count in enumerate(counts, start=1):
        cluster_id = f"cluster{ci}"
        num_workers = node_count - 1          # leader excluded
        region = REGIONS[(ci - 1) % len(REGIONS)]
        for j in range(1, node_count + 1):
            plan.append({
                "cluster_id": cluster_id,
                "port": port,
                "api_port": port + 1000,
                "region": region,
                "latency": 10 * j,
                "client_id": j,
                "num_workers": num_workers,
            })
            port += 1
    return plan


def launch(plan, start_server=True):
    os.makedirs(LOG_DIR, exist_ok=True)
    procs = []
    meta = []   # published to processes.json so the dashboard can identify/kill nodes

    def spawn(name, cmd):
        log_path = os.path.join(LOG_DIR, f"{name}.log")
        f = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
        procs.append((name, p, f))
        return p, log_path

    if start_server:
        cmd = [PYTHON, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", str(SERVER_PORT)]
        p, path = spawn("global_server", cmd)
        meta.append({"name": "global_server", "role": "server", "pid": p.pid, "api_port": SERVER_PORT})
        print(f"[LAUNCH] global server  -> :{SERVER_PORT}   log: {path}")
        print("[LAUNCH] waiting 4s for the global server to come up...")
        time.sleep(4)   # nodes fetch shared-init from it, so it must be up first

    for spec in plan:
        name = f"{spec['cluster_id']}_node{spec['client_id']}"
        cmd = [
            PYTHON, "-m", "nodes.node",
            "--cluster_id", spec["cluster_id"],
            "--port", str(spec["port"]),
            "--region", spec["region"],
            "--latency", str(spec["latency"]),
            "--client_id", str(spec["client_id"]),
            "--num_workers", str(spec["num_workers"]),
        ]
        p, path = spawn(name, cmd)
        meta.append({"name": name, "cluster_id": spec["cluster_id"], "client_id": spec["client_id"],
                     "raft_port": spec["port"], "api_port": spec["api_port"], "pid": p.pid})
        print(f"[LAUNCH] {name:18s} -> raft :{spec['port']}  api :{spec['api_port']}  "
              f"workers={spec['num_workers']}   log: {path}   pid: {p.pid}")

    with open(os.path.join(LOG_DIR, "processes.json"), "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)

    return procs


def supervise(procs):
    print(f"\n[LAUNCH] {len(procs)} process(es) running. Tail a log with:")
    print(f"         Get-Content -Wait {os.path.join(LOG_DIR, 'global_server.log')}")
    print("[LAUNCH] Press Ctrl+C to stop the whole swarm.\n")
    try:
        while True:
            time.sleep(1)
            for name, p, _ in procs:
                if p.poll() is not None and p.returncode not in (0, None):
                    # a process exited unexpectedly; report once
                    pass
    except KeyboardInterrupt:
        print("\n[LAUNCH] stopping all processes...")
    finally:
        for name, p, f in procs:
            if p.poll() is None:
                p.terminate()
        # give them a moment, then hard-kill stragglers
        time.sleep(2)
        for name, p, f in procs:
            if p.poll() is None:
                p.kill()
            f.close()
        print("[LAUNCH] all stopped.")


def main():
    parser = argparse.ArgumentParser(description="Launch a federated-learning swarm.")
    parser.add_argument("--topology", type=str, default=None,
                        help='Comma-separated nodes per cluster, e.g. "3,2" (2 clusters).')
    parser.add_argument("--no-server", action="store_true",
                        help="Do not start the global server (assume it is already running).")
    args = parser.parse_args()

    counts = parse_topology(args)
    plan = build_plan(counts)

    print("\n=== SWARM PLAN ===")
    print(f"clusters: {len(counts)} | total nodes: {sum(counts)} | layout: {counts}")
    print("==================\n")

    procs = launch(plan, start_server=not args.no_server)
    supervise(procs)


if __name__ == "__main__":
    main()
