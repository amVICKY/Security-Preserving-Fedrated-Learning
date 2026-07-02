"""
Swarm dashboard (standalone, read-only viewer + node-kill control).

Does NOT import or modify any project code. It consumes what the running system
already produces:
  - GET /registered_nodes, /backup_log         (live topology + roles + backups)
  - parses logs/launch/global_server.log        (accuracy + version history)
  - reads logs/launch/processes.json            (node -> PID map, from the launcher)

Interactive: click a node to select it, press K to kill that node's process, then
watch the swarm fail over (Raft re-elects; the gold leader marker moves). Killing
is done via the OS (taskkill) on the PID published by the launcher — the dashboard
still never touches the FL code.

Usage:
    python dashboard.py
"""

import argparse
import ast
import json
import os
import re
import subprocess
import threading
import time

import pygame
import requests

ACC_RE = re.compile(r"Test Accuracy\s*:\s*([\d.]+)%")
LOSS_RE = re.compile(r"Test Loss\s*:\s*([\d.]+)")
AGG_GV_RE = re.compile(r"\[GLOBAL AGG\].*->\s*global_version\s*(\d+)")
AGG_CLUSTERS_RE = re.compile(r"\[GLOBAL AGG\].*merging.*?(\{.*?\})")


def parse_log(path):
    history, cluster_versions, cur_gv, pending_acc = [], {}, 0, None
    if not os.path.exists(path):
        return history, cluster_versions
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = AGG_GV_RE.search(line)
                if m:
                    cur_gv = int(m.group(1))
                    mc = AGG_CLUSTERS_RE.search(line)
                    if mc:
                        try:
                            cluster_versions = ast.literal_eval(mc.group(1))
                        except Exception:
                            pass
                    continue
                ma = ACC_RE.search(line)
                if ma:
                    pending_acc = float(ma.group(1))
                    continue
                ml = LOSS_RE.search(line)
                if ml and pending_acc is not None:
                    history.append((cur_gv, pending_acc, float(ml.group(1))))
                    pending_acc = None
    except Exception:
        pass
    return history, cluster_versions


def load_processes(path):
    """raft_port -> {pid, name} (only node entries have raft_port)."""
    out = {}
    if not os.path.exists(path):
        return out
    try:
        for m in json.load(open(path, encoding="utf-8")):
            if "raft_port" in m:
                out[int(m["raft_port"])] = {"pid": m["pid"], "name": m["name"]}
    except Exception:
        pass
    return out


def running_pids():
    """Set of running PIDs (str) via tasklist. None if unavailable -> treat all alive."""
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=4).stdout
        pids = set()
        for line in out.splitlines():
            parts = line.split('","')
            if len(parts) >= 2:
                pids.add(parts[1].strip('"').strip())
        return pids or None
    except Exception:
        return None


def kill_pid(pid):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=4)
        return True
    except Exception:
        return False


class Poller(threading.Thread):
    def __init__(self, server, log_path, proc_path, interval=1.0):
        super().__init__(daemon=True)
        self.server = server.rstrip("/")
        self.log_path = log_path
        self.proc_path = proc_path
        self.interval = interval
        self._lock = threading.Lock()
        self._state = {"online": False, "nodes": {}, "backups": {}, "history": [],
                       "cluster_versions": {}, "procs": {}, "running": None}
        self._running = True

    def run(self):
        while self._running:
            s = {"online": False, "nodes": {}, "backups": {}, "history": [],
                 "cluster_versions": {}, "procs": {}, "running": None}
            try:
                s["nodes"] = requests.get(self.server + "/registered_nodes", timeout=2).json()
                s["backups"] = requests.get(self.server + "/backup_log", timeout=2).json()
                s["online"] = True
            except Exception:
                pass
            s["history"], s["cluster_versions"] = parse_log(self.log_path)
            s["procs"] = load_processes(self.proc_path)
            s["running"] = running_pids()
            with self._lock:
                self._state = s
            time.sleep(self.interval)

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def stop(self):
        self._running = False


BG = (18, 20, 26)
PANEL = (30, 33, 43)
EDGE = (52, 56, 70)
TEXT = (222, 224, 230)
MUTED = (140, 146, 160)
LEADER = (240, 192, 64)
FOLLOWER = (84, 152, 232)
DEAD = (70, 60, 64)
GREEN = (96, 214, 124)
RED = (228, 96, 96)
SEL = (255, 255, 255)


class Dashboard:
    def __init__(self, poller):
        pygame.init()
        pygame.display.set_caption("Federated Swarm — Dashboard")
        self.W, self.H = 1120, 700
        self.BOTTOM = 34
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.poller = poller
        self.f_big = pygame.font.SysFont("consolas", 46)
        self.f_h = pygame.font.SysFont("consolas", 24)
        self.f = pygame.font.SysFont("consolas", 18)
        self.f_s = pygame.font.SysFont("consolas", 14)
        self.clickable = []          # per-frame: (cx, cy, r, node_id, cluster, name, pid, alive)
        self.selected = None         # node_id
        self.message = "click a node to select, then press K to kill it"

    def text(self, s, x, y, font=None, color=TEXT):
        self.screen.blit((font or self.f).render(str(s), True, color), (x, y))

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_k:
                        self._kill_selected()
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self._select_at(e.pos)
            self.draw(self.poller.snapshot())
            pygame.display.flip()
            self.clock.tick(30)
        self.poller.stop()
        pygame.quit()

    def _select_at(self, pos):
        mx, my = pos
        for (cx, cy, r, nid, cid, name, pid, alive) in self.clickable:
            if (mx - cx) ** 2 + (my - cy) ** 2 <= (r + 3) ** 2:
                self.selected = nid
                tag = name or nid[:8]
                self.message = f"selected {tag} ({'alive' if alive else 'dead'}) — press K to kill"
                return
        self.selected = None

    def _kill_selected(self):
        for (cx, cy, r, nid, cid, name, pid, alive) in self.clickable:
            if nid == self.selected:
                if pid and alive:
                    ok = kill_pid(pid)
                    self.message = f"killed {name or nid[:8]} (pid {pid})" if ok else f"kill failed for {name}"
                else:
                    self.message = "selected node has no known/alive PID (launch via swarm_launcher.py)"
                return
        self.message = "no node selected"

    def draw(self, st):
        self.clickable = []
        self.screen.fill(BG)
        self._top_bar(st)
        self._clusters(st)
        self._chart(st)
        self._footer()

    def _cluster_map(self, st):
        clusters = {}
        for nid, info in (st["nodes"] or {}).items():
            cid = info.get("cluster_id", "?")
            clusters.setdefault(cid, []).append((nid, info.get("consensus_state"), info.get("port")))
        return clusters

    def _top_bar(self, st):
        pygame.draw.rect(self.screen, PANEL, (0, 0, self.W, 72))
        self.text("FEDERATED SWARM", 20, 10, self.f_h, TEXT)
        pygame.draw.circle(self.screen, GREEN if st["online"] else RED, (250, 22), 7)
        self.text("online" if st["online"] else "waiting for swarm...", 264, 14, self.f_s, MUTED)
        hist = st["history"]
        acc = hist[-1][1] if hist else None
        loss = hist[-1][2] if hist else None
        gv = hist[-1][0] if hist else 0
        nodes = st["nodes"] or {}
        leaders = sum(1 for n in nodes.values() if n.get("consensus_state") == "leader")
        self.text(f"{acc:.2f}%" if acc is not None else "--", 20, 34, self.f_big, GREEN if acc else MUTED)
        self.text("global accuracy", 220, 46, self.f_s, MUTED)
        self._stat("global ver", gv, 430, 20)
        self._stat("loss", f"{loss:.4f}" if loss is not None else "--", 560, 20)
        self._stat("peers", len(nodes), 700, 20)
        self._stat("leaders", leaders, 820, 20)
        self._stat("clusters", len(self._cluster_map(st)), 940, 20)

    def _stat(self, label, value, x, y):
        self.text(value, x, y, self.f_h, TEXT)
        self.text(label, x, y + 28, self.f_s, MUTED)

    def _clusters(self, st):
        clusters = self._cluster_map(st)
        cversions = st.get("cluster_versions", {})
        backups = st.get("backups", {})
        procs = st.get("procs", {})
        running = st.get("running")
        x0, y0, w = 16, 88, 528
        area_h = self.H - y0 - self.BOTTOM
        if not clusters:
            self.text("no nodes registered yet — start the swarm with swarm_launcher.py",
                      x0 + 12, y0 + 16, self.f, MUTED)
            return
        n = len(clusters)
        ph = min(200, (area_h - (n - 1) * 12) // n)
        y = y0
        for cid in sorted(clusters):
            self._panel(cid, clusters[cid], cversions.get(cid, "?"), backups.get(cid, 0),
                        procs, running, x0, y, w, ph)
            y += ph + 12

    def _panel(self, cid, members, version, backup_depth, procs, running, x, y, w, h):
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, EDGE, (x, y, w, h), 1, border_radius=8)
        self.text(cid, x + 14, y + 10, self.f_h, TEXT)
        self.text(f"v{version}", x + w - 92, y + 12, self.f_h, GREEN)
        self.text(f"nodes: {len(members)}   backup buf: {backup_depth}", x + 14, y + 40, self.f_s, MUTED)
        members = sorted(members, key=lambda m: (m[1] != "leader", m[0]))
        gap = 84
        per_row = max(1, (w - 60) // gap)
        for i, (nid, role, port) in enumerate(members):
            meta = procs.get(int(port)) if port is not None else None
            pid = meta["pid"] if meta else None
            name = meta["name"] if meta else None
            alive = True if running is None or pid is None else (str(pid) in running)
            col = DEAD if not alive else (LEADER if role == "leader" else FOLLOWER)
            px = x + 42 + (i % per_row) * gap
            py = y + 92 + (i // per_row) * 56
            r = 19
            if nid == self.selected:
                pygame.draw.circle(self.screen, SEL, (px, py), r + 4, 2)
            pygame.draw.circle(self.screen, col, (px, py), r)
            pygame.draw.circle(self.screen, BG, (px, py), r, 2)
            label = "L" if role == "leader" else "W"
            if not alive:
                label = "x"
            self.text(label, px - 5, py - 10, self.f_s, BG)
            self.text((name or nid[:6]).replace(cid + "_", ""), px - 24, py + r + 1, self.f_s, MUTED)
            self.clickable.append((px, py, r, nid, cid, name, pid, alive))

    def _chart(self, st):
        x, y, w, h = 560, 88, self.W - 560 - 16, self.H - 88 - self.BOTTOM
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, EDGE, (x, y, w, h), 1, border_radius=8)
        self.text("GLOBAL ACCURACY", x + 14, y + 10, self.f_h, TEXT)
        hist = st["history"]
        pad = 44
        gx, gy = x + pad, y + 44
        gw, gh = w - pad - 20, h - 44 - 30
        for frac, lab in [(0.0, "100"), (0.5, "50"), (1.0, "0")]:
            yy = gy + int(frac * gh)
            pygame.draw.line(self.screen, EDGE, (gx, yy), (gx + gw, yy), 1)
            self.text(lab, x + 12, yy - 9, self.f_s, MUTED)
        if not hist:
            self.text("waiting for first global round...", gx + 10, gy + gh // 2, self.f, MUTED)
            return
        gvs = [p[0] for p in hist]
        lo, hi = min(gvs), max(gvs)
        span = max(1, hi - lo)
        pts = [(gx + int((gv - lo) / span * gw), gy + int((100 - acc) / 100 * gh))
               for gv, acc, _ in hist]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, GREEN, False, pts, 2)
        for p in pts:
            pygame.draw.circle(self.screen, GREEN, p, 3)
        self.text(f"v{hist[-1][0]}: {hist[-1][1]:.2f}%", gx + gw - 150, gy + 4, self.f_s, GREEN)
        self.text("global version ->", gx + gw - 150, gy + gh + 10, self.f_s, MUTED)

    def _footer(self):
        y = self.H - self.BOTTOM + 6
        pygame.draw.line(self.screen, EDGE, (0, self.H - self.BOTTOM), (self.W, self.H - self.BOTTOM), 1)
        self.text(self.message, 16, y, self.f_s, TEXT)
        self.text("click = select   |   K = kill selected   |   ESC = quit",
                  self.W - 430, y, self.f_s, MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--log", default=os.path.join("logs", "launch", "global_server.log"))
    ap.add_argument("--procs", default=os.path.join("logs", "launch", "processes.json"))
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    poller = Poller(args.server, args.log, args.procs, args.interval)
    poller.start()
    Dashboard(poller).run()


if __name__ == "__main__":
    main()
