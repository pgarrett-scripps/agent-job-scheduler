# agent_job_scheduler (`ajs`) — Design

## Problem

Several coding agents (Claude Code, Codex) run simultaneously on **one laptop**, each on its own
project. They independently launch work: test suites, MS searches, Rust builds, benchmark timings.
Nothing coordinates them, so they oversubscribe the machine — and a benchmark timing run that
happens to overlap with someone's `cargo build -j20` produces a number that is silently wrong.

**Goal:** one local admission-control service that every agent asks before running expensive work,
so jobs queue instead of colliding, and timing runs get a provably quiet machine.

**Non-goal (decided):** the scheduler does not launch or supervise agents. Agents are started by
hand as always; they merely submit jobs to the scheduler. No DAGs, no retries, no pipelines.

## Target machine

i7-12700H (20 threads / 14 cores), 62 GB RAM, RTX 3050 Ti (4 GB), NVMe with ~18 GB free of 1.9 TB.
Single-node. The GPU is small enough that it is effectively an exclusive resource, and **free disk
is scarce enough that it must be schedulable, not assumed.**

## Architecture

```
  Claude Code (proj A)      Codex (proj B)       you, in a shell        cron / hooks
         |                        |                      |                   |
    ajs-mcp (stdio)          ajs-mcp (stdio)         ajs CLI             ajs CLI
         \________________________\______________________/___________________/
                                   |
                          unix socket (JSON-RPC)
                                   |
                          +--------v---------+
                          |      ajsd        |   singleton daemon
                          |  scheduler loop  |   systemd --user service
                          |  SQLite (WAL)    |   sole writer of state
                          +--------+---------+
                                   |
                        systemd-run --user --scope
                          (cgroup limits + accounting)
                                   |
                            the actual job
```

### The one constraint that drives everything

**MCP servers are spawned per client session.** Every Claude Code window and every Codex session
gets its *own* `ajs-mcp` process. So the MCP server cannot hold scheduler state — if it did, you'd
have N independent schedulers that each think they own the machine, which is the exact problem
we're solving.

Therefore: **`ajsd` is a singleton daemon and the only writer.** The CLI and the MCP server are
both thin, stateless clients over a Unix socket. The MCP server is ~150 lines on top of the CLI's
client library. This is why "make it an MCP server" is the wrong frame — MCP is one of three front
doors onto something else.

### Why three front doors

| Interface | For | Why it's needed |
|---|---|---|
| `ajs` CLI | any agent, shell, cron, git hooks | Zero config. Works in Codex, Claude Code, a plain terminal, or a CI script. Universal fallback. |
| `ajs-mcp` | Claude Code + Codex | Structured tools with descriptions the model actually reads — it teaches the agent *when* to schedule, which a CLI can't. |
| `ajsd` socket | the above | Implementation detail, not a user surface. |

Both Claude Code and Codex speak MCP over stdio, so one server registers in both:
- Claude Code: `claude mcp add ajs -- ajs-mcp` (or `.mcp.json` committed per project)
- Codex: `[mcp_servers.ajs]` in `~/.codex/config.toml`

This — not cross-OS portability — is what "works across platforms" means here.

## Execution model: executor, not broker

Two ways to gate a job:

- **Broker/lease:** agent asks permission, gets a lease, runs the command *itself*, releases.
- **Executor:** agent submits the command, *the daemon* runs it, agent polls for the result.

**Executor is primary.** It's more robust (an agent that dies mid-job can't leak a lease and wedge
the queue) and it gives you things you want anyway: captured logs, a run history, real cgroup
accounting of what each job actually consumed, and timing metadata attached to the run record.

A **lease API is kept as an escape hatch** for work that genuinely must run in the agent's own
process (interactive REPL, something needing the agent's TTY). Leases carry a heartbeat and expire,
so a dead agent's hold is reclaimed automatically.

## Resource model

Everything is a **counted semaphore**. Nothing is special-cased.

| Resource | Capacity | Notes |
|---|---|---|
| `cpu` | 20 | slots, declared per job |
| `mem_mb` | ~56000 | leaves headroom for the desktop |
| `gpu` | 1 | 4 GB — effectively exclusive |
| `disk_mb` | dynamic | **guard**: refuse to start if free space would drop below a floor |
| named locks | 1 each | e.g. `lock:sage-index`, `lock:scratch-dir` — for logical conflicts, not hardware |
| custom semaphores | configurable | e.g. `api:anthropic=3` to cap concurrent API-hammering jobs |

A job declares what it needs:

```yaml
cmd: ["cargo", "test", "--release"]
cwd: ~/Repos/koth_rust
cpu: 8
mem_mb: 8000
max_runtime: 30m          # required — see backfill below
class: batch              # interactive | batch | background
```

Acquisition of all resources for a job happens in **one SQLite transaction**. A job either gets
everything or waits — no partial holds, so no deadlock between two half-satisfied jobs.

## Foreign load

The capacity table above describes the machine, not the scheduler's share of it. Anything
launched outside ajs — a hand-run search, a browser, another agent shelling out directly —
takes cores that ajs would otherwise count as free. Left uncorrected the scheduler admits
jobs onto a saturated box, and an exclusive timing run is contaminated by precisely the
load it was meant to exclude.

So every tick measures what it cannot account for:

```
external_cpu = (system-wide busy delta) − (delta summed over ajs job cgroups)
external_mem = (MemTotal − MemAvailable) − (sum of ajs job cgroup memory)
```

This is the same subtraction the contention monitor performs per job, applied continuously
to the whole machine. Two properties are deliberate:

- **It is reported as usage, never as reduced capacity.** Shrinking `cap` would make a
  20-core exclusive job *impossible* the moment a browser opened. Treating foreign work as
  usage makes it wait instead.
- **A job blocked only by foreign load gets no reservation promise.** Reservations bound a
  wait by projecting running jobs' `max_runtime`; nothing declares when a browser closes.
  Such a reservation is flagged `external` and does not veto backfill — holding cores empty
  for a start time we cannot predict would strand small jobs for nothing.

CPU is smoothed (15 s half-life) because a one-second sample is spiky enough that a single
compile would evict a queued job. Memory is not: it is a level rather than a rate, and
reacting late to it risks an OOM. Foreign CPU is floored to whole cores and capped at
`cap.cpu − 1`, so a pathological reading throttles the queue but can never wedge it.

This is also what makes `contention_threshold: 2.0` defensible. Ambient desktop load on
this machine alone exceeds two cores, so without the admission guard every timing run
would be stamped `contended` until the flag stopped meaning anything.

## Timing runs

A timing run is **not a mode**. It is:

```yaml
exclusive: true     # == requests all 20 cpu slots + the global `machine` lock
```

Because it requests the entire CPU capacity, the existing resource model already guarantees nothing
else can be running. Draining is emergent: running jobs finish naturally (never killed), new ones
can't start, the exclusive job takes the machine.

Three small additions on top, and only three:

1. **Settle delay.** After the last job exits, wait `settle_seconds` (default 10) before starting —
   page cache writeback and dying processes are still perturbing things.
2. **Contamination stamp.** Sample `/proc/loadavg` (and optionally cgroup CPU pressure) immediately
   before and after. If load exceeded a threshold, stamp the record `contended: true`. The number
   still gets recorded — you just know not to publish it. `ajs ps --all` marks such runs
   `!contended`. **This is the part that protects you from silently wrong benchmarks.**
3. **Foreign-load warning.** The daemon only knows about *its* jobs. If a browser or a stray
   terminal is eating cores, warn at submit time and stamp the record.

### Starvation, and why `max_runtime` is mandatory

An exclusive job that needs all 20 slots will never run if small jobs keep trickling in — there's
never a moment when 20 slots are free. Standard fix, borrowed from Slurm's EASY backfill:

- The head-of-queue job gets a **reservation**: a computed future start time, based on the declared
  `max_runtime` of everything currently running.
- Smaller jobs may **backfill** — start ahead of it — *only if* they provably finish before that
  reservation.

So your benchmark gets a guaranteed start time, and the machine still stays busy until then. This
only works if every job declares a max runtime, which is why it's required rather than optional
(and it gives you free job timeouts as a side effect).

## Fairness between projects

Agents are selfish; one project queueing 50 jobs shouldn't starve another's single job.

- **Priority classes:** `interactive` (an agent is blocked waiting) > `batch` > `background`.
- **Fair-share within a class:** round-robin across projects, least-recently-served first.
- **Per-project concurrency cap** as a blunt backstop.

Every job carries `project` + `session_id`, so the scheduler can attribute work and `ajs status`
shows you which agent is hogging the box.

## Agent ergonomics — the token-burn trap

The obvious failure mode: an agent submits a job, then busy-polls `job_status` every few seconds and
burns tokens doing nothing. Countermeasures, in order of importance:

1. **Long-polling `wait`.** The daemon holds the request open until the job finishes or the timeout
   expires. One tool call, agent sleeps efficiently, no polling loop.
2. **`submit_and_wait`** for the common short-job case — behaves like just running the command.
3. **Documented async pattern** for long jobs: submit → go do unrelated work → `wait` at the end.
4. **Tool descriptions that state the policy**, e.g. *"Use this for any command expected to exceed
   30 seconds or 2 cores. Do not run such commands directly with Bash."* The description is the only
   place an agent reliably learns the rule.

Nothing *forces* an agent to cooperate. If advisory turns out to be insufficient, Claude Code's
`PreToolUse` hook can intercept `Bash` calls matching known-expensive patterns and rewrite them
through `ajs`. That's phase 2 — try trust first, it's usually enough and far less brittle.

## Failure and recovery

- **Daemon restart:** on boot, reconcile — scan for orphaned cgroups/PIDs from the previous life,
  adopt or reap them, rebuild the available-resource counts from what's actually running.
- **Dead agent:** its *jobs* keep running (they're the daemon's children, not the agent's) and
  results are retrievable later by job id. Its *leases* expire on heartbeat timeout.
- **Disk guard:** with 18 GB free, a job that writes a large search output can fill the root
  filesystem and take the desktop down with it. Hard floor, checked at admission *and* periodically
  during a run.
- **Global kill switch:** `ajs drain` (finish current, start nothing new) and `ajs pause`.

## Stack

Python 3.12 + `uv`, matching your existing tooling and cookiecutter.

- State: **SQLite in WAL mode**. Single writer, crash-safe, zero operational burden, and the run
  history is queryable with `sqlite3` when you want to analyze benchmark results.
- MCP: **FastMCP**.
- CLI: **Typer**.
- Execution: `systemd-run --user --scope` with `CPUQuota` / `MemoryMax` / `AllowedCPUs` for real
  enforcement plus free accounting; plain subprocess + process group as fallback.
- Daemon: systemd user unit, socket in `$XDG_RUNTIME_DIR`.

The daemon is not performance-critical — it sleeps, wakes on events, and does bookkeeping. If the
CLI's ~100 ms Python startup becomes annoying in tight loops, port *only* the thin client to Rust
later; the protocol makes that a drop-in swap.

## Build order

Each phase is independently useful — you can stop after any of them.

| Phase | Delivers | You can... |
|---|---|---|
| **0** | schema, daemon, socket, `ajs submit/status/wait/logs`, cpu+mem semaphores, FIFO | ...stop oversubscribing the CPU. Usable day one. |
| **1** | `exclusive`, settle delay, contamination stamp, reservations + backfill | ...trust your timing runs. |
| **2** | `ajs-mcp`, registered in Claude Code + Codex, long-poll `wait` | ...have agents schedule themselves. |
| **3** | cgroup enforcement, disk guard, leases, `ajs status` | ...stop jobs from taking down the desktop. |
| **4** | fair-share, per-project caps, benchmark history queries | ...keep multiple agents polite, and mine past runs. |

Phase 0 + 1 is the real core; 2 is what makes it disappear into your workflow.

## Status

Phases 0-4 are implemented and tested (86 tests). Not built: a live-updating `ajs top`
TUI (`ajs status` is a snapshot instead), and learned per-command resource defaults --
see below.

## Open questions

- **Declared vs. measured resources.** Agents will guess `cpu: 8` badly. Phase 3's cgroup accounting
  gives you actuals — worth feeding back as a per-command default learned from history?
- **GPU at 4 GB.** Probably always `gpu: 1` exclusive. Revisit if you ever run two small models.
- **Cross-machine.** Design is single-node throughout. If a second box ever appears, the socket
  becomes TCP and `resources` grows a `node` column — but don't build for it now.
