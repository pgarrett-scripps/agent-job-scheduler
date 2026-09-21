# agent_job_scheduler (`ajs`)

Local admission control so several coding agents can share one machine without
oversubscribing it — and without silently corrupting each other's benchmark timings.

Several agents (Claude Code, Codex) work on different projects on the same computer.
Each independently launches test suites, builds and searches. Nothing coordinates them,
so the machine thrashes, and a timing run that overlaps someone else's `cargo build -j20`
produces a number that is wrong in a way you cannot see afterwards.

`ajs` gives them one queue to ask.

```console
$ ajs submit --cpu 8 --mem 8G -- cargo test --release
queued job 41: cargo test --release

$ ajs submit --exclusive --max-runtime 20m -- ./bench.sh
queued job 42: ./bench.sh

$ ajs status
cpu 8/20   mem 8192/56000 MB   gpu 0/1   load 3.41
running  41  koth_rust   8  8192M   12s   cargo test --release
queued   42  blitz       exclusive     waiting for resources; reserved to start by 47s from now
```

Job 42 waits for the machine to empty, waits another 10 seconds for it to go quiet, runs
alone, and afterwards tells you whether anything else interfered.

## Install

```bash
uv sync --extra mcp
uv run ajs daemon start
```

The daemon is a singleton — one per machine. Everything else is a stateless client.

To keep it running across logins:

```bash
ajs service install    # writes a systemd --user unit
```

## What to queue

Heavy work only: **several cores at once, many GB of RAM, or the GPU.** Parallel builds
and test suites, data processing over large files, searches, sweeps, model runs,
benchmarks you intend to report.

Everything else runs directly — document builds, linters, git, package installs,
single-threaded scripts. Wall-clock time is not the test: a five-minute single-threaded
LaTeX build is not worth queueing, a ten-second 20-core compile is.

The scheduler queues work; it never vetoes it. A busy machine is not a reason for an
agent to refuse or postpone a task, and the MCP instructions say so explicitly — that
distinction has to be stated, or agents read "the machine is busy" as "I should stop".

## Use from an agent

Register the MCP server once per agent platform:

```bash
ajs mcp-config            # prints the Claude Code snippet
ajs mcp-config --codex    # prints the Codex snippet
```

Agents then get `submit_job`, `run_job`, `wait_for_job`, `job_status`, `get_job_logs`,
`cancel_job`, `list_jobs` and `scheduler_status`. The tool descriptions tell the model
when to queue and when not to bother.

Anything that can run a shell command can use the CLI instead — no configuration needed.

## Timing runs

`--exclusive` is not a separate mode. It expands into a request for *every* CPU slot on
the machine, so the ordinary resource accounting already guarantees nothing runs
alongside it. On top of that it:

1. waits `settle_seconds` after the last job exits, so writeback and dying processes are
   not still perturbing the measurement;
2. measures **foreign CPU** over the run — system-wide busy time minus the job's own
   cgroup usage — which is the one quantity that actually invalidates a benchmark;
3. stamps the record `contended` if something else was using more than
   `contention_threshold` cores, so you know not to publish that number.

```console
$ ajs ps --all
 43  done !contended   blitz   61s   ./bench.sh
$ ajs logs 43 -n 1
# and the record says: foreign load 3.812 cores over 61.0s (232.5 CPU-s not
# attributable to this job; threshold 2.00)
```

Sampling load average before and after does *not* work for this, because at the end of a
run the load is dominated by the job itself. Foreign CPU is the signal that means
something.

## What a job sees

Jobs are launched with their granted allocation in the environment:

| Variable | Meaning |
|---|---|
| `AJS_JOB_ID` | this job's id |
| `AJS_CPU` | CPU slots granted |
| `AJS_MEM_MB` | memory granted |
| `AJS_EXCLUSIVE` | `1` during a timing run |

Use `$AJS_CPU` rather than `nproc` to size thread pools. `CPUQuota` caps a job's
throughput but does not hide cores, so `nproc` still reports all 20 and a job that trusts
it will oversubscribe its own allocation.

```bash
ajs submit --cpu 8 -- sh -c 'cargo test --release -j $AJS_CPU'
```

## Scheduling

- **Counted semaphores** for cpu / mem / gpu. A job declares what it needs and waits
  until that much is free.
- **Named locks** for logical conflicts (`--lock sage-index`), independent of hardware.
- **Disk floor.** Jobs are refused admission if free space would drop below
  `disk_floor_mb`. On a nearly full disk this is what stops a job taking the desktop down
  with it.
- **Reservations and backfill.** A job needing most of the machine would otherwise never
  run — small jobs keep trickling in and there is never an instant when enough is free.
  The head of the queue gets a promised start time, and later jobs may jump ahead only if
  they provably finish before it. This is why `--max-runtime` is mandatory.
- **Fair share.** Priority bands (`interactive` > `batch` > `background`), then
  round-robin between projects, then FIFO. One agent with fifty queued jobs cannot starve
  another agent's one job.
- **Leases** (`acquire_lease`) for work that must run inside the agent's own process.
  Heartbeat-based, so a dead agent's hold is reclaimed rather than wedging the queue.
- **GPU by VRAM.** The card is scheduled on memory, not device count: `--gpu 1 --gpu-mem
  2G` lets two such jobs share a 4 GB card, while `--gpu 1` alone reserves all of it
  (safe, but serialising). `--gpu-exclusive` is a separate switch from `--exclusive`, so
  a CPU benchmark does not idle the GPU and vice versa. Jobs that do not ask for the GPU
  run with `CUDA_VISIBLE_DEVICES=""`.
- **Foreign load.** Processes you started by hand are measured and deducted from what the
  scheduler will hand out, so `ajs` does not admit a job onto cores something else is
  already using. `ajs status` shows it on its own line:

  ```
  cpu 0/20   mem 0/57851 MB   gpu 0/1   load 8.23
  outside ajs 3 cpu, 25297 MB (deducted from what the scheduler will hand out)
  ```

  A job held up by this says so, and gets no start-time estimate — nothing tells the
  scheduler when a process it did not start will exit. Set `track_external_load: false`
  if ajs is the only thing that ever runs on the machine.

  Two allowances keep this usable on a laptop. A desktop baseline
  (`external_cpu_allowance`, 2 cores, plus `mem_reserve_mb`) is subtracted before
  anything is charged, and **exclusive jobs ignore foreign load entirely** — on a machine
  that always has a browser open, a job asking for the whole machine would otherwise
  never start. Exclusivity means no other *ajs job* runs alongside; the contention report
  tells you what the desktop actually did.

## Enforcement

Jobs run in transient systemd scopes with `CPUQuota` and `MemoryMax` applied, so a job
that declares 4 GB cannot quietly take 40. Without systemd it falls back to a plain
process group, which still allows clean termination of the whole subtree.

## Configuration

`ajs config --init` writes defaults to `~/.config/ajs/config.json`. Capacities are
autodetected; `mem_reserve_mb` is held back so the desktop stays responsive.

## Development

```bash
just            # lint, format, typecheck, test
just dev        # run the daemon in the foreground
```

See `DESIGN.md` for the reasoning behind the architecture.
