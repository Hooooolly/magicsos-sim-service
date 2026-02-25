# Sim Service Log Persistence

## Goal
- Keep interactive bridge and collector logs on persistent storage so pod restart does not lose debugging history.

## Runtime Log Paths
- Interactive bridge (`run_interactive.py`):
  - default log dir: `/data/embodied/logs/sim-service`
  - file pattern: `run_interactive_<session>_<timestamp>_<pid>.log`
  - latest symlink: `run_interactive_<session>_latest.log`
- Standalone collector (`isaac_pick_place_collector.py`):
  - default log dir: `/data/embodied/logs/sim-service/collector`
  - fallback: `<output_dir>/_logs`
  - file pattern: `isaac_pick_place_collector_<timestamp>_<pid>.log`

## Health Endpoint
- Bridge health now returns log metadata:
  - `log_file`
  - `log_dir`
- Endpoint: `GET /health` on bridge port.

## Environment Variables
- `SIM_LOG_DIR`: override base runtime log directory for interactive bridge.
- `SIM_STDIO_LOG_TEE`:
  - default `1` (enabled)
  - set `0` to disable stdout/stderr tee logging.
- `SIM_LOG_LEVEL`: Python logging level for interactive bridge (`INFO` default).
- `COLLECT_LOG_DIR`: override standalone collector log directory.
- `COLLECT_LOG_LEVEL`: standalone collector logging level (`INFO` default).

## Quick Checks
```bash
# Find latest interactive bridge log
ls -lt /data/embodied/logs/sim-service | head

# Tail current session log by symlink
tail -f /data/embodied/logs/sim-service/run_interactive_<session>_latest.log

# Read log path from bridge health
curl -s http://127.0.0.1:5800/health | jq '.log_file, .log_dir'
```
