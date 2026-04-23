# GRSA Dev/Deploy Quick Reference

## Machines
| Machine | Role | Access |
|---|---|---|
| thebunker | Development | Local |
| TheSentinel | Production | `ssh -p 2222 matt_sent@[IP]` or Tailscale |

---

## Production Services (TheSentinel)

| Service | Type | Schedule |
|---|---|---|
| `grsa-pipeline.timer` | One-shot daily pipeline | 06:30 ACST |
| `grsa02.service` | Live selector daemon | Continuous |

### Check service status
```bash
systemctl --user status grsa-pipeline.timer
systemctl --user status grsa02.service
```

### View logs
```bash
# Pipeline
journalctl --user -u grsa-pipeline.service --since today

# Live selector
journalctl --user -u grsa02.service -f
```

### Restart services
```bash
systemctl --user restart grsa02.service
# Pipeline restarts automatically at 06:30 — no manual restart needed
```

---

## Key Paths

| What | Path |
|---|---|
| Project code | `~/projects/grsa/` |
| Data vault | `/media/matt_sent/vault/dishlicker_data/` |
| Model 1 | `/media/matt_sent/vault/dishlicker_data/models/random_forest_baseline.pkl` |
| Model 2 artifacts | `/media/matt_sent/vault/dishlicker_data/models/v2/` |
| Betfair certs | `~/projects/grsa/certs/` |
| Live selector logs | `~/projects/grsa/logs/live_selector_YYYY-MM-DD.log` |
| Systemd units | `~/.config/systemd/user/` |

### Data subdirectories
```
dishlicker_data/data/
├── raw/
├── jsonl/
├── predictions/
├── results/
├── bets/
├── analysis/
├── track_statistics/
├── index/
└── logs/
    └── bets/
        ├── daily/
        ├── monthly/
        ├── settled/
        └── daily_updated/
```

---

## Dev Workflow (thebunker)

### Setup (first time or after re-clone)
```bash
cd ~/projects/grsa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# .env and certs/ are NOT in the repo — copy manually if missing
cp /media/matthew/Projects_Vault/grsa_prod_dev/.env ~/projects/grsa/
cp -r /media/matthew/Projects_Vault/grsa_prod_dev/certs ~/projects/grsa/
```

### Daily dev session
```bash
cd ~/projects/grsa
source .venv/bin/activate
git pull                  # always pull before starting work
# ... make changes, test ...
git add .
git commit -m "brief description of change"
git push
```

---

## Deploy to TheSentinel

### One command
```bash
~/deploy_grsa.sh
```

### What it does
```bash
cd ~/projects/grsa
git pull
systemctl --user restart grsa02.service
echo "Deployed $(git log -1 --oneline)"
```

> **Note:** Pipeline changes take effect at next 06:30 run automatically.
> Only `grsa02.service` needs a manual restart.

---

## Manual Pipeline Run (TheSentinel)
```bash
cd ~/projects/grsa
source .venv/bin/activate
python -m src.pipeline.daily_pipeline --date YYYY-MM-DD
```

---

## Monthly Tasks (thebunker or TheSentinel)
```bash
# Download fresh track statistics
python -m src.scraper.greyhound_track_scraper_20251102_f --all

# Merge predictions
python -m src.tools.merge_predictions --date YYYY-MM-DD

# Update edge finder date range first (edit edge_finder_all_data.py lines ~561-568)
# Then run:
python src/analytics/edge_finder_all_data.py
python src/analytics/analyse_box_bias_20260218_a.py
# Output: box_bias_lookup_YYYY-MM-DD.json — used automatically by bet_smart_box_bias
```

---

## Git Basics (if rusty)
```bash
git status              # what's changed
git diff                # see changes in detail
git log --oneline -10   # last 10 commits
git pull                # get latest from GitHub
git push                # send commits to GitHub
```

---

## Environment Variables (.env)
Located at `~/projects/grsa/.env` — never committed to Git.
```
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
BETFAIR_API_KEY=
BETFAIR_USERNAME=
BETFAIR_PASSWORD=
```

---

## Troubleshooting

**Pipeline didn't fire at 06:30**
```bash
journalctl --user -u grsa-pipeline.service --since today
systemctl --user status grsa-pipeline.timer
```

**grsa02 crashed**
```bash
journalctl --user -u grsa02.service -n 50
systemctl --user restart grsa02.service
```

**Config verification (model 2)**
```bash
cd ~/projects/grsa/src/model_v2
source ../../.venv/bin/activate
python config.py
```

**Uptime Kuma**
Monitor at `100.90.210.35:3001`
- GRSA Pipeline (Model 1) — heartbeat every 1440 min
- GRSA Live Selector (Model 2) — heartbeat every 60 min

