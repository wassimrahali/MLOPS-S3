# Atelier 3 : Création d'un pipeline avec Prefect

This project demonstrates a reproducible scikit-learn training workflow
orchestrated with Prefect. It uses the Iris dataset bundled with scikit-learn, so
no external data file is required. The model and held-out evaluation accuracy are
saved/reported locally.

## Setup (Ubuntu, Python 3.10+)

```bash
cd mlops-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The regular Python entry point trains and evaluates the model:

```bash
python main.py
```

The Prefect CLI supports `all`, `train` (also `entrainement`), `evaluate`,
`code`, and `install`:

```bash
python pipeline_prefect.py --flow train
python pipeline_prefect.py --flow evaluate
python pipeline_prefect.py --flow code
python pipeline_prefect.py --flow all
```

The code flow installs dependencies, formats and checks only
`model_pipeline.py`, `main.py`, and `pipeline_prefect.py`, then runs `pytest tests/`.
The remote Git fetch step is disabled by default. To enable it, set `REPO_URL` and
pass `--fetch-project` to the `code` or `all` flow. The repository is cloned into
`project_clone/` on the first run and pulled on subsequent runs.

## Prefect server and deployments (three terminals)

Activate the virtual environment in each terminal, and run these commands from
the project directory.

**Terminal 1 — start the local Prefect server:**

```bash
prefect server start
```

The UI is available at <http://localhost:4200>.

**Terminal 2 — point the client at the server and serve deployments:**

```bash
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
python deploiement_prefect.py
```

This keeps the deployments served and available to run. The all-pipeline
deployment is scheduled daily at 02:00.

**Terminal 3 — list and start deployments:**

```bash
prefect deployment ls
prefect deployment run 'all/ml-pipeline-all'
prefect deployment run 'train/ml-pipeline-train'
```

For development/testing, the two long-running processes can be started with
`nohup`, for example `nohup prefect server start > prefect-server.log 2>&1 &`
and `nohup python deploiement_prefect.py > prefect-deployments.log 2>&1 &`.
For production, run the server and deployment serving process as managed
`systemd` services with a dedicated user, pinned dependencies, and suitable
logging/restart policies.

## Push to a remote Git repository

Create an empty remote repository first, then from this project directory:

```bash
git init
git add .
git commit -m "Add Prefect MLOps workshop pipeline"
git branch -M main
git remote add origin https://github.com/<owner>/<repository>.git
git push -u origin main
```

Alternatively, with GitHub CLI installed and authenticated:

```bash
gh repo create <owner>/<repository> --public --source=. --remote=origin --push
```
# MLOPS-S3
