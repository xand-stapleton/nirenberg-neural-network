import wandb

# -------- CONFIG --------
ENTITY = "logml"
PROJECT = "Ainstein_kw"
# ------------------------

api = wandb.Api()

runs = {
    "sh_3_0": "vjk6kx0u",
    "sh_2_0": "e773rhg9",
    "sh_3_2": "rf0lm5ev",
    "sh_3_1": "qmsjhlip",
    "sh_3_3": "cq7rqkl3",
    "sh_5_3": "m94oq4hr",
    "sh_5_4": "f13xumyb",
    "sh_5_1": "ffiaq8e1",
    "sh_7_1": "3fc3a4uo",
    "sh_5_2": "6uw455s5",
    "sh_1_1": "jztzjqc8",
    "sh_1_0": "ggngsigl",
    "sh_7_4": "vqrku3ip",
    "sh_7_2": "0kmiruxj",
    "sh_7_3": "uyi69tuj",
}

results = {}

for label, run_id in runs.items():
    path = f"{ENTITY}/{PROJECT}/{run_id}"
    try:
        run = api.run(path)
        results[label] = {
            "run_id": run_id,
            "pretty_name": run.name,
            "group": run.group,  # 👈 here
            "state": run.state,
            "url": run.url,
        }
    except Exception as e:
        results[label] = {
            "run_id": run_id,
            "error": str(e),
        }

# Pretty print
for k, v in results.items():
    print(f"{k:35s} → {v}")
