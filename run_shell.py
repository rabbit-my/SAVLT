import itertools
import os
import subprocess
import json
import re

param_space = {
    "shots":[16,32],
    "lr": [1e-3],
    "vision_lora_config": [
        json.dumps({"r": 8, "lora_alpha": 4, "gradual_step": 12})
    ],
    "seed": [5, 42, 123],
}

param_combinations = list(itertools.product(*param_space.values()))
param_keys = list(param_space.keys())

output_dir = "output_info_dir"
os.makedirs(output_dir, exist_ok=True)
results_path = os.path.join(output_dir, "res_avg.json")

results = []

for idx, combination in enumerate(param_combinations):
    params = dict(zip(param_keys, combination))
    print(f"\n>>> Running combination {idx + 1}/{len(param_combinations)}: {params}")

    cmd = [
    "python", "main.py",
    f"--shots={params['shots']}",
    f"--lr={params['lr']}",
    f"--vision_lora_config={params['vision_lora_config']}",
    f"--seed={params['seed']}",
    "--use_vision_peft",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout

        match = re.search(r"\[RESULT\] (.+)", stdout)
        if match:
            result_json = json.loads(match.group(1))
            result_json["status"] = "success"
            results.append(result_json)
            print("✓ Result recorded.")
        else:
            print("⚠️ No [RESULT] found.")
            results.append({
                "params": params,
                "status": "success",
                "note": "No [RESULT] found"
            })

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e.stderr.strip()}")
        results.append({
            "params": params,
            "status": "failure",
            "error": e.stderr.strip()
        })

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
