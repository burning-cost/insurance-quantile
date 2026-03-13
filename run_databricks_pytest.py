"""
Submit insurance-quantile pytest suite (v0.2.0, including EQRN subpackage)
to Databricks serverless compute.

Uses the REST API directly — no cluster spec required.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

api_base = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

RUN_ID = uuid.uuid4().hex[:8]
WORKSPACE_FOLDER = "/Workspace/insurance-quantile"
NOTEBOOK_PATH = f"{WORKSPACE_FOLDER}/run_pytest_v2"

# ---------------------------------------------------------------------------
# Read source files
# ---------------------------------------------------------------------------
BASE = "/home/ralph/repos/insurance-quantile"


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


# Main package source files
src_files = {
    "__init__.py":      read_file(f"{BASE}/src/insurance_quantile/__init__.py"),
    "_types.py":        read_file(f"{BASE}/src/insurance_quantile/_types.py"),
    "_model.py":        read_file(f"{BASE}/src/insurance_quantile/_model.py"),
    "_calibration.py":  read_file(f"{BASE}/src/insurance_quantile/_calibration.py"),
    "_tvar.py":         read_file(f"{BASE}/src/insurance_quantile/_tvar.py"),
    "_loading.py":      read_file(f"{BASE}/src/insurance_quantile/_loading.py"),
    "_exceedance.py":   read_file(f"{BASE}/src/insurance_quantile/_exceedance.py"),
}

# EQRN subpackage source files
eqrn_src_files = {
    "eqrn/__init__.py":      read_file(f"{BASE}/src/insurance_quantile/eqrn/__init__.py"),
    "eqrn/gpd.py":           read_file(f"{BASE}/src/insurance_quantile/eqrn/gpd.py"),
    "eqrn/network.py":       read_file(f"{BASE}/src/insurance_quantile/eqrn/network.py"),
    "eqrn/intermediate.py":  read_file(f"{BASE}/src/insurance_quantile/eqrn/intermediate.py"),
    "eqrn/model.py":         read_file(f"{BASE}/src/insurance_quantile/eqrn/model.py"),
    "eqrn/diagnostics.py":   read_file(f"{BASE}/src/insurance_quantile/eqrn/diagnostics.py"),
}

# Original test files
test_files = {
    "conftest.py":          read_file(f"{BASE}/tests/conftest.py"),
    "test_model.py":        read_file(f"{BASE}/tests/test_model.py"),
    "test_calibration.py":  read_file(f"{BASE}/tests/test_calibration.py"),
    "test_tvar.py":         read_file(f"{BASE}/tests/test_tvar.py"),
    "test_loading.py":      read_file(f"{BASE}/tests/test_loading.py"),
    "test_exceedance.py":   read_file(f"{BASE}/tests/test_exceedance.py"),
    "test_types.py":        read_file(f"{BASE}/tests/test_types.py"),
}

# EQRN test files
eqrn_test_files = {
    "eqrn/__init__.py":          "",
    "eqrn/conftest.py":          read_file(f"{BASE}/tests/eqrn/conftest.py"),
    "eqrn/test_gpd.py":          read_file(f"{BASE}/tests/eqrn/test_gpd.py"),
    "eqrn/test_network.py":      read_file(f"{BASE}/tests/eqrn/test_network.py"),
    "eqrn/test_intermediate.py": read_file(f"{BASE}/tests/eqrn/test_intermediate.py"),
    "eqrn/test_model.py":        read_file(f"{BASE}/tests/eqrn/test_model.py"),
    "eqrn/test_diagnostics.py":  read_file(f"{BASE}/tests/eqrn/test_diagnostics.py"),
}

all_src = {**src_files, **eqrn_src_files}
all_tests = {**test_files, **eqrn_test_files}

files_json = json.dumps({
    "src": all_src,
    "tests": all_tests,
})

pyproject_content = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "insurance-quantile"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "polars>=0.20",
    "catboost>=1.2",
    "scikit-learn>=1.3",
    "torch>=2.0",
    "lightgbm>=4.0",
    "scipy>=1.10",
    "pandas>=2.0",
    "matplotlib>=3.6",
]

[tool.hatch.build.targets.wheel]
packages = ["src/insurance_quantile"]
"""
pyproject_json = json.dumps(pyproject_content)

# ---------------------------------------------------------------------------
# Build notebook source
# ---------------------------------------------------------------------------
NOTEBOOK_SOURCE = f"""# Databricks notebook source
# MAGIC %pip install polars>=0.20 numpy>=1.24 catboost>=1.2 scikit-learn>=1.3 torch>=2.0 lightgbm>=4.0 scipy>=1.10 pandas>=2.0 matplotlib>=3.6 pytest>=7.0 hatchling --quiet

# COMMAND ----------

import json, os, sys, uuid, subprocess

pkg_id = uuid.uuid4().hex[:8]
pkg_dir = f"/tmp/insurance_quantile_{{pkg_id}}"
src_dir = f"{{pkg_dir}}/src/insurance_quantile"
eqrn_src_dir = f"{{src_dir}}/eqrn"
tests_dir = f"{{pkg_dir}}/tests"
eqrn_tests_dir = f"{{tests_dir}}/eqrn"

for d in [src_dir, eqrn_src_dir, tests_dir, eqrn_tests_dir]:
    os.makedirs(d, exist_ok=True)

FILES_JSON = {files_json!r}
PYPROJECT_JSON = {pyproject_json!r}

data = json.loads(FILES_JSON)
pyproject_src = json.loads(PYPROJECT_JSON)

# Write source files
for name, content in data["src"].items():
    path = f"{{src_dir}}/{{name}}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)

# Write test files
for name, content in data["tests"].items():
    path = f"{{tests_dir}}/{{name}}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)

with open(f"{{pkg_dir}}/pyproject.toml", "w") as fh:
    fh.write(pyproject_src)

n_src = len(data["src"])
n_tests = len(data["tests"])
print(f"Written {{n_src}} source files and {{n_tests}} test files to {{pkg_dir}}")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", pkg_dir, "--quiet"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("Install error:", r.stderr[:2000])
else:
    print("insurance-quantile v0.2.0 installed from", pkg_dir)

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short",
     "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=pkg_dir
)

print(r.stdout[-12000:])
if r.stderr:
    print("STDERR:", r.stderr[-2000:])

if r.returncode == 0:
    print("\\n=== ALL TESTS PASSED ===")
    try:
        dbutils.notebook.exit("ALL TESTS PASSED")
    except NameError:
        pass
else:
    msg = f"TESTS FAILED (exit {{r.returncode}})"
    print(f"\\n=== {{msg}} ===")
    try:
        dbutils.notebook.exit(msg)
    except NameError:
        pass
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def api_call(method: str, endpoint: str, body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(
        f"{api_base}/{endpoint}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"API {method} {endpoint} failed {e.code}: {err_body}")


# ---------------------------------------------------------------------------
# Ensure workspace folder exists
# ---------------------------------------------------------------------------
print(f"Creating workspace folder {WORKSPACE_FOLDER} ...")
try:
    api_call("POST", "api/2.0/workspace/mkdirs", {"path": WORKSPACE_FOLDER})
    print("Folder ready.")
except RuntimeError as exc:
    print(f"mkdirs note: {exc}")

# ---------------------------------------------------------------------------
# Upload notebook
# ---------------------------------------------------------------------------
print(f"Uploading notebook to {NOTEBOOK_PATH} ...")
notebook_b64 = base64.b64encode(NOTEBOOK_SOURCE.encode("utf-8")).decode("ascii")

api_call("POST", "api/2.0/workspace/import", {
    "path": NOTEBOOK_PATH,
    "format": "SOURCE",
    "language": "PYTHON",
    "content": notebook_b64,
    "overwrite": True,
})
print("Upload OK")

# ---------------------------------------------------------------------------
# Submit serverless run
# ---------------------------------------------------------------------------
print(f"Submitting serverless run {RUN_ID} ...")
submit_body = {
    "run_name": f"insurance-quantile-pytest-v0.2.0-{RUN_ID}",
    "tasks": [
        {
            "task_key": "pytest",
            "notebook_task": {
                "notebook_path": NOTEBOOK_PATH,
                "source": "WORKSPACE",
            },
        }
    ],
}
result = api_call("POST", "api/2.1/jobs/runs/submit", submit_body)
run_id = result["run_id"]
print(f"Run submitted: run_id={run_id}")

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------
print("Polling ...")
lc, rs = "PENDING", "-"
for i in range(180):
    time.sleep(15)
    run_state = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    lc = run_state.get("state", {}).get("life_cycle_state", "UNKNOWN")
    rs = run_state.get("state", {}).get("result_state", "-")
    print(f"  [{i * 15}s] {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print(f"\nFinal state: {lc} / {rs}")

# ---------------------------------------------------------------------------
# Fetch output
# ---------------------------------------------------------------------------
try:
    output = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={run_id}")
    notebook_result = output.get("notebook_output", {}).get("result", "")
    error = output.get("error", "")
    error_trace = output.get("error_trace", "")
    logs = output.get("logs", "")

    if notebook_result:
        print(f"\nExit value: {notebook_result}")
    if error:
        print(f"Error: {error}")
    if error_trace:
        print(f"Trace:\n{error_trace[:3000]}")
    if logs:
        print(f"\nLogs:\n{logs[-12000:]}")
except Exception as e:
    print(f"Could not fetch output: {e}")

if rs == "SUCCESS":
    print("\n=== PASS: All tests completed on Databricks. ===")
    sys.exit(0)
else:
    print(f"\n=== FAIL: Run ended with state {rs}. ===")
    sys.exit(1)
