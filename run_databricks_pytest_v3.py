"""
Submit insurance-quantile pytest suite (v0.3.3, torch/lightgbm optional [eqrn] extra)
to Databricks serverless compute.

Runs both core tests and EQRN tests (torch is installed on Databricks).

Run from the project root:
    python run_databricks_pytest_v3.py
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
WORKSPACE_FOLDER = "/Workspace/insurance-quantile-v033"
NOTEBOOK_PATH = f"{WORKSPACE_FOLDER}/run_pytest_v033"

# ---------------------------------------------------------------------------
# Read source files
# ---------------------------------------------------------------------------
BASE = "/home/ralph/burning-cost/repos/insurance-quantile"


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


src_files = {
    "__init__.py":      read_file(f"{BASE}/src/insurance_quantile/__init__.py"),
    "_types.py":        read_file(f"{BASE}/src/insurance_quantile/_types.py"),
    "_model.py":        read_file(f"{BASE}/src/insurance_quantile/_model.py"),
    "_calibration.py":  read_file(f"{BASE}/src/insurance_quantile/_calibration.py"),
    "_tvar.py":         read_file(f"{BASE}/src/insurance_quantile/_tvar.py"),
    "_loading.py":      read_file(f"{BASE}/src/insurance_quantile/_loading.py"),
    "_exceedance.py":   read_file(f"{BASE}/src/insurance_quantile/_exceedance.py"),
    "_two_part.py":     read_file(f"{BASE}/src/insurance_quantile/_two_part.py"),
}

eqrn_src_files = {
    "eqrn/__init__.py":      read_file(f"{BASE}/src/insurance_quantile/eqrn/__init__.py"),
    "eqrn/gpd.py":           read_file(f"{BASE}/src/insurance_quantile/eqrn/gpd.py"),
    "eqrn/network.py":       read_file(f"{BASE}/src/insurance_quantile/eqrn/network.py"),
    "eqrn/intermediate.py":  read_file(f"{BASE}/src/insurance_quantile/eqrn/intermediate.py"),
    "eqrn/model.py":         read_file(f"{BASE}/src/insurance_quantile/eqrn/model.py"),
    "eqrn/diagnostics.py":   read_file(f"{BASE}/src/insurance_quantile/eqrn/diagnostics.py"),
}

test_files = {
    "conftest.py":          read_file(f"{BASE}/tests/conftest.py"),
    "test_model.py":        read_file(f"{BASE}/tests/test_model.py"),
    "test_calibration.py":  read_file(f"{BASE}/tests/test_calibration.py"),
    "test_tvar.py":         read_file(f"{BASE}/tests/test_tvar.py"),
    "test_loading.py":      read_file(f"{BASE}/tests/test_loading.py"),
    "test_exceedance.py":   read_file(f"{BASE}/tests/test_exceedance.py"),
    "test_types.py":        read_file(f"{BASE}/tests/test_types.py"),
    "test_two_part.py":     read_file(f"{BASE}/tests/test_two_part.py"),
}

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

# v0.3.3: torch and lightgbm are now optional [eqrn] extras, not hard deps.
# We still install them for CI because Databricks has torch available.
pyproject_content = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "insurance-quantile"
version = "0.3.3"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "polars>=1.0",
    "catboost>=1.2",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "pandas>=2.0",
    "matplotlib>=3.6",
]

[project.optional-dependencies]
eqrn = [
    "torch>=2.0",
    "lightgbm>=4.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/insurance_quantile"]
"""
pyproject_json = json.dumps(pyproject_content)

# ---------------------------------------------------------------------------
# Build notebook source
# Use a plain string template with __FILES_JSON__ and __PYPROJECT_JSON__ as
# placeholders to avoid f-string escaping issues with notebook Python code.
# ---------------------------------------------------------------------------
NOTEBOOK_TEMPLATE = r"""# Databricks notebook source
# MAGIC %pip install polars>=1.0 numpy>=1.24 catboost>=1.2 scikit-learn>=1.3 torch>=2.0 lightgbm>=4.0 scipy>=1.10 pandas>=2.0 matplotlib>=3.6 pytest>=7.0 hatchling --quiet

# COMMAND ----------

import json, os, sys, uuid, subprocess

pkg_id = uuid.uuid4().hex[:8]
pkg_dir = f"/tmp/insurance_quantile_{pkg_id}"
src_dir = f"{pkg_dir}/src/insurance_quantile"
eqrn_src_dir = f"{src_dir}/eqrn"
tests_dir = f"{pkg_dir}/tests"
eqrn_tests_dir = f"{tests_dir}/eqrn"

for d in [src_dir, eqrn_src_dir, tests_dir, eqrn_tests_dir]:
    os.makedirs(d, exist_ok=True)

FILES_JSON = __FILES_JSON__
PYPROJECT_JSON = __PYPROJECT_JSON__

data = json.loads(FILES_JSON)
pyproject_src = json.loads(PYPROJECT_JSON)

for name, content in data["src"].items():
    path = f"{src_dir}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)

for name, content in data["tests"].items():
    path = f"{tests_dir}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)

with open(f"{pkg_dir}/pyproject.toml", "w") as fh:
    fh.write(pyproject_src)

print(f"Written {len(data['src'])} source files, {len(data['tests'])} test files to {pkg_dir}")

# COMMAND ----------

# Verify the lazy-import machinery is in place
print("Smoke test: __getattr__ in __init__.py ->", "__getattr__" in open(f"{src_dir}/__init__.py").read())
print("Smoke test: torch NOT in hard deps ->", "torch" not in pyproject_src.split("[project.optional-dependencies]")[0])

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", pkg_dir, "--quiet"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("Install error:", r.stderr[:2000])
else:
    print("insurance-quantile v0.3.3 installed from", pkg_dir)

# COMMAND ----------

# Run core (non-EQRN) tests
print("=== Running core tests (no EQRN) ===")
r_core = subprocess.run(
    [sys.executable, "-m", "pytest", tests_dir,
     "--ignore", f"{tests_dir}/eqrn",
     "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=pkg_dir
)
print(r_core.stdout[-10000:])
if r_core.stderr:
    print("STDERR:", r_core.stderr[-1000:])

# COMMAND ----------

# Run EQRN tests (torch is installed on this cluster)
print("=== Running EQRN tests ===")
r_eqrn = subprocess.run(
    [sys.executable, "-m", "pytest", f"{tests_dir}/eqrn",
     "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=pkg_dir
)
print(r_eqrn.stdout[-10000:])
if r_eqrn.stderr:
    print("STDERR:", r_eqrn.stderr[-1000:])

# COMMAND ----------

total_pass = r_core.returncode == 0 and r_eqrn.returncode == 0
if total_pass:
    msg = "ALL TESTS PASSED (core + EQRN)"
    print(f"\n=== {msg} ===")
    try:
        dbutils.notebook.exit(msg)
    except NameError:
        pass
else:
    codes = f"core={r_core.returncode} eqrn={r_eqrn.returncode}"
    msg = f"TESTS FAILED ({codes})"
    print(f"\n=== {msg} ===")
    try:
        dbutils.notebook.exit(msg)
    except NameError:
        pass
"""

# Inject the data blobs
NOTEBOOK_SOURCE = NOTEBOOK_TEMPLATE.replace("__FILES_JSON__", repr(files_json)).replace(
    "__PYPROJECT_JSON__", repr(pyproject_json)
)

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
print("Notebook uploaded.")

# ---------------------------------------------------------------------------
# Submit job run
# ---------------------------------------------------------------------------
print("Submitting job run ...")
run_resp = api_call("POST", "api/2.1/jobs/runs/submit", {
    "run_name": f"insurance-quantile-v033-pytest-{RUN_ID}",
    "tasks": [{
        "task_key": "pytest",
        "notebook_task": {
            "notebook_path": NOTEBOOK_PATH,
            "source": "WORKSPACE",
        },

    }],
})
run_id = run_resp["run_id"]
print(f"Job run submitted: run_id={run_id}")
print(f"UI: {api_base}/#job/runs/{run_id}")

# ---------------------------------------------------------------------------
# Poll for completion
# ---------------------------------------------------------------------------
print("Polling for completion (timeout 20 min) ...")
deadline = time.time() + 1200
while time.time() < deadline:
    status = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    life = status["state"]["life_cycle_state"]
    result = status["state"].get("result_state", "")
    print(f"  [{time.strftime('%H:%M:%S')}] {life} / {result}")
    if life in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(30)

if result == "SUCCESS":
    output = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={run_id}")
    notebook_out = output.get("notebook_output", {}).get("result", "")
    print(f"\nNotebook exit message: {notebook_out}")
    print("\nRESULT: PASSED")
    sys.exit(0)
else:
    output = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={run_id}")
    notebook_out = output.get("notebook_output", {}).get("result", "")
    error = output.get("error", "")
    print(f"\nNotebook exit message: {notebook_out}")
    if error:
        print(f"Error: {error}")
    print(f"\nRESULT: FAILED ({result})")
    sys.exit(1)
