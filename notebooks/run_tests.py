# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-quantile: run full test suite
# MAGIC Uploads the library from the repo root and runs pytest.

# COMMAND ----------
import subprocess, sys

# Install the library and test dependencies
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "insurance-quantile==0.3.0",
     "scikit-learn>=1.3", "pytest"],
    check=True
)

# COMMAND ----------
# Run tests — output goes to stdout / notebook cell output
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--pyargs", "insurance_quantile",
     "-x", "-q", "--tb=short"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
if result.returncode != 0:
    raise RuntimeError("Tests failed — see output above")
