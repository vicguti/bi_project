# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #### Test notebooks/configuration/functons.py
# MAGIC
# MAGIC This notebook can be used to run unit tests within the Databricks workspace.

# COMMAND ----------

# MAGIC %pip install pytest

# COMMAND ----------

import os
import sys
from os.path import dirname, join, realpath

import pytest

# Get the repo's root directory name.
repo_root = dirname(realpath(__name__))

# # Prepare to run pytest from the repo.
os.chdir(join(repo_root))
print(os.getcwd())

# Skip writing pyc files on a readonly filesystem.
sys.dont_write_bytecode = True

# Run pytest.
retcode = pytest.main(["test_functions.py"])

# Fail the cell execution if there are any test failures.
assert retcode == 0, "The pytest invocation failed. See the log for details."
