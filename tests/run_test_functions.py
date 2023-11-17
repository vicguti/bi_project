# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #### Test notebooks/configuration/functons.py
# MAGIC
# MAGIC This notebook can be used to run unit tests within the Databricks workspace.
# MAGIC
# MAGIC
# MAGIC This notebook is in progress. All our helpers are Databrick notebooks but the unit test are in python files and are using imports from those Databricks notebooks, this does not work (this only works in our local computer).
# MAGIC
# MAGIC - Next steps:
# MAGIC   - Tranform our helpers into python files
# MAGIC   - Check CD pipeline to do not transform Python files to Databricks notebooks automatically

# COMMAND ----------

# MAGIC %pip install pytest

# COMMAND ----------

import os
import pytest
import sys



from os.path import join, dirname, realpath

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
