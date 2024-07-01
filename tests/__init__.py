"""The tests module.

This file is required to make tests a module. For some reason, `mypy` will only respect
per-module options, such as suppressing typing checks in tests, for _modules_. Adding a
`[mypy-tests.*]` section in setup.cfg does nothing if tests is not a module.
"""
