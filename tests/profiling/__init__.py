"""These tests generate profiling data that is used by profiling/run_benchmarking.py.

Each profiling test package provides a fixture with shared set of data with package
scope and then individual tests in modules. The tests in the `profiling` package are
excluded from standard pytest runs as `setup.cfg` includes `-m 'not profiling'`.
"""
