"""Limita el paralelismo en las pruebas de Open3D."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
