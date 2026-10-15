"""
04_baselines.py -- Pipeline step 4. Runs the canonical baseline benchmark in train_classical_baselines.py
(Morgan-FP Random Forest + L1000 Logistic Regression as the pre-specified primary pair, plus exploratory models).

This file previously held a separate 2-model implementation that tuned on validation AUC-PR (the wrong metric for a
majority-positive set) and used an unseeded bootstrap; it is now a thin wrapper so there is ONE source of truth.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_classical_baselines.py"), run_name="__main__")
