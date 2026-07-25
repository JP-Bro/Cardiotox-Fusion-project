"""
13_rigorous_tests.py -- Full QA regression + bug-verification suite.

Tests run against the live Flask server on localhost:5000.
All 5 known bugs have explicit regression tests.

Groups:
  A  Ground-truth regression (9 reference controls)
  B  SMILES canonicalization / permutation invariance
  C  Edge-case / boundary / security inputs
  D  Gene list health (disjoint, namespace, z-range)
  E  Cross-request determinism (same compound run 3x)
  F  Duplicate-output detection (unrelated pairs must differ)
"""
import sys
import os
import time
import requests
import numpy as np
from rdkit import Chem

BASE = "http://127.0.0.1:5000"
API  = f"{BASE}/predict"

# ---------- helpers -----------------------------------------------------------
def post(smiles: str, timeout: int = 15):
    return requests.post(API, json={"smiles": smiles}, timeout=timeout)

def wait_for_server(retries: int = 12, delay: float = 3.0):
    print("Waiting for server...", end="", flush=True)
    for _ in range(retries):
        try:
            r = requests.get(BASE, timeout=5)
            if r.status_code == 200:
                print(" ready.")
                return True
        except requests.exceptions.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(delay)
    print(" FAILED.")
    return False

# ---------- test runner -------------------------------------------------------
PASSED = []
FAILED = []

def check(name, condition, detail=""):
    sym = "PASS" if condition else "FAIL"
    print(f"  [{sym}] {name}" + (f" -- {detail}" if detail else ""))
    (PASSED if condition else FAILED).append(name)
    return condition

# ---------- compound library --------------------------------------------------
CONTROLS = [
    ("Aspirin",     "CC(=O)Oc1ccccc1C(=O)O",                                    "Low Risk"),
    ("Caffeine",    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                            "Low Risk"),
    ("Ibuprofen",   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                              "Low Risk"),
    ("Metformin",   "CN(C)C(=N)NC(=N)N",                                        "Low Risk"),
    ("Haloperidol", "C1CN(CCC1(c2ccc(Cl)cc2)O)CCCC(=O)c3cccc(F)c3",            "High Risk"),
    ("Verapamil",   "N#CC(C(C)C)(CCCN(C)CCc1ccc(OC)c(OC)c1)c2ccc(OC)c(OC)c2", "High Risk"),
    ("Terfenadine", "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4","High Risk"),
    ("Astemizole",  "COc1ccc(cc1)CCN2CCC(CC2)Nc3nc4ccccc4n3Cc5ccc(F)cc5",      "High Risk"),
    ("Dofetilide",  "CNS(=O)(=O)c1ccc(CCN(C)CCc2ccc(NS(C)(=O)=O)cc2)cc1",     "High Risk"),
]

DIVERSE = [
    ("Testosterone", "O=C1CC[C@H]2CC[C@@H]3[C@@H](CC[C@]4(O)CC(=O)CC[C@@H]34)[C@@H]2[C@@H]1"),
    ("Cholesterol",  "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@H]1CC=C3C[C@@H](O)CC[C@@]23C"),
    ("Nicotine",     "CN1CCC[C@H]1c2cccnc2"),
    ("Quinine",      "OC(C1CC2CCN1CC2C=C)c3ccnc4ccc(OC)cc34"),
    ("Aniline",      "Nc1ccccc1"),
]

# ---------- Group A: ground-truth regression ----------------------------------
def test_group_A():
    print("\n--- Group A: Ground-Truth Regression (9 controls) ---")
    for name, smi, expected in CONTROLS:
        r = post(smi)
        if r.status_code != 200:
            check(f"A/{name}", False, f"HTTP {r.status_code}")
            continue
        d   = r.json()
        got = d["risk_level"]
        check(f"A/{name}", got == expected,
              f"expected={expected!r} got={got!r} score={d['probability']:.1%}")

# ---------- Group B: permutation invariance -----------------------------------
def test_group_B():
    print("\n--- Group B: SMILES Permutation Invariance ---")
    pairs = [
        ("Aspirin",  "CC(=O)Oc1ccccc1C(=O)O",       "O=C(C)Oc1ccccc1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C","O=C1N(C)C(=O)N(C)c2ncn(C)c12"),
    ]
    for label, a, b in pairs:
        r1, r2 = post(a), post(b)
        if r1.status_code != 200 or r2.status_code != 200:
            check(f"B/{label}", False, "request failed")
            continue
        d1, d2 = r1.json(), r2.json()
        prob_ok = abs(d1["probability"] - d2["probability"]) < 1e-5
        gene_ok = (
            [g["name"] for g in d1["top_activated"]]  == [g["name"] for g in d2["top_activated"]]  and
            [g["name"] for g in d1["top_suppressed"]] == [g["name"] for g in d2["top_suppressed"]]
        )
        check(f"B/{label} score invariant", prob_ok,
              f"delta={abs(d1['probability']-d2['probability']):.2e}")
        check(f"B/{label} gene invariant",  gene_ok, "gene lists must match")

# ---------- Group C: edge-cases / security ------------------------------------
def test_group_C():
    print("\n--- Group C: Edge-Case / Security Inputs ---")
    cases = [
        ("Empty string",      "",                           400),
        ("Whitespace only",   "   ",                        400),
        ("Malformed ring",    "c1ccccc1(",                  400),
        ("SQL injection",     "CC(=O)O';DROP TABLE--",      400),
        ("Script injection",  "<script>alert(1)</script>",  400),
        ("Single atom C",     "C",                          200),
        ("Methane [CH4]",     "[CH4]",                      200),
    ]
    for label, smi, exp in cases:
        try:
            r = post(smi, timeout=10)
            check(f"C/{label}", r.status_code == exp,
                  f"expected {exp}, got {r.status_code}")
        except requests.exceptions.RequestException as e:
            check(f"C/{label}", False, str(e))

# ---------- Group D: gene list health -----------------------------------------
def test_group_D():
    print("\n--- Group D: Gene List Health ---")

    # D1 disjointness
    r = post("C1CN(CCC1(c2ccc(Cl)cc2)O)CCCC(=O)c3cccc(F)c3")
    if r.status_code == 200:
        d = r.json()
        up   = {g["name"] for g in d["top_activated"]}
        down = {g["name"] for g in d["top_suppressed"]}
        check("D/Gene list disjoint", len(up & down) == 0, f"overlap={up & down}")
    else:
        check("D/Gene list disjoint", False, f"HTTP {r.status_code}")

    # D2 namespace (no Gene_N placeholders; no Entrez+symbol mix)
    r = post("CC(=O)Oc1ccccc1C(=O)O")
    if r.status_code == 200:
        d  = r.json()
        gs = [g["name"] for g in d["top_activated"] + d["top_suppressed"]]
        legacy  = [g for g in gs if g.startswith("Gene_")]
        entrez  = [g for g in gs if g.startswith("Entrez:")]
        symbols = [g for g in gs if not g.startswith("Gene_") and not g.startswith("Entrez:")]
        check("D/No Gene_N placeholders", len(legacy) == 0, f"found: {legacy}")
        check("D/Namespace consistent",   not (entrez and symbols),
              f"mixed: entrez={entrez} symbols={symbols}")
    else:
        check("D/Namespace consistent", False, f"HTTP {r.status_code}")

    # D3 z-score sanity
    r = post("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    if r.status_code == 200:
        d = r.json()
        all_z = ([g["zscore"] for g in d["top_activated"] + d["top_suppressed"]] +
                 list(d["pathways"].values()))
        mx = max(abs(v) for v in all_z)
        check("D/Z-score range sane (< 15)", mx < 15, f"max|z|={mx:.2f}")
    else:
        check("D/Z-score range sane (< 15)", False, f"HTTP {r.status_code}")

# ---------- Group E: determinism ----------------------------------------------
def test_group_E():
    print("\n--- Group E: Determinism -- Same Compound x3 ---")
    cases = [
        ("Aspirin",     "CC(=O)Oc1ccccc1C(=O)O"),
        ("Haloperidol", "C1CN(CCC1(c2ccc(Cl)cc2)O)CCCC(=O)c3cccc(F)c3"),
        ("Nicotine",    "CN1CCC[C@H]1c2cccnc2"),
    ]
    for name, smi in cases:
        rs = [post(smi) for _ in range(3)]
        ds = [r.json() for r in rs if r.status_code == 200]
        if len(ds) < 3:
            check(f"E/{name} determinism", False, "request failed")
            continue
        p0   = ds[0]["probability"]
        up0  = [g["name"] for g in ds[0]["top_activated"]]
        dn0  = [g["name"] for g in ds[0]["top_suppressed"]]
        ok   = all(
            abs(d["probability"] - p0) < 1e-5 and
            [g["name"] for g in d["top_activated"]]  == up0 and
            [g["name"] for g in d["top_suppressed"]] == dn0
            for d in ds[1:]
        )
        check(f"E/{name} determinism", ok, f"probs={[d['probability'] for d in ds]}")

# ---------- Group F: distinct compounds -> distinct gene profiles --------------
def test_group_F():
    print("\n--- Group F: Distinct-Compound Gene Outputs Must Differ ---")
    outputs = {}
    for name, smi in DIVERSE:
        r = post(smi)
        if r.status_code == 200:
            d = r.json()
            outputs[name] = (
                tuple(g["name"] for g in d["top_activated"]),
                tuple(g["name"] for g in d["top_suppressed"]),
            )

    names = list(outputs.keys())
    collisions = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            same = outputs[names[i]] == outputs[names[j]]
            if same:
                collisions += 1
            check(f"F/{names[i]} vs {names[j]}", not same,
                  "IDENTICAL gene lists -- mode-collapse" if same else "")
    if collisions == 0:
        print("  All diverse compound pairs produce distinct gene profiles.")

# ---------- main --------------------------------------------------------------
def main():
    print("=" * 60)
    print("CARDIOTOX-FUSION -- FULL QA REGRESSION SUITE")
    print("=" * 60)

    if not wait_for_server():
        print("ERROR: Server not reachable. Run:  python app.py")
        sys.exit(1)

    test_group_A()
    test_group_B()
    test_group_C()
    test_group_D()
    test_group_E()
    test_group_F()

    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASSED)}/{total} passed")
    if FAILED:
        print("FAILED:")
        for f in FAILED:
            print(f"  x {f}")
    else:
        print("All tests passed.")
    print("=" * 60)

    # Write report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "qa_test_report.md",
    )
    rows = ("\n".join(f"| {n} | PASS |" for n in PASSED) + "\n" +
            "\n".join(f"| {n} | **FAIL** |" for n in FAILED))
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"# QA Report\n\nPassed {len(PASSED)}/{total}\n\n"
                 f"| Test | Status |\n|---|---|\n{rows}\n")
    print(f"Report saved: {report_path}")
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
