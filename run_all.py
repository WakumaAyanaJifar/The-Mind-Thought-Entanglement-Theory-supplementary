"""Reproduce every number and every figure of the MTE paper (main text and supplement).

Usage:  python run_all.py            # everything (about 50 min on one core; the sheet simulation dominates)
        python run_all.py --quick    # skips the sheet simulation and the 8-parameter recovery study;
                                     # Fig. 4 is then drawn from the reference results in expected_output/

Each script runs with outputs/ as working directory; printed output goes to outputs/logs/<script>.txt.
Compare with expected_output/ (same file names).
"""
import subprocess, sys, time, pathlib, shutil
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "outputs"; LOG = OUT / "logs"; LOG.mkdir(parents=True, exist_ok=True)
SLOW = {"recovery7.py", "recovery_sheet.py", "snr_sensitivity.py", "power_p5.py"}
STEPS = [
    ("check1.py",            "Damped example (Sec. 3.3, Fig. 2b): resonances, kappa_pm, R_-, identity G1"),
    ("hankel_check.py",      "Two-component form vs numerical Hankel transform (Sec. 2.4)"),
    ("twofibre.py",          "Two-fibre competitor (Sec. 3.3, Suppl. S2)"),
    ("boundary.py",          "Boundary with cross term (Sec. 3.4, Fig. 3, Suppl. S4): exponents, ratio, factor 4, bias crossover"),
    ("boundary_window.py",   "Steady states, window budget, bias scaling (LOC ~ J, ROC ~ sqrt J), P2/P5 coexistence (Sec. 3.4, Suppl. S4, Table S4)"),
    ("nucleation2d.py",      "2D nucleation barrier constant and local logarithmic law (Sec. 3.4, Suppl. S4)"),
    ("potential_splitting.py","Lifting of vacuum degeneracy (Sec. 3.6, Fig. S4)"),
    ("fisher.py",            "Local identifiability (Sec. 3.8, Table S2)"),
    ("recovery7.py",         "Recovery of the 8 parameters on the k-space grid (Table S3)  [slow]"),
    ("recovery_sheet.py",    "Sheet simulation of Sec. 3.10 (Fig. 4c,d; coverage check)  [slow]"),
    ("global_ident.py",      "Global identifiability: multi-start search and profile (Sec. 3.8, Suppl. S6)"),
    ("snr_sensitivity.py",   "Required SNR when single parameters change (Sec. 3.10, Table S6)  [slow]"),
    ("anisotropy.py",        "Isotropic analysis on an anisotropic sheet (Sec. 4.2, Table S8)"),
    ("noisy_sheet.py",       "Langevin simulation of the boundary on a noisy sheet (Sec. 3.4, Table S5)"),
    ("power_p5.py",          "Power analysis for the exponent test E1 (Suppl. S5, Table S7)  [slow]"),
    ("static1d.py",          "d=1 static threshold state, survey, convergence, dynamics (Suppl. S3, Table S1)"),
    ("sextic_check.py",      "Sextic stabilization of the threshold state (Suppl. S3, Table S1)"),
    ("static2d.py",          "d=2 static threshold state (Suppl. S3, Fig. S3)"),
    ("static2d_conv.py",     "Grid convergence of the d=2 state (Table S1)"),
    ("envelope.py",          "Envelope ground states, width-amplitude slopes (Suppl. S3)"),
    ("make_figs.py",         "Figures 1, 3 and S3"),
    ("fig2_spectrum.py",     "Figure 2"),
    ("fig4_plot.py",         "Figure 4"),
    ("figS1_S4.py",          "Figures S1 and S4"),
    ("figS2_plot.py",        "Figure S2"),
]
quick = "--quick" in sys.argv
if quick:
    for f in ["recovery_sheet.npz"]:
        src = ROOT / "expected_output" / f
        if src.exists(): shutil.copy(src, OUT / f)
for script, what in STEPS:
    if quick and script in SLOW:
        print(f"-- skipping {script} (--quick)"); continue
    print(f"== {script}: {what}", flush=True)
    t = time.time()
    env = dict(**__import__('os').environ, PYTHONPATH=str(ROOT / "code"))
    res = subprocess.run([sys.executable, str(ROOT / "code" / script)], cwd=OUT,
                         capture_output=True, text=True, env=env)
    (LOG / (script[:-3] + ".txt")).write_text(res.stdout + res.stderr)
    print(res.stdout.rstrip())
    if res.returncode != 0:
        print(res.stderr); sys.exit(f"{script} failed")
    print(f"   done in {time.time()-t:.0f} s\n", flush=True)
print("All steps finished. Results in outputs/, logs in outputs/logs/.")
