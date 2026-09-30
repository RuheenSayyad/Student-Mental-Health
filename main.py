"""Run the whole Student Mental Health ML project, lab by lab.

    python main.py            # runs Labs 3 -> 4 -> 5 -> 6
    python main.py --lab 5    # runs a single lab

Labs 4 and 6 need MLflow (see requirements.txt). Lab 6 needs the Lab 5 outputs.
"""
import argparse
import subprocess
import sys

LABS = {
    3: "pipelines/run_lab3_baseline.py",
    4: "pipelines/run_lab4_tracking.py",
    5: "pipelines/run_lab5_pipeline.py",
    6: "pipelines/run_lab6_registry.py",
}


def main():
    parser = argparse.ArgumentParser(description="Student Mental Health ML project runner")
    parser.add_argument("--lab", type=int, choices=sorted(LABS), help="Run only this lab (3-6)")
    args = parser.parse_args()

    labs = [args.lab] if args.lab else sorted(LABS)
    for lab in labs:
        print(f"\n########## LAB {lab} ##########")
        result = subprocess.run([sys.executable, LABS[lab]])
        if result.returncode != 0:
            print(f"[ERROR] Lab {lab} failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print("\n[SUCCESS] All requested labs completed.")


if __name__ == "__main__":
    main()
