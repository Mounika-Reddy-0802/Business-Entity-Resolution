"""Upload bookkeeping and the final package (GIT_RULES.md §5, problem statement "Final Submission
Package"). Both commands refuse to run on the synthetic stand-in data.

    python -m src.common.release archive <n> [--lb <score>] [--from data/submissions/subN]
        output/matching_results.tsv -> benchmarks/raw/submission_<n>.tsv.gz, plus a row in
        benchmarks/experiments.md marked uploaded. Run right after uploading submission n.

    python -m src.common.release package <team_name> [--from data/submissions/subN]
        <team_name>_submission.zip with
          output/{matching_results.tsv, candidate_pairs.tsv}
          code/business_entity_resolution/{src/, scripts/, README.md, requirements.txt}
          Documentation_template.md
"""
import gzip
import shutil
import subprocess
import sys
import zipfile

from .evaluate import git_commit, log_run
from .io_utils import OUTPUT, RAW, ROOT, data_is_synthetic

CODE_DIRS = ("src", "scripts")
NOT_PACKAGED = {"setup_git.sh"}           # team git tooling, not part of the pipeline
CODE_FILES = ("README.md", "requirements.txt")


def packaged(f):
    """True for source files that belong in the package (no caches, no git tooling)."""
    return f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc" \
        and f.name not in NOT_PACKAGED


def refuse_synthetic():
    if data_is_synthetic():
        sys.exit("refusing: data/raw/dataset holds the synthetic stand-in (docs/problems.md)")


def validate(out_dir=OUTPUT):
    """Run the organiser validator when present, and the local check; stop on failure."""
    args = ["--matching", str(out_dir / "matching_results.tsv"), "--candidate",
            str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(RAW / "test")]
    organiser = ROOT / "utils" / "validate_submission.py"
    cmds = ([[sys.executable, str(organiser), *args]] if organiser.exists() else []) + \
        [[sys.executable, "-m", "src.common.check_submission", *args]]
    for cmd in cmds:
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit("refusing: validation failed")


def archive(n, lb=None, out_dir=OUTPUT):
    """Keep an uploaded submission so it can be audited and regenerated."""
    refuse_synthetic()
    validate(out_dir)
    dest = ROOT / "benchmarks" / "raw" / f"submission_{n}.tsv.gz"
    if dest.exists():
        sys.exit(f"refusing: {dest.name} already exists")
    with open(out_dir / "matching_results.tsv", "rb") as src, gzip.open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    print(dest)
    print(log_run(f"submission_{n}", f"uploaded submission {n} (commit {git_commit()})",
                  {"uploaded": True, "lb": lb}))


def package(team, out_dir=OUTPUT):
    """Build <team>_submission.zip in the repo root with the organiser's layout; out_dir holds the
    two uploaded TSVs (default output/)."""
    refuse_synthetic()
    validate(out_dir)
    path = ROOT / f"{team}_submission.zip"
    code = "code/business_entity_resolution"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(out_dir / name, f"output/{name}")
        for d in CODE_DIRS:
            for f in sorted((ROOT / d).rglob("*")):
                if packaged(f):
                    z.write(f, f"{code}/{f.relative_to(ROOT).as_posix()}")
        for name in CODE_FILES:
            z.write(ROOT / name, f"{code}/{name}")
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    print(path, len(names), "files")
    for required in ("output/matching_results.tsv", "output/candidate_pairs.tsv",
                     f"{code}/README.md", f"{code}/requirements.txt", "Documentation_template.md"):
        assert required in names, required
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    out = ROOT / sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else OUTPUT
    if cmd == "archive" and len(sys.argv) > 2:
        lb = float(sys.argv[sys.argv.index("--lb") + 1]) if "--lb" in sys.argv else None
        archive(int(sys.argv[2]), lb, out)
    elif cmd == "package" and len(sys.argv) > 2:
        package(sys.argv[2], out)
    else:
        sys.exit(__doc__)
