# SPDX-License-Identifier: Apache-2.0
"""Run a PhysicsNeMo Sym executable with exact-step asynchronous field archives.

The launcher installs hooks on the legacy Trainer recording events. It reads
freshly written files synchronously, before training can overwrite them, then
writes an immutable, atomically published snapshot in a background worker.
It does not alter the optimizer, model, loss, recording frequencies or sampler.
"""
import argparse
import hashlib
import json
import re
import runpy
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("constraints", "inferencers")
EXTENSIONS = {".vtp", ".vtu", ".vtk", ".vti", ".vts", ".vtr", ".npz", ".npy", ".csv"}


def inventory(directory):
    """Identify supported result files and their current write signatures."""
    if not directory.is_dir():
        return {}
    entries = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            st = p.stat()
            entries[p.relative_to(directory)] = (
                st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
            )
    return entries


def read_recording(directory, before):
    """Capture only files written during the current recording event."""
    after = inventory(directory)
    fresh = {p: sig for p, sig in after.items() if before.get(p) != sig}
    if not fresh:
        raise RuntimeError(f"No fresh result files in {directory}")
    result = {}
    for rel, sig in fresh.items():
        data = (directory / rel).read_bytes()
        if not data or len(data) != sig[1]:
            raise RuntimeError(f"Empty or incomplete result file: {directory / rel}")
        result[rel.as_posix()] = data
    return result


def publish(root, step, run_dir, records, config_text):
    """Write a complete snapshot, then expose its final directory atomically."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step_{step:07d}"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {target}")
    stage = Path(tempfile.mkdtemp(prefix=f".step_{step:07d}_", dir=root))
    manifest = {
        "step": step,
        "source_network_dir": str(run_dir),
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }
    try:
        for kind in KINDS:
            for relative, content in records[kind].items():
                path = stage / kind / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                manifest["files"][f"{kind}/{relative}"] = {
                    "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()
                }
        (stage / "config.yaml").write_text(config_text, encoding="utf-8")
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"[archive] step {step}: {target} ({len(manifest['files'])} files)", flush=True)
    return target


def step_from_trainer(trainer):
    """Constraint recording has no step argument in the legacy Trainer API."""
    match = re.search(r"\[step:\s*(\d+)\]", str(trainer.step_str))
    if not match:
        raise RuntimeError("Cannot identify PhysicsNeMo constraint recording step.")
    return int(match.group(1))


class ArchiveSession:
    def __init__(self, interval, archive_dir=None):
        if interval <= 0:
            raise ValueError("Archive interval must be positive.")
        self.interval = interval
        self.archive_dir = archive_dir
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="field_archive")
        self.future = None
        self.pending = None
        self.submitted = 0
        self.skipped = 0
        self.checked = False

    def due(self, step):
        return step % self.interval == 0

    def validate(self, trainer):
        if self.checked:
            return
        if trainer.manager.distributed:
            raise RuntimeError("Archive launcher supports single-process training only.")
        training = trainer.cfg.training
        for name in ("rec_constraint_freq", "rec_inference_freq"):
            freq = int(getattr(training, name))
            if freq <= 0 or self.interval % freq != 0:
                raise ValueError(
                    f"Archive interval {self.interval} must be a multiple of {name}={freq}."
                )
        self.checked = True

    def drain(self):
        if self.future is not None:
            self.future.result()
            self.future = None

    def close(self):
        try:
            self.drain()
        finally:
            self.worker.shutdown(wait=True)

    def install(self, trainer_class, config_serializer):
        """Install hooks and return a function that restores the original API."""
        original_constraints = trainer_class._record_constraints
        original_inferencers = trainer_class._record_inferencers
        session = self

        def record_constraints(trainer, *args, **kwargs):
            session.validate(trainer)
            step = step_from_trainer(trainer)
            if not session.due(step):
                return original_constraints(trainer, *args, **kwargs)
            session.drain()
            if session.pending is not None:
                raise RuntimeError("A previous archive has no matching inference recording.")
            run_dir = Path(trainer.network_dir).resolve()
            root = session.archive_dir or run_dir / "training_history"
            target = root / f"step_{step:07d}"
            if target.exists():
                # A resumed run may repeat a saved step. Never overwrite history.
                manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
                if manifest["step"] != step or manifest["source_network_dir"] != str(run_dir):
                    raise RuntimeError(f"Archive belongs to a different run: {target}")
                session.pending = (step, run_dir, None)
                return original_constraints(trainer, *args, **kwargs)
            directory = run_dir / "constraints"
            before = inventory(directory)
            result = original_constraints(trainer, *args, **kwargs)
            session.pending = (step, run_dir, read_recording(directory, before))
            return result

        def record_inferencers(trainer, step, *args, **kwargs):
            step = int(step)
            session.validate(trainer)
            if not session.due(step):
                return original_inferencers(trainer, step, *args, **kwargs)
            run_dir = Path(trainer.network_dir).resolve()
            pending = session.pending
            if pending is None or pending[:2] != (step, run_dir):
                # Some executables call eval() after solve(), which records only
                # inference. Do not mistake that evaluation for a training snapshot.
                result = original_inferencers(trainer, step, *args, **kwargs)
                print(f"[archive] step {step}: no matching constraint event; no snapshot created.", flush=True)
                return result
            session.pending = None
            if pending[2] is None:
                session.skipped += 1
                print(f"[archive] step {step}: existing snapshot retained.", flush=True)
                return original_inferencers(trainer, step, *args, **kwargs)
            directory = run_dir / "inferencers"
            before = inventory(directory)
            result = original_inferencers(trainer, step, *args, **kwargs)
            records = {"constraints": pending[2],
                       "inferencers": read_recording(directory, before)}
            root = session.archive_dir or run_dir / "training_history"
            # All mutable output bytes have been captured before returning to training.
            # The worker owns only immutable bytes and absolute paths.
            session.future = session.worker.submit(
                publish, root, step, run_dir, records, config_serializer(trainer.cfg)
            )
            session.submitted += 1
            return result

        trainer_class._record_constraints = record_constraints
        trainer_class._record_inferencers = record_inferencers

        def restore():
            trainer_class._record_constraints = original_constraints
            trainer_class._record_inferencers = original_inferencers

        return restore


def self_test():
    """Exercise the exact-step hooks without importing PhysicsNeMo or CUDA."""
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        for kind in KINDS:
            (run_dir / kind).mkdir()
        class FakeTrainer:
            def __init__(self):
                self.network_dir = str(run_dir)
                self.manager = SimpleNamespace(distributed=False)
                self.cfg = SimpleNamespace(training=SimpleNamespace(
                    rec_constraint_freq=1000, rec_inference_freq=1000))
            def _record_constraints(self):
                (run_dir / "constraints" / "airfoil.vtp").write_text(self.step_str)
            def _record_inferencers(self, step):
                (run_dir / "inferencers" / "boundary_1.vtp").write_text(str(step))

        session = ArchiveSession(50000)
        restore = session.install(FakeTrainer, lambda cfg: "test: true\n")
        try:
            trainer = FakeTrainer()
            for step in (0, 1000, 49000, 50000, 51000, 100000, 101000):
                trainer.step_str = f"[step: {step:10d}]"
                trainer._record_constraints()
                trainer._record_inferencers(step)
            session.drain()
            # A subsequent inference-only evaluation must not overwrite history.
            trainer._record_inferencers(100000)
            root = run_dir / "training_history"
            assert sorted(p.name for p in root.iterdir()) == [
                "step_0000000", "step_0050000", "step_0100000"]
            for step in (0, 50000, 100000):
                folder = root / f"step_{step:07d}"
                manifest = json.loads((folder / "manifest.json").read_text())
                assert manifest["step"] == step
                assert (folder / "inferencers/boundary_1.vtp").read_text() == str(step)
                assert step_from_trainer(SimpleNamespace(
                    step_str=(folder / "constraints/airfoil.vtp").read_text())) == step
                for rel, info in manifest["files"].items():
                    assert hashlib.sha256((folder / rel).read_bytes()).hexdigest() == info["sha256"]
            assert session.submitted == 3
        finally:
            restore()
            session.close()
    print("PASS: archive cadence, matching events, immutable contents and hashes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=50000)
    parser.add_argument("--archive-dir", type=Path, default=None,
                        help="Absolute or current-directory-relative snapshot root. "
                             "Default: actual PhysicsNeMo network_dir/training_history.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("training", nargs=argparse.REMAINDER,
                        help="After --, provide the training Python file and Hydra overrides.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    command = args.training
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Provide a training executable after --.")
    script = Path(command[0]).resolve(strict=True)
    if script.suffix != ".py":
        parser.error("Pass a Python file, not the python executable.")
    archive_dir = args.archive_dir.resolve() if args.archive_dir else None

    from omegaconf import OmegaConf
    from physicsnemo.sym.trainer import Trainer

    session = ArchiveSession(args.interval, archive_dir)
    restore = session.install(Trainer, OmegaConf.to_yaml)
    old_argv = sys.argv[:]
    sys.argv = [str(script), *command[1:]]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        restore()
        sys.argv = old_argv
        sys.path.remove(str(script.parent))
        session.close()
    if session.pending is not None:
        raise RuntimeError("Training ended without a matching inference recording.")
    if session.submitted + session.skipped == 0:
        raise RuntimeError("No complete training snapshot was produced.")
    print(f"[archive] finished: {session.submitted} new, {session.skipped} retained.", flush=True)


if __name__ == "__main__":
    main()
