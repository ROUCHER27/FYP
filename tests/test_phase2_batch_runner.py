from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import run_loss_scale_diagnostics
from run_phase2_robustness import run_command_with_live_log


def test_run_command_with_live_log_streams_stdout_and_persists_log(
    tmp_path: Path,
    capsys,
) -> None:
    log_file = tmp_path / "run.log"
    command = [
        sys.executable,
        "-c",
        "import os; print('unbuffered=' + os.environ.get('PYTHONUNBUFFERED', '')); "
        "print('epoch 1'); print('month 1995-01')",
    ]

    return_code = run_command_with_live_log(
        command,
        log_file,
        timeout_seconds=30,
        header_lines=["run_id=stream_test", "command=python -u -c ...", ""],
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "unbuffered=1" in captured.out
    assert "epoch 1" in captured.out
    assert "month 1995-01" in captured.out
    log_text = log_file.read_text()
    assert "run_id=stream_test" in log_text
    assert "unbuffered=1" in log_text
    assert "epoch 1" in log_text
    assert "month 1995-01" in log_text


def test_loss_scale_diagnostics_streams_child_output_to_cell_and_log(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    runner = tmp_path / "run_sanity_check_dummy.py"
    runner.write_text("print('diagnostic progress line')\n")
    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        output_root=str(tmp_path / "results"),
        checkpoint_root=str(tmp_path / "checkpoints"),
        log_root=str(tmp_path / "logs"),
        data_dir=".",
        pattern="*.csv",
        train_start="1990-01",
        train_end="1994-12",
        test_start="1995-01",
        test_months=1,
        best_config_path="best_hyperparameters.txt",
        max_epochs=1,
        batch_size=8,
        seed=42,
        max_weight="0.05",
        resume_mode="auto",
        timeout_seconds=30,
    )

    return_code = run_loss_scale_diagnostics.run_sanity(args, "dummy")

    captured = capsys.readouterr()
    assert return_code == 0
    assert "diagnostic progress line" in captured.out
    assert "diagnostic progress line" in (tmp_path / "logs" / "dummy.log").read_text()
