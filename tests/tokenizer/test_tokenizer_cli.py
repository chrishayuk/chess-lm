import json
import shutil
import subprocess
import sys

import pytest

# Skip gracefully if python-chess or the CLI isn't available in CI
pytestmark = pytest.mark.skipif(shutil.which(sys.executable) is None, reason="Python executable not found")


def run_cli(*args):
    """Run the tokenizer CLI and return (rc, stdout, stderr)."""
    # Try to find chess-tokenizer in PATH first (for virtual env)
    chess_tokenizer = shutil.which("chess-tokenizer")

    if chess_tokenizer is None:
        # Fallback to running the module directly
        proc = subprocess.run(
            [sys.executable, "-m", "chess_lm.tokenizer.cli", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        # Use the installed command
        proc = subprocess.run(
            [chess_tokenizer, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def test_info_sizes_consistent():
    rc, out, err = run_cli("info")
    assert rc == 0, f"non-zero exit: {rc}, stderr: {err}"
    # Extract the integer values from the lines
    mv = st = tot = None
    for line in out.splitlines():
        if line.startswith("Move vocab size"):
            mv = int(line.split(":")[1].strip())
        if line.startswith("State token count"):
            st = int(line.split(":")[1].strip())
        if line.startswith("Total vocab size"):
            tot = int(line.split(":")[1].strip().split()[0])
    assert isinstance(mv, int) and mv > 1000
    assert isinstance(st, int) and st > 0
    assert isinstance(tot, int) and tot == mv + st


def test_encode_seq_json_roundtrip():
    # Simple opening: e2e4 e7e5
    rc, out, err = run_cli("encode-seq", "--seq", "e2e4 e7e5", "--json")
    assert rc == 0, f"encode-seq failed: {err}"
    payload = json.loads(out)
    tokens = payload["tokens"]
    moves = payload["moves"]
    # Expect [STATE, MOVE, STATE, MOVE]
    assert len(tokens) == 4
    # Moves decoded should match exactly
    assert moves == ["e2e4", "e7e5"]


def test_decode_moves_ignores_state_tokens():
    # Ask CLI for move vocab size so we can synthesize a STATE token id
    rc, out, err = run_cli("info")
    assert rc == 0
    mv = None
    for line in out.splitlines():
        if line.startswith("Move vocab size"):
            mv = int(line.split(":")[1].strip())
            break
    assert mv is not None

    # Provide a fake list: [STATE(=mv), move_id_of_e2e4]
    # We need the real move id for e2e4: get it via encode-seq
    rc, out, err = run_cli("encode-seq", "--seq", "e2e4", "--json")
    assert rc == 0
    payload = json.loads(out)
    tokens = payload["tokens"]
    # tokens[1] is the MOVE id for e2e4
    e2e4_id = tokens[1]

    combo = f"{mv} {e2e4_id}"
    rc, out, err = run_cli("decode-moves", "--ids", combo, "--json")
    assert rc == 0, f"decode-moves failed: {err}"
    decoded = json.loads(out)["decoded_ucis"]
    # STATE should be ignored; only e2e4 should appear
    assert decoded == ["e2e4"]
