"""Tests for the --save-defaults path construction in deploy_bf_weights.

These are pure-path tests: nothing here touches /data/casm, ssh, or scp.

The paths asserted below are the ones bfcorr actually reads on restart, from
fourier-space/sources/casm/backend/ovro_64ant_512beam/medusa_antenna.cfg:

    BFCORR_DEFAULT_BF_WEIGHTS     /data/casm/default_weights_64ant_512beam/direct.dada
    BFCORR_DEFAULT_INCOH_WEIGHTS  /data/casm/default_weights_64ant_512beam/incoh.dada

with ".{stream_id}" appended by casm_bfcorr.py.
"""

import pytest

from bf_weights_generator import deploy_bf_weights as deploy


DEFAULT_DIR = "/data/casm/default_weights_64ant_512beam"


def test_cb_default_paths():
    for stream in range(6):
        assert deploy.default_dest_path("cb", stream) == (
            f"{DEFAULT_DIR}/direct.dada.{stream}"
        )


def test_ib_defaults_go_to_incoh_dada_in_the_cb_directory():
    # Regression: IB defaults used to be written as
    # /data/casm/default_ib_weights_64ant/direct_ib.dada.N, which nothing reads.
    for stream in range(6):
        dst = deploy.default_dest_path("ib", stream)
        assert dst == f"{DEFAULT_DIR}/incoh.dada.{stream}"
        assert "direct_ib" not in dst
        assert "default_ib_weights_64ant" not in dst


def test_local_staged_names():
    assert deploy.local_staged_name("cb", 3) == "direct.dada.3"
    assert deploy.local_staged_name("ib", 3) == "direct_ib.dada.3"


def test_copy_plan_maps_streams_to_hosts():
    plan = deploy.default_copy_plan("ib", range(6), "/tmp/stage")
    hosts = [host for host, _, _ in plan]
    assert hosts == ["casm-corr1"] * 3 + ["casm-corr2"] * 3
    assert plan[0] == (
        "casm-corr1",
        "/tmp/stage/direct_ib.dada.0",
        f"{DEFAULT_DIR}/incoh.dada.0",
    )
    assert plan[5] == (
        "casm-corr2",
        "/tmp/stage/direct_ib.dada.5",
        f"{DEFAULT_DIR}/incoh.dada.5",
    )


def test_copy_plan_honours_a_stream_subset():
    plan = deploy.default_copy_plan("cb", [4], "/stage")
    assert plan == [
        ("casm-corr2", "/stage/direct.dada.4", f"{DEFAULT_DIR}/direct.dada.4")
    ]


@pytest.mark.parametrize("bad", [6, -1, "0", None])
def test_bad_stream_rejected(bad):
    with pytest.raises(ValueError):
        deploy.default_dest_path("cb", bad)


def test_bad_kind_rejected():
    with pytest.raises(ValueError):
        deploy.default_dest_path("incoherent", 0)


def test_save_defaults_raises_when_staged_file_is_missing(tmp_path):
    # Empty staging dir: must raise, not print-and-continue.
    with pytest.raises(RuntimeError, match="does not exist"):
        deploy.save_defaults("ib", [0], str(tmp_path))


def test_save_defaults_dry_run_touches_nothing(tmp_path, capsys):
    deploy.save_defaults("ib", range(6), str(tmp_path), dry_run=True)
    out = capsys.readouterr().out
    assert out.count("[DRY RUN]") == 6
    assert f"casm-corr1:{DEFAULT_DIR}/incoh.dada.0" in out
    assert f"casm-corr2:{DEFAULT_DIR}/incoh.dada.5" in out


def test_scp_failure_raises(monkeypatch, tmp_path):
    """A non-zero scp must abort the run, not print FAILED and carry on."""
    src = tmp_path / "direct_ib.dada.0"
    src.write_bytes(b"x" * 16)

    class FakeProc:
        returncode = 1
        stdout = b""
        stderr = b"scp: /data/casm/...: No such file or directory"

    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: FakeProc())
    with pytest.raises(RuntimeError, match="scp .* FAILED"):
        deploy.save_defaults("ib", [0], str(tmp_path))


def test_size_mismatch_raises(monkeypatch, tmp_path):
    """A short/truncated remote copy must abort even though scp exited 0."""
    src = tmp_path / "direct_ib.dada.0"
    src.write_bytes(b"x" * 16)

    class OkProc:
        returncode = 0
        stderr = b""

        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(argv, **kwargs):
        return OkProc(b"" if argv[0] == "scp" else b"8\n")

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="copy is incomplete"):
        deploy.save_defaults("ib", [0], str(tmp_path))
