import os

from skp.environment import configure_gcp_gpu_environment, load_project_dotenv


def test_load_project_dotenv_does_not_override_exported_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TEST_DOTENV=from_file\n")
    monkeypatch.setenv("TEST_DOTENV", "exported")

    load_project_dotenv()

    assert os.environ["TEST_DOTENV"] == "exported"


def test_configure_gcp_gpu_environment_overrides_bad_network_defaults(monkeypatch):
    monkeypatch.setenv("NCCL_NET", "Plugin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("GCP_NCCL_SOCKET_HACK_DISABLE", raising=False)

    configure_gcp_gpu_environment()

    assert "LD_LIBRARY_PATH" not in os.environ
    assert os.environ["NCCL_NET"] == "Socket"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "ens"


def test_configure_gcp_gpu_environment_sets_socket_default(monkeypatch):
    monkeypatch.delenv("NCCL_NET", raising=False)
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)

    configure_gcp_gpu_environment()

    assert os.environ["NCCL_NET"] == "Socket"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "ens"


def test_configure_gcp_gpu_environment_respects_socket_interface_override(monkeypatch):
    monkeypatch.setenv("NCCL_NET", "Plugin")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth")

    configure_gcp_gpu_environment()

    assert os.environ["NCCL_NET"] == "Socket"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "eth"


def test_configure_gcp_gpu_environment_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GCP_NCCL_SOCKET_HACK_DISABLE", "true")
    monkeypatch.setenv("NCCL_NET", "Plugin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")

    configure_gcp_gpu_environment()

    assert os.environ["NCCL_NET"] == "Plugin"
    assert os.environ["LD_LIBRARY_PATH"] == "/usr/local/lib"
