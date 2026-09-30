import os

import pytest

# Headless rendering: prefer EGL (GPU), fall back to OSMesa (CPU) in CI.
os.environ.setdefault("MUJOCO_GL", os.environ.get("DLB_TEST_GL", "osmesa"))


@pytest.fixture(scope="session")
def mock_server():
    from dlb.backends.mock_server import serve

    srv = serve(port=8099, accept_images=True, background=True)
    yield "http://127.0.0.1:8099"
    srv.shutdown()


def render_available() -> bool:
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
        mujoco.Renderer(m, 16, 16)
        return True
    except Exception:  # noqa: BLE001
        return False
