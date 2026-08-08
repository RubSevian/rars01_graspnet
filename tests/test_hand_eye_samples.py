import numpy as np

from rars01_graspnet.hand_eye import append_sample, load_samples


def test_samples_append_and_load(tmp_path):
    path = tmp_path / "samples.npz"
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[0, 3] = 0.2
    assert append_sample(path, T1, T2, np.arange(7.0)) == 1
    assert append_sample(path, T2, T1, np.arange(7.0) + 1) == 2
    loaded = load_samples(path)
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded.T_tcp_base[1], T2)
    np.testing.assert_allclose(loaded.T_marker_camera[0], T2)
    np.testing.assert_allclose(loaded.joints[1], np.arange(6.0) + 1)
