import subprocess
from pathlib import Path
import h5py

EXPECTED_KEYS = {
    'tensor_a',
    'tensor_b',
    'scalar_data',
    'vector_data',
    'image_grayscale_data',
    'image_rgb_data',
    'video_frames_data',
    'simulation_data',
}

def test_generate_tensors_runs(tmp_path: Path):
    out_dir = tmp_path
    # Call the script using subprocess to mimic CLI usage
    result = subprocess.run(
        ["python", str(Path('scripts/generate_tensors.py')), str(out_dir), "--seed", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    output_file = out_dir / "demo_tensors.h5"
    assert output_file.is_file(), "Output file was not created"

    with h5py.File(output_file, 'r') as f:
        assert set(f.keys()) == EXPECTED_KEYS

