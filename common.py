from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parent
CFAS_DIR = ROOT / "CFAs"
RESULTS_DIR = ROOT / "results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image_rgb(path: str | Path) -> np.ndarray:
    image = np.asarray(iio.imread(Path(path)))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=2)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    iio.imwrite(output_path, image)


def matlab_uint8(value: np.ndarray) -> np.ndarray:
    real_value = np.real(value)
    return np.clip(np.rint(real_value), 0, 255).astype(np.uint8)


def matlab_im2gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return matlab_uint8(gray)


def save_cfa(path: str | Path, cfa: dict[str, Any]) -> None:
    normalized = normalize_cfa(cfa)
    output_path = Path(path)
    ensure_dir(output_path.parent)
    savemat(output_path, {"cfa": normalized})


def save_cfa_by_name(cfa_name: str, cfa: dict[str, Any]) -> Path:
    output_path = CFAS_DIR / f"cfa_{cfa_name}.mat"
    save_cfa(output_path, cfa)
    return output_path


def load_cfa_by_name(cfa_name: str) -> dict[str, np.ndarray]:
    mat_path = CFAS_DIR / f"cfa_{cfa_name}.mat"
    if mat_path.exists():
        cfa = _load_cfa_from_mat(mat_path)
        if "M" in cfa and "pcfa" in cfa:
            return cfa

    matlab_mat_path = ROOT / "CFAs" / f"cfa_{cfa_name}.mat"
    if matlab_mat_path.exists():
        return _load_cfa_from_mat(matlab_mat_path)

    raise FileNotFoundError(f"Cannot find CFA asset for {cfa_name}.")


def normalize_cfa(cfa: dict[str, Any]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key, value in cfa.items():
        array = np.asarray(value)
        if key.endswith("_loc"):
            if array.size == 0:
                array = np.zeros((0, 2), dtype=np.float64)
            elif array.ndim == 1:
                array = array.reshape(1, -1)
        elif key.endswith("_isconj") or key.endswith("_coeff"):
            array = np.atleast_1d(array)
        normalized[key] = array
    return normalized


def _load_cfa_from_mat(path: Path) -> dict[str, np.ndarray]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    cfa_obj = data["cfa"]

    if isinstance(cfa_obj, dict):
        fields = {name: np.asarray(value) for name, value in cfa_obj.items()}
    elif hasattr(cfa_obj, "_fieldnames"):
        fields = {name: np.asarray(getattr(cfa_obj, name)) for name in cfa_obj._fieldnames}
    elif isinstance(cfa_obj, np.ndarray) and cfa_obj.dtype.names is not None:
        fields = {name: np.asarray(cfa_obj[name]).squeeze() for name in cfa_obj.dtype.names}
    else:
        fields = {
            name: np.asarray(getattr(cfa_obj, name))
            for name in dir(cfa_obj)
            if not name.startswith("_") and not callable(getattr(cfa_obj, name))
        }

    return normalize_cfa(fields)
