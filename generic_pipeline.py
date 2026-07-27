from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d
from tqdm.auto import tqdm

from common import ROOT, matlab_uint8, read_image_rgb


CAVE_RGBW_MAX_VALUE = 1023.0

# Noise sigma for RGB channels; W channel uses sigma / 3 (same as mosaick_with_noise).
LOW_LIGHT_NOISE_SIGMA = {
    "low_light": 5.0,
    "extreme_lowlight": 10.0,
}


@dataclass(frozen=True)
class DatasetSample:
    name: str
    original_image: np.ndarray
    input_image: np.ndarray


class ImageDataset:
    def __init__(self, dataset_name: str, samples: list[DatasetSample]) -> None:
        self.dataset_name = dataset_name
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    def output_file_name(self, index: int) -> str:
        return self.samples[index].name


def _chroma_indices(cfa: dict) -> list[int]:
    indices: list[int] = []
    for key in cfa:
        match = re.fullmatch(r"C(\d+)_loc", key)
        if match is None:
            continue
        index = int(match.group(1))
        if f"C{index}_isconj" not in cfa or f"C{index}_coeff" not in cfa:
            raise KeyError(f"Incomplete chroma specification for C{index}.")
        indices.append(index)
    if not indices:
        raise ValueError("No chroma components found in CFA.")
    return sorted(indices)


def infer_chroma_count(cfa: dict) -> int:
    return len(_chroma_indices(cfa))


def _chroma_specs(cfa: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    specs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index in _chroma_indices(cfa):
        specs.append((cfa[f"C{index}_loc"], cfa[f"C{index}_isconj"], cfa[f"C{index}_coeff"]))
    return specs


def dataset_file_name(dataset_name: str, index: int) -> str:
    if dataset_name == "kodak":
        return f"kodim{index:02d}.png"
    if dataset_name in {"low_light", "extreme_lowlight"}:
        return f"low{index:02d}.png"
    if dataset_name in {"cave_lowlight", "cave_extreme_lowlight", "cave_normal"}:
        raise ValueError(f"Use dataset.output_file_name() for scene-based dataset: {dataset_name}")
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _make_noisy_rgbw_input(image: np.ndarray, sigma: float) -> np.ndarray:
    """Build noisy RGBW input from clean RGB (same noise model as mosaick_with_noise).

    - RGB: independent Gaussian noise N(0, sigma^2)
    - W = (R+G+B)/3 from the clean image, then N(0, (sigma/3)^2)
    - No clipping (float64 HxWx4), matching demo_gehler_noise_new.py
    """
    img_float = np.asarray(image, dtype=np.float64)
    if img_float.ndim != 3 or img_float.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {img_float.shape}")

    noise_rgb = np.random.normal(0.0, sigma, img_float.shape)
    img_rgb_noisy = img_float + noise_rgb

    w_matrix = np.mean(img_float, axis=2)
    noise_w = np.random.normal(0.0, sigma / 3.0, w_matrix.shape)
    w_matrix_noisy = w_matrix + noise_w

    return np.dstack((img_rgb_noisy, w_matrix_noisy))


def _convert_to_uint8_image(image: np.ndarray, source_max_value: float | None = None) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    image = image.astype(np.float64)
    if source_max_value is not None:
        if source_max_value <= 0:
            raise ValueError(f"source_max_value must be positive, got {source_max_value}")
        scaled = image * (255.0 / source_max_value)
    elif np.max(image) <= 1.0:
        scaled = image * 255.0
    elif np.max(image) <= 255.0:
        scaled = image
    else:
        scaled = image * (255.0 / CAVE_RGBW_MAX_VALUE)
    return matlab_uint8(scaled)


def _load_cave_rgbw_mat(path: Path) -> np.ndarray:
    data = loadmat(path)
    if "rgbw" not in data:
        raise KeyError(f"Missing rgbw field in {path}")
    image = np.asarray(data["rgbw"])
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected H x W x 4 rgbw image in {path}, got {image.shape}")
    return image


def _build_generic_dataset(dataset_name: str) -> ImageDataset:
    if dataset_name == "kodak":
        dataset_dir = ROOT / "dataset" / "kodak"
        image_count = 24
        noise_sigma: float | None = None
    elif dataset_name in {"low_light", "extreme_lowlight"}:
        dataset_dir = ROOT / "dataset" / "low_light"
        image_count = 10
        noise_sigma = LOW_LIGHT_NOISE_SIGMA[dataset_name]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    samples = []
    for index in range(1, image_count + 1):
        file_name = dataset_file_name(dataset_name, index)
        image = read_image_rgb(dataset_dir / file_name)
        if noise_sigma is None:
            input_image = image
        else:
            input_image = _make_noisy_rgbw_input(image, noise_sigma)
        samples.append(DatasetSample(name=file_name, original_image=image, input_image=input_image))
    return ImageDataset(dataset_name, samples)


def _build_cave_dataset(dataset_name: str) -> ImageDataset:
    if dataset_name == "cave_lowlight":
        original_dir = ROOT / "CAVE_D65_Lowlight_Raw"
        input_dir = ROOT / "CAVE_D65_Lowlight"
    elif dataset_name == "cave_extreme_lowlight":
        original_dir = ROOT / "CAVE_D65_Extreme_Lowlight_Raw"
        input_dir = ROOT / "CAVE_D65_Extreme_Lowlight"
    elif dataset_name == "cave_normal":
        original_dir = ROOT / "CAVE_D65_Raw"
        input_dir = ROOT / "CAVE_D65_Raw"
    else:
        raise ValueError(f"Unsupported CAVE dataset: {dataset_name}")

    original_files = sorted(original_dir.glob("*.mat"))
    if not original_files:
        raise ValueError(f"No CAVE reference mat files found in {original_dir}")

    samples = []
    for original_path in original_files:
        input_path = input_dir / original_path.name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input mat file for scene {original_path.name}: {input_path}")

        original_rgbw = _load_cave_rgbw_mat(original_path)
        input_rgbw = _load_cave_rgbw_mat(input_path)
        original_image = _convert_to_uint8_image(original_rgbw[:, :, :3], source_max_value=CAVE_RGBW_MAX_VALUE)
        if dataset_name == "cave_normal":
            input_image = _convert_to_uint8_image(input_rgbw[:, :, :3], source_max_value=CAVE_RGBW_MAX_VALUE)
        else:
            input_rgb = _convert_to_uint8_image(input_rgbw[:, :, :3], source_max_value=CAVE_RGBW_MAX_VALUE)
            input_w = _convert_to_uint8_image(input_rgbw[:, :, 3], source_max_value=CAVE_RGBW_MAX_VALUE)
            input_image = np.dstack((input_rgb, input_w))
        samples.append(
            DatasetSample(
                name=f"{original_path.stem}.png",
                original_image=original_image,
                input_image=input_image,
            )
        )
    return ImageDataset(dataset_name, samples)


def load_dataset(dataset_name: str) -> ImageDataset:
    if dataset_name in {"kodak", "low_light", "extreme_lowlight"}:
        return _build_generic_dataset(dataset_name)
    if dataset_name in {"cave_lowlight", "cave_extreme_lowlight", "cave_normal"}:
        return _build_cave_dataset(dataset_name)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def compute_cpsnr(I: np.ndarray, Igt: np.ndarray, bounder: int) -> float:
    if I.shape != Igt.shape:
        raise ValueError("The image size should be the same!")

    if bounder > 0:
        I = I[bounder:-bounder, bounder:-bounder, :]
        Igt = Igt[bounder:-bounder, bounder:-bounder, :]

    E = I.astype(np.float64) - Igt.astype(np.float64)
    mse = np.mean(E ** 2)
    if np.max(Igt) > 2:
        return 10.0 * np.log10((255.0 ** 2) / mse)
    return 10.0 * np.log10(1.0 / mse)


def mosaick(img: np.ndarray, pcfa: np.ndarray) -> np.ndarray:
    r, c, _ = pcfa.shape
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected H x W x 3 or H x W x 4 image for mosaicking, got {img.shape}")

    rr, cc, _ = img.shape
    r_ceil = int(np.ceil(rr / r))
    c_ceil = int(np.ceil(cc / c))
    cfa = np.tile(pcfa, (r_ceil, c_ceil, 1))
    cfa = cfa[:rr, :cc, :]

    mosaicked = np.sum(img[:, :, :3].astype(np.float64) * cfa, axis=2)
    if img.shape[2] == 4:
        w_mask = np.all(np.isclose(cfa, 1.0 / 3.0), axis=2)
        mosaicked[w_mask] = img[:, :, 3].astype(np.float64)[w_mask]
    return mosaicked


def _coeff_modulated(dx: float, dy: float, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.exp(2.0 * np.pi * 1j * (X * dx + Y * dy))


def modulation_functions(m: int, n: int, locations: np.ndarray) -> list[np.ndarray]:
    y = np.arange(m)
    x = np.arange(n)
    X, Y = np.meshgrid(x, y)
    return [_coeff_modulated(location[0], location[1], X, Y) for location in locations]


def multiplex(img: np.ndarray, M: np.ndarray) -> list[np.ndarray]:
    img = img.astype(np.float64)
    channels = [img[:, :, channel] for channel in range(img.shape[2])]
    outputs: list[np.ndarray] = []
    for row in np.asarray(M):
        component = np.zeros_like(channels[0], dtype=np.complex128)
        for coeff, channel in zip(row, channels, strict=True):
            component = component + coeff * channel
        outputs.append(component)
    return outputs


def demultiplex(components: list[np.ndarray], M: np.ndarray) -> np.ndarray:
    D = np.linalg.inv(M) if M.shape[0] == M.shape[1] == 3 else np.linalg.pinv(M)
    m, n = components[0].shape
    img = np.zeros((m, n, 3), dtype=np.complex128)
    for channel_index in range(3):
        value = np.zeros((m, n), dtype=np.complex128)
        for coeff, component in zip(D[channel_index], components, strict=True):
            value = value + coeff * component
        img[:, :, channel_index] = value
    return img


def least_square(
    train_img: list[DatasetSample],
    cfa: dict,
    Nfilt: int = 21,
    progress_desc: str = "Training",
    show_progress: bool = True,
) -> tuple[list[np.ndarray], ...]:
    MS = 96
    MS2 = MS * MS
    frame = 11
    Nfiltp = (Nfilt - 1) // 2
    NT = Nfilt * Nfilt

    pcfa = cfa["pcfa"]
    M = cfa["M"]
    specs = _chroma_specs(cfa)
    if not train_img:
        raise ValueError("Training set cannot be empty.")

    m, n, _ = train_img[0].original_image.shape
    NBLK1 = int(np.fix((m - 2 * frame) / MS))
    NBLK2 = int(np.fix((n - 2 * frame) / MS))
    modulation_bank = [modulation_functions(m, n, locations) for locations, _, _ in specs]

    ATf = [[np.zeros((NT,), dtype=np.complex128) for _ in coeffs] for _, _, coeffs in specs]
    ATA = np.zeros((NT, NT), dtype=np.float64)

    if show_progress:
        progress_bar = tqdm(
            enumerate(train_img, start=1),
            total=len(train_img),
            desc=progress_desc,
            leave=False,
        )
    else:
        progress_bar = enumerate(train_img, start=1)

    for train_index, sample in progress_bar:
        if show_progress:
            progress_bar.set_postfix(train=f"{train_index}/{len(train_img)}")

        original_image = sample.original_image.astype(np.float64)
        input_image = sample.input_image.astype(np.float64)
        img_mosaicked = mosaick(input_image, pcfa)
        components = multiplex(original_image, M)
        chroma_components = components[1:]

        chroma_modulated_sets: list[list[np.ndarray]] = []
        for chroma, (_, is_conj_flags, coeffs), modulations in zip(chroma_components, specs, modulation_bank, strict=True):
            modulated_group: list[np.ndarray] = []
            for modulation, is_conj, coeff in zip(modulations, is_conj_flags, coeffs, strict=True):
                base = np.conj(chroma) if is_conj else chroma
                modulated_group.append((base * modulation) * coeff)
            chroma_modulated_sets.append(modulated_group)

        for b1 in range(NBLK1):
            row_start = frame + b1 * MS
            row_end = frame + (b1 + 1) * MS
            for b2 in range(NBLK2):
                col_start = frame + b2 * MS
                col_end = frame + (b2 + 1) * MS

                target_vectors = [
                    [np.reshape(item[row_start:row_end, col_start:col_end], MS2, order="F") for item in modulated_group]
                    for modulated_group in chroma_modulated_sets
                ]

                A = np.zeros((MS2, NT), dtype=np.float64)
                for c1 in range(Nfilt):
                    for c2 in range(Nfilt):
                        col = c1 * Nfilt + c2
                        rs = row_start - (c1 - Nfiltp)
                        re = row_end - (c1 - Nfiltp)
                        cs = col_start - (c2 - Nfiltp)
                        ce = col_end - (c2 - Nfiltp)
                        patch = img_mosaicked[rs:re, cs:ce]
                        A[:, col] = np.reshape(patch, MS2, order="F")

                AT = A.T
                for chroma_index, chroma_vectors in enumerate(target_vectors):
                    for redundant_index, vector in enumerate(chroma_vectors):
                        ATf[chroma_index][redundant_index] = ATf[chroma_index][redundant_index] + AT @ vector
                ATA = ATA + AT @ A

    filters: list[list[np.ndarray]] = []
    for chroma_vectors in ATf:
        one_chroma_filters: list[np.ndarray] = []
        for target in chroma_vectors:
            coeff_vector = np.linalg.lstsq(ATA, target, rcond=None)[0]
            one_chroma_filters.append(np.conj(np.reshape(coeff_vector, (Nfilt, Nfilt), order="F").T))
        filters.append(one_chroma_filters)
    return tuple(filters)


def demosaick(img_mosaicked: np.ndarray, cfa: dict, *filters: list[np.ndarray]) -> np.ndarray:
    specs = _chroma_specs(cfa)
    if len(filters) != len(specs):
        raise ValueError(f"Expected {len(specs)} chroma filter groups, got {len(filters)}.")

    M = cfa["M"]
    m, n = img_mosaicked.shape
    modulation_bank = [modulation_functions(m, n, locations) for locations, _, _ in specs]

    chroma_estimates: list[np.ndarray] = []
    for filter_group, (_, is_conj_flags, coeffs), modulations in zip(filters, specs, modulation_bank, strict=True):
        filtered_group = [convolve2d(img_mosaicked, filt, mode="same") for filt in filter_group]
        chroma_candidates: list[np.ndarray] = []
        for filtered, modulation, coeff, is_conj in zip(filtered_group, modulations, coeffs, is_conj_flags, strict=True):
            estimate = (filtered * modulation) / coeff
            if is_conj:
                estimate = np.conj(estimate)
            chroma_candidates.append(estimate)
        chroma = np.zeros_like(chroma_candidates[0])
        for candidate in chroma_candidates:
            chroma = chroma + candidate / len(chroma_candidates)
        chroma_estimates.append(np.conj(chroma))

    non_L = np.zeros((m, n), dtype=np.complex128)
    for chroma, (_, is_conj_flags, coeffs), modulations in zip(chroma_estimates, specs, modulation_bank, strict=True):
        for modulation, is_conj, coeff in zip(modulations, is_conj_flags, coeffs, strict=True):
            chroma_modulated = np.conj(chroma) if is_conj else chroma
            non_L = non_L + (chroma_modulated * modulation) * coeff

    L = img_mosaicked - non_L
    img_estimated = demultiplex([L, *chroma_estimates], M)
    return matlab_uint8(np.real(img_estimated))