from __future__ import annotations

import argparse

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from common import RESULTS_DIR, ensure_dir, load_cfa_by_name, write_image
from generic_pipeline import compute_cpsnr, demosaick, infer_chroma_count, least_square, load_dataset, mosaick


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfa_name", default="SD8")
    parser.add_argument(
        "--dataset_name",
        choices=("kodak", "low_light", "extreme_lowlight", "cave_lowlight", "cave_extreme_lowlight", "cave_normal"),
        default="kodak",
    )
    return parser.parse_args()


def main() -> np.ndarray:
    args = parse_args()
    cfa_name = args.cfa_name
    dataset_name = args.dataset_name

    bounder_size = 0
    output_folder = RESULTS_DIR / dataset_name / cfa_name
    log_path = output_folder / "log.txt"
    ensure_dir(output_folder)

    cfa = load_cfa_by_name(cfa_name)
    chroma_count = infer_chroma_count(cfa)
    norm2_invM = np.linalg.norm(np.linalg.pinv(cfa["M"]), 2)
    print(f"{cfa_name} CFA: chromas={chroma_count}, ||M^{{-1}}||_2 = {norm2_invM:g}")

    plt.figure()
    plt.imshow(cfa["pcfa"])
    plt.title(f"{cfa_name}:{norm2_invM:g}")
    plt.axis("off")

    dataset = load_dataset(dataset_name)

    cpsnr = np.zeros((len(dataset) + 1,), dtype=np.float64)
    log_lines: list[str] = []
    outer_progress = tqdm(range(len(dataset)), desc="Testing images")
    for i_img in outer_progress:
        outer_progress.set_description(f"Testing image {i_img + 1}/{len(dataset)}")
        print(f"Running test image {i_img + 1}/{len(dataset)}")
        this_train = dataset.samples[:i_img] + dataset.samples[i_img + 1 :]
        trained = least_square(this_train, cfa, progress_desc=f"Train for test image {i_img + 1}")

        this_sample = dataset[i_img]
        img_mosaicked = mosaick(this_sample.input_image, cfa["pcfa"])
        this_estimated = demosaick(img_mosaicked, cfa, *trained)

        cpsnr[i_img] = compute_cpsnr(this_estimated, this_sample.original_image, bounder_size)
        print(f"{i_img + 1}:cpsnr={cpsnr[i_img]:g}")
        log_lines.append(f"{i_img + 1}:cpsnr={cpsnr[i_img]:g}")
        write_image(output_folder / dataset.output_file_name(i_img), this_estimated)

        outer_progress.set_postfix(cpsnr=f"{cpsnr[i_img]:.4f}")

    cpsnr[-1] = np.mean(cpsnr[:-1])
    avg_message = f"Avg. CPSNR of {cfa_name} CFA on the {dataset_name} dataset:{cpsnr[-1]:g}!"
    print(avg_message)
    log_lines.append(avg_message)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return cpsnr


if __name__ == "__main__":
    main()
