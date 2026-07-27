from __future__ import annotations

import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import freeze_support, get_context
from pathlib import Path
from queue import Empty
from typing import Any


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from tqdm.auto import tqdm

from common import ROOT, RESULTS_DIR, ensure_dir, load_cfa_by_name, write_image
from generic_pipeline import compute_cpsnr, demosaick, infer_chroma_count, least_square, load_dataset, mosaick


WORKER_CONTEXT: dict[str, Any] = {}
STATUS_QUEUE: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfa_name", default="SD8")
    parser.add_argument(
        "--dataset_name",
        choices=("kodak", "low_light", "extreme_lowlight", "cave_lowlight", "cave_extreme_lowlight", "cave_normal"),
        default="kodak",
    )
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def init_worker_context(
    cfa_name: str,
    dataset_name: str,
    bounder_size: int,
    output_folder: str,
    status_queue: Any,
) -> None:
    global STATUS_QUEUE

    cfa = load_cfa_by_name(cfa_name)
    dataset = load_dataset(dataset_name)
    STATUS_QUEUE = status_queue
    WORKER_CONTEXT.clear()
    WORKER_CONTEXT.update(
        {
            "cfa_name": cfa_name,
            "dataset_name": dataset_name,
            "bounder_size": bounder_size,
            "output_folder": Path(output_folder),
            "cfa": cfa,
            "dataset": dataset,
        }
    )
    if STATUS_QUEUE is not None:
        STATUS_QUEUE.put(("worker_ready", os.getpid(), len(dataset), infer_chroma_count(cfa)))


def process_single_image(i_img: int) -> tuple[int, float]:
    cfa = WORKER_CONTEXT["cfa"]
    dataset = WORKER_CONTEXT["dataset"]
    dataset_name = WORKER_CONTEXT["dataset_name"]
    bounder_size = WORKER_CONTEXT["bounder_size"]
    output_folder = WORKER_CONTEXT["output_folder"]

    if STATUS_QUEUE is not None:
        STATUS_QUEUE.put(("image_start", os.getpid(), i_img + 1, len(dataset)))

    this_train = dataset.samples[:i_img] + dataset.samples[i_img + 1 :]
    trained = least_square(
        this_train,
        cfa,
        progress_desc=f"Train for test image {i_img + 1}",
        show_progress=False,
    )

    this_sample = dataset[i_img]
    img_mosaicked = mosaick(this_sample.input_image, cfa["pcfa"])
    this_estimated = demosaick(img_mosaicked, cfa, *trained)
    cpsnr = float(compute_cpsnr(this_estimated, this_sample.original_image, bounder_size))

    write_image(output_folder / dataset.output_file_name(i_img), this_estimated)

    if STATUS_QUEUE is not None:
        STATUS_QUEUE.put(("image_done", os.getpid(), i_img + 1, cpsnr))

    return i_img, cpsnr


def run_parallel(
    workers: int,
    cfa_name: str,
    dataset_name: str,
    bounder_size: int,
    output_folder: Path,
    image_count: int,
) -> np.ndarray:
    cpsnr = np.zeros((image_count + 1,), dtype=np.float64)
    context = get_context("spawn")
    status_queue = context.Queue()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=init_worker_context,
        initargs=(cfa_name, dataset_name, bounder_size, str(output_folder), status_queue),
    ) as executor:
        future_map = {executor.submit(process_single_image, i_img): i_img for i_img in range(image_count)}
        progress = tqdm(total=image_count, desc="Testing images")
        pending = set(future_map)
        completed = 0
        active_images: set[int] = set()
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

            while True:
                try:
                    message = status_queue.get_nowait()
                except Empty:
                    break

                tag = message[0]
                if tag == "worker_ready":
                    _, pid, worker_image_count, chroma_count = message
                    tqdm.write(f"Worker {pid} ready, image_count={worker_image_count}, chromas={chroma_count}")
                elif tag == "image_start":
                    _, pid, image_index, total_images = message
                    active_images.add(image_index)
                    progress.set_postfix(active=len(active_images), last_start=f"{image_index}/{total_images}")
                    tqdm.write(f"Worker {pid} started image {image_index}/{total_images}")
                elif tag == "image_done":
                    _, pid, image_index, psnr_value = message
                    active_images.discard(image_index)
                    progress.set_postfix(active=len(active_images), last_done=f"{image_index}", cpsnr=f"{psnr_value:.4f}")
                    tqdm.write(f"Worker {pid} finished image {image_index}, cpsnr={psnr_value:.4f}")

            for future in done:
                i_img, psnr_value = future.result()
                cpsnr[i_img] = psnr_value
                completed += 1
                progress.update(1)
                progress.set_postfix(completed=f"{completed}/{image_count}", active=len(active_images), last=f"img {i_img + 1}", cpsnr=f"{psnr_value:.4f}")
        progress.close()
    return cpsnr


def run_sequential(
    cfa: dict[str, np.ndarray],
    dataset,
    dataset_name: str,
    bounder_size: int,
    output_folder: Path,
) -> np.ndarray:
    cpsnr = np.zeros((len(dataset) + 1,), dtype=np.float64)
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
        write_image(output_folder / dataset.output_file_name(i_img), this_estimated)

        outer_progress.set_postfix(cpsnr=f"{cpsnr[i_img]:.4f}")
    return cpsnr


def main() -> np.ndarray:
    args = parse_args()
    cfa_name = args.cfa_name
    dataset_name = args.dataset_name
    workers = args.workers or max(1, (os.cpu_count() or 1) - 1)

    bounder_size = 0
    output_folder = RESULTS_DIR / dataset_name / cfa_name
    log_path = output_folder / "log.txt"
    ensure_dir(output_folder)

    cfa = load_cfa_by_name(cfa_name)
    chroma_count = infer_chroma_count(cfa)
    norm2_invM = np.linalg.norm(np.linalg.pinv(cfa["M"]), 2)
    print(f"{cfa_name} CFA: chromas={chroma_count}, ||M^{{-1}}||_2 = {norm2_invM:g}")

    dataset = load_dataset(dataset_name)
    print(f"Dataset: {dataset_name}, image_count={len(dataset)}")

    if workers > 1:
        print(f"Running with {workers} worker processes.")
        cpsnr = run_parallel(
            workers=workers,
            cfa_name=cfa_name,
            dataset_name=dataset_name,
            bounder_size=bounder_size,
            output_folder=output_folder,
            image_count=len(dataset),
        )
    else:
        print("Running sequentially with 1 worker.")
        cpsnr = run_sequential(cfa, dataset, dataset_name, bounder_size, output_folder)

    log_lines = [f"{i_img + 1}:cpsnr={cpsnr[i_img]:g}" for i_img in range(len(dataset))]
    cpsnr[-1] = np.mean(cpsnr[:-1])
    avg_message = f"Avg. CPSNR of {cfa_name} CFA on the {dataset_name} dataset:{cpsnr[-1]:g}!"
    print(avg_message)
    log_lines.append(avg_message)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return cpsnr


if __name__ == "__main__":
    freeze_support()
    main()
