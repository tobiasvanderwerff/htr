"""
Run inference on a model and write predictions to a file. NOTE: currently only
works for FPHTR.

Also worth noting is that because the images are not batched but processed
individually, no padding is applied to them, in contrast to the setup during training.
"""


import argparse
from pathlib import Path
from typing import Tuple, Union, List, Optional, Callable

from htr.data import IAMDataset
import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import pandas as pd

from htr.models.lit_models import LitFullPageHTREncoderDecoder
from htr.transforms import IAMImageTransforms
from htr.util import LabelEncoder


def run_model(
    model_path: Path,
    imgs: Union[Path, List[Path]],
    label_encoder: LabelEncoder,
    img_transform: Optional[Callable] = None,
) -> None:
    """Run inference and write all predictions to a file."""
    all_preds, pred_strs = [], []

    # Load model.
    model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
        str(model_path),
        label_encoder=label_encoder,
    )

    # Make prediction(s).
    model.eval()
    if isinstance(imgs, Path):
        img_path = imgs
        img = prepare_image(img_path, img_transform)
        with torch.inference_mode():
            _, preds, _ = model(img.unsqueeze(0))
            preds = preds.squeeze().numpy()
        all_preds.append((str(img_path), preds))
    else:  # List[Path]
        for img_path in imgs:
            img = prepare_image(img_path, img_transform)
            with torch.inference_mode():
                _, preds, _ = model(img.unsqueeze(0))
                preds = preds.squeeze().numpy()
            all_preds.append((str(img_path), preds))

    # Decode the prediction(s).
    for img_path, sampled_ids in all_preds:
        if sampled_ids[0] == label_encoder.transform(["<SOS>"])[0]:
            sampled_ids = sampled_ids[1:]
        if sampled_ids[-1] == label_encoder.transform(["<EOS>"])[0]:
            sampled_ids = sampled_ids[:-1]
        pred_strs.append(
            (str(img_path), "".join((label_encoder.inverse_transform(sampled_ids))))
        )

    # Write the predictions to a file.
    pred_strs.sort(key=lambda x: x[0])
    to_write = "\n".join("\t".join([pth, pred]) for pth, pred in pred_strs)
    Path("model_predictions.txt").write_text(to_write)


def prepare_image(img_path: Path, transform: Optional[Callable] = None):
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    assert isinstance(img, np.ndarray), (
        f"Error: image at path {img_path} is not properly loaded. "
        f"Is there something wrong with this image?"
    )
    if transform is not None:
        img = torch.from_numpy(transform(image=img)["image"])
    return img


def load_label_encoder(model_path: Path):
    le_dir = model_path.parent.parent
    le_path = (
        le_dir / "label_encoder.pkl"
        if (le_dir / "label_encoder.pkl").is_file()
        else le_dir / "label_encoding.txt"
    )
    return LabelEncoder().read_encoding(le_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a trained model.")

    # fmt: off
    parser.add_argument("--img_path", type=Path, required=True,
                        help="Path to an image for which a prediction should be made. "
                             "If a directory is specified, run inference on all "
                             "images in that directory.")
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to a trained model checkpoint, which will be loaded "
                             "for inference.")
    parser.add_argument("--data_format", type=str, choices=["form", "line", "word"],
                        required=True, help=("Data format of the image(s). Is it a "
                                             "full page, a line, or a word?"))
    args = parser.parse_args()
    # fmt: on

    img_path, model_path = args.img_path, args.model_path
    assert model_path.is_file(), f"{model_path} does not point to a file."
    assert (
        img_path.is_file() or img_path.is_symlink() or img_path.is_dir()
    ), f"Image path {img_path} does not point to a file or directory."

    # Load the label encoder for the trained model.
    label_encoder = load_label_encoder(model_path)

    # Load image transform.
    img_transform = IAMImageTransforms(
        (0, 0), args.data_format, (IAMDataset.MEAN, IAMDataset.STD)
    ).test_trnsf

    imgs = img_path
    if img_path.is_dir():
        # Search for all images in a directory.
        imgs = list(img_path.rglob("*.png")) + list(img_path.rglob("*.jpg"))

    run_model(model_path, imgs, label_encoder, img_transform)
    print("Done.")
