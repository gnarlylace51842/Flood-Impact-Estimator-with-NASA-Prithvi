import os
import argparse
import numpy as np
import cv2
import rasterio
import torch
from huggingface_hub import hf_hub_download

from mmcv import Config
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose, LoadImageFromFile
from mmcv.parallel import collate, scatter


def inference_segmentor(model, img_path, custom_test_pipeline=None):
    cfg = model.cfg
    device = next(model.parameters()).device

    test_pipeline = [LoadImageFromFile()] + cfg.data.test.pipeline[1:] if custom_test_pipeline is None else custom_test_pipeline
    test_pipeline = Compose(test_pipeline)

    data = {"img_info": {"filename": img_path}}
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        img_metas = data["img_metas"].data[0]
        img = data["img"]
        data = {"img": img, "img_metas": img_metas}

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result[0]


def save_mask_png(mask01, out_path):
    cv2.imwrite(out_path, (mask01.astype(np.uint8) * 255))


def make_quick_rgb_preview(tif_path, out_path="image.png"):
    with rasterio.open(tif_path) as src:
        x = src.read()

    b = x[0].astype(np.float32)
    g = x[1].astype(np.float32)
    r = x[2].astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)

    rgb = rgb - np.nanmin(rgb)
    denom = np.nanmax(rgb)
    if denom > 0:
        rgb = rgb / denom
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Path to input 6-band GeoTIFF (single timestamp)")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0, cuda:1, etc.")
    parser.add_argument("--foreground_class", type=int, default=1, help="Foreground class index (binary mask = label==this)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    repo_id = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11"
    config_path = hf_hub_download(repo_id=repo_id, filename="sen1floods11_Prithvi_100M.py")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="sen1floods11_Prithvi_100M.pth")

    cfg = Config.fromfile(config_path)
    cfg.model.backbone.pretrained = None

    model = init_segmentor(cfg, ckpt_path, device=args.device)

    label_map = inference_segmentor(model, args.tif)

    mask01 = (label_map == args.foreground_class).astype(np.uint8)

    image_png = os.path.join(args.outdir, "image.png")
    mask_png = os.path.join(args.outdir, "mask.png")

    make_quick_rgb_preview(args.tif, out_path=image_png)
    save_mask_png(mask01, mask_png)

    print("Saved:")
    print(" ", image_png)
    print(" ", mask_png)


if __name__ == "__main__":
    main()
