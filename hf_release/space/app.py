from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from skp.configs import Config
from skp.configs.defaults import classification_2d_defaults, segmentation_2d_defaults
from skp.models.classification.net2d import Net as Classifier
from skp.models.segmentation.base import Net as Segmenter


MODEL_REPO = "ianpan/ct-head-hemorrhage-detection"
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR")
CLASS_NAMES = [
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
    "Any hemorrhage",
]
SEGMENTATION_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
EXAMPLE_TRUTH = {}
LARGE_IMAGE_RESIZE_THRESHOLD = 640


def model_artifact(filename: str) -> str:
    if LOCAL_MODEL_DIR:
        return str(Path(LOCAL_MODEL_DIR) / filename)
    return hf_hub_download(repo_id=MODEL_REPO, filename=filename)


def classifier_config() -> Config:
    cfg = Config()
    classification_2d_defaults(cfg)
    cfg.backbone = "maxvit_tiny_tf_512.in1k"
    cfg.pretrained = False
    cfg.num_input_channels = 9
    cfg.num_classes = 6
    cfg.pool = "avg"
    cfg.dropout = 0.2
    cfg.normalization = "linear"
    cfg.normalization_params = {
        "input_min": 0.0,
        "input_max": 1.0,
        "output_min": -1.0,
        "output_max": 1.0,
    }
    cfg.backbone_img_size = False
    cfg.image_height = 512
    cfg.image_width = 512
    return cfg


def segmenter_config() -> Config:
    cfg = Config()
    segmentation_2d_defaults(cfg)
    cfg.backbone = "maxvit_tiny_tf_512.in1k"
    cfg.pretrained = False
    cfg.num_input_channels = 9
    cfg.num_classes = 6
    cfg.normalization = "linear"
    cfg.normalization_params = {
        "input_min": 0.0,
        "input_max": 1.0,
        "output_min": -1.0,
        "output_max": 1.0,
    }
    cfg.backbone_img_size = False
    cfg.image_height = 512
    cfg.image_width = 512
    cfg.decoder_type = "DeepLabV3PlusDecoder"
    cfg.decoder_out_channels = 256
    cfg.decoder_norm_layer = "bn"
    cfg.decoder_act_layer = "relu"
    cfg.decoder_attention_type = None
    cfg.decoder_center_block = False
    cfg.aspp_separable = True
    cfg.aspp_dropout = 0.1
    cfg.atrous_rates = (6, 12, 18, 24)
    cfg.seg_dropout = 0.0
    cfg.deep_supervision = False
    cfg.enable_gradient_checkpointing = False
    return cfg


class HemorrhageModel:
    def __init__(self) -> None:
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classifier_path = model_artifact("classifier/model.safetensors")
        self.classifier_state = load_file(classifier_path, device="cpu")
        classifier = Classifier(classifier_config())
        classifier.load_state_dict(self.classifier_state, strict=True)
        classifier.eval()

        segmenter = Segmenter(segmenter_config())
        segmenter_state = segmenter.state_dict()
        shared_encoder = {}
        for key, value in self.classifier_state.items():
            if not key.startswith("backbone."):
                continue
            if key.startswith("backbone.head."):
                continue
            encoder_key = key.removeprefix("backbone.")
            if encoder_key.startswith("stages."):
                encoder_key = "stages_" + encoder_key.removeprefix("stages.")
            segmenter_key = "encoder." + encoder_key
            if segmenter_key in segmenter_state:
                shared_encoder[segmenter_key] = value
        expected_encoder_keys = {
            f"encoder.{key}" for key in segmenter.encoder.state_dict()
        }
        if set(shared_encoder) != expected_encoder_keys:
            missing = sorted(expected_encoder_keys - set(shared_encoder))
            raise RuntimeError(f"Missing shared encoder keys: {missing[:5]}")
        incompatible = segmenter.load_state_dict(shared_encoder, strict=False)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected shared encoder keys: {sorted(unexpected)}")
        self.encoder = segmenter.encoder.eval().to(self.device)
        self.classifier_head = classifier.backbone.head.eval().to(self.device)
        self.classifier_pooling = classifier.pooling.eval().to(self.device)
        self.classifier_linear = classifier.linear.eval().to(self.device)

        self.segmentation_decoders = []
        self.segmentation_heads = []
        for fold in range(5):
            fold_state = load_file(
                model_artifact(f"segmentation/fold{fold}.safetensors"),
                device="cpu",
            )
            decoder = copy.deepcopy(segmenter.decoder)
            head = copy.deepcopy(segmenter.segmentation_head)
            decoder.load_state_dict(
                {
                    key.removeprefix("decoder."): value
                    for key, value in fold_state.items()
                    if key.startswith("decoder.")
                },
                strict=True,
            )
            head.load_state_dict(
                {
                    key.removeprefix("segmentation_head."): value
                    for key, value in fold_state.items()
                    if key.startswith("segmentation_head.")
                },
                strict=True,
            )
            self.segmentation_decoders.append(decoder.eval().to(self.device))
            self.segmentation_heads.append(head.eval().to(self.device))

    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(x[None]).float().to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            feature_maps = self.encoder(tensor * 2.0 - 1.0)
            classifier_map = self.classifier_head(feature_maps[-1], pre_logits=True)
            cls_logits = self.classifier_linear(self.classifier_pooling(classifier_map))
        cls_probabilities = cls_logits.float().sigmoid().cpu().numpy()[0]

        fold_probabilities = []
        for decoder, head in zip(
            self.segmentation_decoders,
            self.segmentation_heads,
        ):
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                logits = head(decoder(feature_maps)[-1])
            fold_probabilities.append(logits.float().sigmoid().cpu())
        segmentation = torch.stack(fold_probabilities).mean(0).numpy()[0]
        return cls_probabilities, segmentation


MODEL = HemorrhageModel()


def center_crop_or_pad(image: np.ndarray, size: int = 512) -> np.ndarray:
    height, width = image.shape
    if height > LARGE_IMAGE_RESIZE_THRESHOLD or width > LARGE_IMAGE_RESIZE_THRESHOLD:
        return cv2.resize(
            image.astype(np.float32),
            (size, size),
            interpolation=cv2.INTER_AREA,
        )
    top = max((height - size) // 2, 0)
    left = max((width - size) // 2, 0)
    image = image[top : top + size, left : left + size]
    pad_height = size - image.shape[0]
    pad_width = size - image.shape[1]
    return np.pad(
        image,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
        ),
        mode="constant",
        constant_values=0,
    )


def load_png(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise gr.Error(f"Could not read {Path(path).name}.")
    if image.ndim != 2:
        raise gr.Error("Each input must be a single-channel 16-bit PNG.")
    if image.dtype != np.uint16:
        raise gr.Error(f"{Path(path).name} has dtype {image.dtype}; expected uint16.")
    return center_crop_or_pad(image)


def window_slice(
    image: np.ndarray | None,
    slope: float,
    intercept: float,
) -> np.ndarray:
    if image is None:
        return np.zeros((3, 512, 512), dtype=np.float32)
    hu = image.astype(np.float32) * float(slope) + float(intercept)
    centers = np.asarray([40.0, 80.0, 600.0], dtype=np.float32)
    widths = np.asarray([80.0, 200.0, 2800.0], dtype=np.float32)
    lower = centers - widths / 2
    upper = centers + widths / 2
    x = np.clip(hu[None], lower[:, None, None], upper[:, None, None])
    return (x - lower[:, None, None]) / (widths[:, None, None] + 1e-6)


def example_key(center_path: str | None) -> str | None:
    if not center_path:
        return None
    stem = Path(center_path).stem
    return "_".join(stem.split("_")[:2]) if stem.startswith("example_") else None


def make_overlay(base: np.ndarray, probability: np.ndarray) -> np.ndarray:
    base_rgb = np.repeat((np.clip(base, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1)
    heat = np.zeros_like(base_rgb)
    heat[..., 0] = 255
    heat[..., 1] = 42
    heat[..., 2] = 24
    alpha = (np.clip(probability, 0, 1) ** 0.7 * 0.78)[..., None]
    return np.clip(base_rgb * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def predict(
    previous_path: str | None,
    center_path: str | None,
    next_path: str | None,
    slope: float,
    intercept: float,
    segmentation_class: str,
):
    if not center_path:
        raise gr.Error("A center slice is required.")
    images = [load_png(previous_path), load_png(center_path), load_png(next_path)]
    windowed = np.stack(
        [window_slice(image, slope, intercept) for image in images],
        axis=0,
    )
    x = np.ascontiguousarray(windowed.reshape(9, 512, 512))
    cls_probabilities, segmentation = MODEL.predict(x)

    label_output = {
        name: float(probability)
        for name, probability in zip(CLASS_NAMES, cls_probabilities)
    }
    class_index = SEGMENTATION_INDEX[segmentation_class]
    center_brain = windowed[1, 0]
    overlay = make_overlay(center_brain, segmentation[class_index])

    key = example_key(center_path)
    truth = EXAMPLE_TRUTH.get(key)
    truth_line = f"\n\n**Heldout example label:** {truth}." if truth else ""
    note = (
        f"**Standalone slice prediction.** Any hemorrhage: "
        f"{cls_probabilities[-1]:.1%}. Heatmap channel: {segmentation_class}."
        f"{truth_line}\n\n"
        "The heatmap is a continuous five-fold ensemble probability and has not "
        "been calibrated for clinical use. The series BiGRU is not run in this "
        "single-slab demo."
    )
    return label_output, overlay, note


EXAMPLES = []
manifest_path = Path(__file__).parent / "examples" / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    for item in manifest:
        EXAMPLE_TRUTH[item["key"]] = item["ground_truth"]
        prefix = Path(__file__).parent / "examples" / item["key"]
        EXAMPLES.append(
            [
                str(prefix.with_name(prefix.name + "_previous.png")),
                str(prefix.with_name(prefix.name + "_center.png")),
                str(prefix.with_name(prefix.name + "_next.png")),
                item["rescale_slope"],
                item["rescale_intercept"],
                "Any hemorrhage",
            ]
        )


with gr.Blocks(title="CT Head Hemorrhage Detection") as demo:
    gr.Markdown(
        "# CT Head Hemorrhage Detection\n"
        "Research demonstration for a three-adjacent-slice head CT slab. "
        "Upload the original 16-bit PNGs and series rescale values. "
        "**Not for clinical use.**"
    )
    with gr.Row():
        previous = gr.File(
            label="Previous slice (optional)", file_types=[".png"], type="filepath"
        )
        center = gr.File(label="Center slice", file_types=[".png"], type="filepath")
        next_slice = gr.File(
            label="Next slice (optional)", file_types=[".png"], type="filepath"
        )
    with gr.Row():
        slope = gr.Number(label="Rescale slope", value=1.0)
        intercept = gr.Number(label="Rescale intercept", value=-1024.0)
        segmentation_class = gr.Dropdown(
            choices=CLASS_NAMES,
            value="Any hemorrhage",
            label="Heatmap class",
        )
        run = gr.Button("Run inference", variant="primary")
    with gr.Row():
        classification = gr.Label(label="Slice classification", num_top_classes=6)
        overlay = gr.Image(label="Center brain window + segmentation heatmap")
    details = gr.Markdown()

    run.click(
        predict,
        inputs=[previous, center, next_slice, slope, intercept, segmentation_class],
        outputs=[classification, overlay, details],
    )
    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=[previous, center, next_slice, slope, intercept, segmentation_class],
            outputs=[classification, overlay, details],
            fn=predict,
            cache_examples=False,
            label="Deidentified heldout test examples",
        )

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()
