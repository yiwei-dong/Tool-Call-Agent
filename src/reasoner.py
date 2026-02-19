import re
import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

from agent_model import Qwen2_5_VL_Agent
from utils.heads import AgentConfig
from utils.tools import (
    grounded_segmentation, zoom_in_image_by_bbox, visual_search,
    ocr_extract_text, overlay, crop_image_action, segment_image,
    calculate_image_to_images_similarity,
    calculate_text_to_images_similarity,
    calculate_image_to_texts_similarity,
    get_depth,
)
from utils.prompts import load_model_response_prompt, load_vts_system_prompt

ID_TO_ACTION = {
    0:  "GROUNDING",
    1:  "DEPTH",
    2:  "ZOOMIN",
    3:  "VISUALSEARCH",
    4:  "TEXT",
    5:  "OVERLAY",
    6:  "CROP",
    7:  "SEGMENT",
    8:  "OCR",
    9:  "TEXT_TO_IMAGES_SIMILARITY",
    10: "IMAGE_TO_TEXTS_SIMILARITY",
    11: "IMAGE_TO_IMAGES_SIMILARITY",
    12: "TERMINATE",
}
ACTION_TO_ID = {v: k for k, v in ID_TO_ACTION.items()}

_PROMPT_NAME_TO_INTERNAL = {
    name.replace("_", "").upper(): internal
    for internal, name in {
        "GROUNDING":                  "Grounding",
        "DEPTH":                      "Depth",
        "ZOOMIN":                     "ZoomIn",
        "VISUALSEARCH":               "VisualSearch",
        "TEXT":                       "Text",
        "OVERLAY":                    "Overlay",
        "CROP":                       "Crop",
        "SEGMENT":                    "Segment",
        "OCR":                        "OCR",
        "TEXT_TO_IMAGES_SIMILARITY":  "TextToImagesSimilarity",
        "IMAGE_TO_TEXTS_SIMILARITY":  "ImageToTextsSimilarity",
        "IMAGE_TO_IMAGES_SIMILARITY": "ImageToImagesSimilarity",
        "TERMINATE":                  "Terminate",
    }.items()
}


def _parse_action_tag(text: str):
    """
    Extract and normalise the action name from an <action>…</action> tag.
    Also detects bare 'Terminate' keyword (e.g. from chat-template tokens).
    Returns the internal uppercase-with-underscores name, or None.
    """
    match = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip().upper().replace("_", "").replace("ACTION", "")
        return _PROMPT_NAME_TO_INTERNAL.get(raw)

    # Fallback: detect bare 'Terminate' keyword emitted by the model via
    # chat-template tokens (e.g. <|im_start|>Terminate<|im_end|>)
    clean = re.sub(r"<\|[^|]+\|>", " ", text)  # strip <|…|> tokens
    if re.search(r"\bTerminate\b", clean, re.IGNORECASE):
        return "TERMINATE"

    return None


def _truncate_at_turn_end(text: str) -> str:
    """
    Truncate generated text at the first turn-end marker so that the model's
    reply for one turn does not bleed into role prefixes of the next turn.
    This is the primary defence against 'Thought: user...' corruption.
    """
    for marker in ("<|im_end|>", "<|endoftext|>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text


def _clean_special_tokens(text: str) -> str:
    """
    Remove any remaining Qwen chat-template special tokens (<|...|>) and
    strip surrounding whitespace.
    """
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


def _extract_thought_from_json(text: str) -> str:
    """
    If the model output is a JSON object with a 'thought' key, return the value.
    Otherwise return the original text.
    """
    # Try to extract JSON from the beginning of the text
    try:
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "thought" in data:
                return str(data["thought"]).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    return text


class Reasoner:
    """Inference engine for the trained Qwen2.5-VL Agent."""

    def __init__(
        self,
        model_path: str,
        base_model_path: str = None,
        device: str = "cuda",
        max_history_images: int = 100,
    ):
        """
        Args:
            model_path:          Path to the saved checkpoint directory.
            base_model_path:     Path to the original base model weights
                                 (used when model_path contains only adapter/head
                                 weights, not a merged backbone).
            device:              Torch device string.
            max_history_images:  Maximum images kept in the episode history.
        """
        self.device             = device
        self.max_history_images = max_history_images

        # Prefer merged_backbone if it exists inside the checkpoint dir
        merged_backbone_path = os.path.join(model_path, "merged_backbone")
        if os.path.isdir(merged_backbone_path):
            backbone_path = merged_backbone_path
            print(f"Found merged backbone at {backbone_path}; will use it directly.")
        else:
            backbone_path = base_model_path or model_path
        print(f"Checkpoint : {model_path}")
        print(f"Base model : {backbone_path}")

        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                config_data = json.load(f)
            agent_config = AgentConfig(**config_data.get("agent_config", {}))
        else:
            print("Warning: config.json not found, using AgentConfig defaults.")
            agent_config = AgentConfig()

        proc_path = (
            os.path.join(model_path, "processor")
            if os.path.isdir(os.path.join(model_path, "processor"))
            else backbone_path
        )
        print(f"Loading processor from {proc_path}...")
        self.processor = AutoProcessor.from_pretrained(
            proc_path,
            min_pixels=256  * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
        special_tokens = [
            "[EXECUTE]", "<textlist>", "</textlist>",
            "<text>", "</text>", "<action>", "</action>",
        ]
        self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.execute_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "[EXECUTE]"
        )
        print(f"[EXECUTE] token ID: {self.execute_token_id}")

        print(f"Loading model from {backbone_path}...")
        self.model = Qwen2_5_VL_Agent(
            model_path=backbone_path,
            agent_config=agent_config,
            execute_token_id=self.execute_token_id,
            lora_config=None,        # no LoRA at inference time
            freeze_backbone=False,
            vocab_size=len(self.processor.tokenizer),
        )

        # Load LoRA adapters only if no merged backbone was found
        lora_path = os.path.join(model_path, "lora_adapters")
        if os.path.isdir(lora_path) and not os.path.isdir(merged_backbone_path):
            print(f"Loading LoRA from {lora_path}...")
            self.model.backbone = PeftModel.from_pretrained(
                self.model.backbone, lora_path
            )
            # FIX: merge_and_unload() returns the merged model; must capture it.
            self.model.backbone = self.model.backbone.merge_and_unload()
            print("LoRA weights merged into backbone.")
        else:
            print("Using backbone as-is (already merged or no adapters found).")

        # Load agent head weights
        head_path = os.path.join(model_path, "agent_head.pth")
        if os.path.exists(head_path):
            print(f"Loading agent head from {head_path}...")
            state_dict = torch.load(head_path, map_location=device, weights_only=True)
            self.model.agent_head.load_state_dict(state_dict)
        else:
            print("Warning: agent_head.pth not found â€” head weights are random.")

        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully.")

    # Core prediction

    def predict_next_step(self, message_list: list, images: list) -> dict:
        """
        Generate a reasoning step: run the LLM to produce text and the agent
        head to predict action / box / image indices.

        Args:
            message_list: Conversation history (list of role/content dicts).
            images:       Current list of PIL images in the episode.

        Returns:
            Dict with keys:
              thought, action, action_id,
              box, boxes,
              image_index, image_index_2, image_indices_multi,
              text_args
        """
        if not images:
            images = [Image.new("RGB", (640, 480))]

        if len(images) > self.max_history_images:
            images = images[-self.max_history_images:]

        # Prepare inputs for backbone generation
        text_prompt = self.processor.apply_chat_template(
            message_list, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message_list)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
                
        # Generate the reasoning text
        with torch.no_grad():
            generated_ids = self.model.backbone.generate(
                **inputs,
                max_new_tokens=512,
                tokenizer=self.processor.tokenizer,
                stop_strings=["<|im_end|>"],
                do_sample=False,
            )

        input_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_len:]

        _stop_ids = set()
        for _tok in ("<|im_end|>", "<|endoftext|>"):
            _tid = self.processor.tokenizer.convert_tokens_to_ids(_tok)
            if isinstance(_tid, int) and _tid != self.processor.tokenizer.unk_token_id:
                _stop_ids.add(_tid)
        if _stop_ids:
            _token_list = new_tokens.tolist()
            for _pos, _tid in enumerate(_token_list):
                if _tid in _stop_ids:
                    new_tokens = new_tokens[:_pos]
                    break

        full_text_raw = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=False
        )
        # Strip any remaining <|...|> special-token strings
        full_text = _clean_special_tokens(full_text_raw)

        # Parse <action> tag from generated text (operates on raw text to catch
        # both <action>...</action> tags and bare chat-template Terminate tokens)
        text_action = _parse_action_tag(full_text_raw)

        # Check for [EXECUTE] token in the output
        exec_mask = new_tokens == self.execute_token_id
        if exec_mask.any():
            exec_new_pos = exec_mask.nonzero(as_tuple=False)[0, 0].item()
        
            gen_up_to_exec = new_tokens[: exec_new_pos + 1].unsqueeze(0)  # [1, n]
        
            valid_seq  = torch.cat([inputs.input_ids, gen_up_to_exec], dim=1)
            valid_mask = torch.cat(
                [inputs.attention_mask, torch.ones_like(gen_up_to_exec)], dim=1
            )
        
            with torch.no_grad():
                head_outputs = self.model.backbone(
                    input_ids=valid_seq,
                    attention_mask=valid_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    output_hidden_states=True,
                )
            feat = head_outputs.hidden_states[-1][:, -1:, :]  # [1, 1, H]

            (
                act_logits,
                box_preds,
                img1_logits,
                img2_logits,
                img_multi_logits,
            ) = self.model.agent_head(memory=feat, num_valid_images=len(images))

            # Resolve action: explicit <action> tag takes priority over head
            # prediction to stay consistent with the training objective.
            head_action_id = act_logits.argmax(-1).item()
            if text_action is not None:
                final_action_id = ACTION_TO_ID.get(text_action, head_action_id)
            else:
                final_action_id = head_action_id

            # All box predictions (normalised cxcywh), clamped to [0, 1]
            all_boxes = [
                [max(0.0, min(1.0, v)) for v in box_preds[0, i].tolist()]
                for i in range(box_preds.shape[1])
            ]

            img_idx_1 = min(img1_logits.argmax(-1).item(), len(images) - 1)
            img_idx_2 = min(img2_logits.argmax(-1).item(), len(images) - 1)

            multi_probs        = torch.sigmoid(img_multi_logits[0])
            img_indices_multi  = (multi_probs > 0.5).nonzero(as_tuple=True)[0].tolist()
            img_indices_multi  = [i for i in img_indices_multi if i < len(images)]
            if not img_indices_multi:
                img_indices_multi = list(range(len(images)))

            return {
                "thought":             full_text,
                "action":              ID_TO_ACTION.get(final_action_id, "TEXT"),
                "action_id":           final_action_id,
                "box":                 all_boxes[0] if all_boxes else [0.5, 0.5, 0.1, 0.1],
                "boxes":               all_boxes,
                "image_index":         img_idx_1,
                "image_index_2":       img_idx_2,
                "image_indices_multi": img_indices_multi,
                "text_args":           self._parse_text_args(full_text),
            }

        # No [EXECUTE] token found.
        # Still respect any <action> tag or bare Terminate keyword in the text.
        fallback_action    = text_action if text_action else "TEXT"
        fallback_action_id = ACTION_TO_ID.get(fallback_action, 4)
        return {
            "thought":             full_text,
            "action":              fallback_action,
            "action_id":           fallback_action_id,
            "box":                 [0.5, 0.5, 0.1, 0.1],
            "boxes":               [],
            "image_index":         0,
            "image_index_2":       0,
            "image_indices_multi": list(range(len(images))),
            "text_args":           self._parse_text_args(full_text),
        }

    # Text argument parsing

    def _parse_text_args(self, text: str) -> list[str]:
        """Extract arguments from <text>â€¦</text> or <textlist>â€¦</textlist>."""
        m = re.search(r"<text>(.*?)</text>", text, re.DOTALL)
        if m:
            return [m.group(1).strip()]
        m = re.search(r"<textlist>(.*?)</textlist>", text, re.DOTALL)
        if m:
            return [item.strip() for item in m.group(1).split(",") if item.strip()]
        return []

    # Box conversion

    def _denormalize_box(
        self, norm_box: list[float], img_w: int, img_h: int
    ) -> list[float]:
        """
        Normalised (cx, cy, w, h) â†’ pixel (x1, y1, x2, y2), clamped to bounds.
        """
        cx, cy, w, h = norm_box
        x1 = max(0, min(img_w, (cx - w / 2) * img_w))
        y1 = max(0, min(img_h, (cy - h / 2) * img_h))
        x2 = max(0, min(img_w, (cx + w / 2) * img_w))
        y2 = max(0, min(img_h, (cy + h / 2) * img_h))
        return [x1, y1, x2, y2]

    # Parameter building

    def _build_tool_parameters(self, prediction: dict, images: list) -> dict:
        """
        Translate the raw prediction dict into the parameter format expected
        by each tool function.
        """
        action  = prediction["action"]
        img_idx = min(prediction.get("image_index", 0), len(images) - 1)
        img_w, img_h = images[img_idx].size if images else (640, 480)

        if action == "GROUNDING":
            return {"image_index": img_idx,
                    "text": prediction.get("text_args", [])}

        if action == "DEPTH":
            return {"image_index": img_idx}

        if action in ("ZOOMIN", "CROP"):
            x1, y1, x2, y2 = self._denormalize_box(prediction["box"], img_w, img_h)
            params = {
                "image_index": img_idx,
                "bounding_box": {
                    "x_min": int(x1), "y_min": int(y1),
                    "x_max": int(x2), "y_max": int(y2),
                },
            }
            if action == "CROP":
                params["padding"] = 0.05
            return params

        if action == "VISUALSEARCH":
            return {"image_index": img_idx,
                    "objects": prediction.get("text_args", [])}

        if action == "SEGMENT":
            bboxes = []
            for nb in prediction.get("boxes", [prediction["box"]]):
                x1, y1, x2, y2 = self._denormalize_box(nb, img_w, img_h)
                bboxes.append({
                    "x_min": int(x1), "y_min": int(y1),
                    "x_max": int(x2), "y_max": int(y2),
                })
            return {"image_index": img_idx, "bounding_boxes": bboxes}

        if action == "OCR":
            return {"image_index": img_idx, "engine": "easyocr"}

        if action == "TEXT_TO_IMAGES_SIMILARITY":
            text_args = prediction.get("text_args", [])
            return {
                "text": text_args[0] if text_args else "",
                "image_indices": prediction.get(
                    "image_indices_multi", list(range(len(images)))
                ),
            }

        if action == "IMAGE_TO_TEXTS_SIMILARITY":
            return {"image_index": img_idx,
                    "texts": prediction.get("text_args", [])}

        if action == "IMAGE_TO_IMAGES_SIMILARITY":
            others = [
                i for i in prediction.get(
                    "image_indices_multi", list(range(len(images)))
                )
                if i != img_idx
            ]
            if not others:
                others = [i for i in range(len(images)) if i != img_idx]
            return {
                "reference_image_index": img_idx,
                "other_image_indices": others,
            }

        if action == "OVERLAY":
            x1, y1, x2, y2 = self._denormalize_box(prediction["box"], img_w, img_h)
            return {
                "background_image_index": img_idx,
                "overlay_image_index": prediction.get("image_index_2", 0),
                "overlay_proportion": {
                    "x_min": x1 / img_w, "y_min": y1 / img_h,
                    "x_max": x2 / img_w, "y_max": y2 / img_h,
                },
            }

        if action == "TERMINATE":
            text_args = prediction.get("text_args", [])
            thought   = prediction.get("thought", "")
            if text_args:
                final = text_args[0]
            else:
                # Try to extract meaningful answer from JSON-format thought
                # (model sometimes outputs {"thought": "..."} without <text> tags)
                final = _extract_thought_from_json(thought)
            return {"final_response": final}

        return {}

    # Tool execution

    def _execute_tool(
        self, prediction: dict, images: list, verbose: bool = True
    ) -> tuple:
        """
        Execute the tool corresponding to the predicted action.

        Args:
            prediction: Output from predict_next_step.
            images:     Mutable list of PIL images; new images are appended here.
            verbose:    Print execution details.

        Returns:
            (obs_text: str, new_image: PIL.Image or None)
        """
        action     = prediction["action"]
        params     = self._build_tool_parameters(prediction, images)
        obs_text   = "Action executed."
        new_image  = None

        img_idx    = min(
            params.get("image_index", prediction.get("image_index", 0)),
            len(images) - 1,
        )
        target_img = images[img_idx]
        img_w, img_h = target_img.size

        try:
            if action == "GROUNDING":
                labels_list = params.get("text", [])
                if not labels_list:
                    obs_text = "No labels provided for grounding."
                else:
                    result_images, boxes, labels = grounded_segmentation(
                        image=target_img, labels=labels_list, polygon_refinement=True
                    )
                    if boxes:
                        box_str = "\n".join(
                            f"{lbl}: {box}" for lbl, box in zip(labels, boxes)
                        )
                        obs_text = f"Grounding: found {len(boxes)} object(s).\n{box_str}"
                        if result_images:
                            new_image = result_images[0]
                            images.append(new_image)
                            obs_text += f"\nMarked image added (Index: {len(images)-1})"
                    else:
                        obs_text = "Grounding: no objects found."
                    if verbose:
                        print(f"  Grounding: {len(boxes)} object(s): {labels}")

            elif action == "DEPTH":
                depth_maps = get_depth(image=target_img)
                if depth_maps:
                    new_image = depth_maps[0]
                    images.append(new_image)
                    obs_text = f"Depth map added (Index: {len(images)-1})"
                else:
                    obs_text = "Depth estimation failed."
                if verbose:
                    print("  Depth map created.")

            elif action == "ZOOMIN":
                bbox = params.get("bounding_box")
                if bbox:
                    result = zoom_in_image_by_bbox(image=target_img, bounding_box=bbox)
                    if result:
                        new_image = result[0]
                        images.append(new_image)
                        # FIX: was `bl` (undefined variable); corrected to `bbox`
                        obs_text = f"Zoomed into {bbox}. New image (Index: {len(images)-1})"
                    else:
                        obs_text = "Zoom failed."
                else:
                    obs_text = "No bounding box for zoom."
                if verbose:
                    print(f"  ZoomIn: {bbox}")

            elif action == "VISUALSEARCH":
                objects = params.get("objects", [])
                if not objects:
                    obs_text = "No objects specified for visual search."
                else:
                    result_images, boxes_list, labels_list = visual_search(
                        image=target_img, objects=objects
                    )
                    total    = sum(len(b) for b in boxes_list)
                    obs_text = f"VisualSearch: found {total} match(es)."
                    for i, patch in enumerate(result_images):
                        images.append(patch)
                        obs_text += f"\nPatch {i} (Index: {len(images)-1})"
                    new_image = result_images[0] if result_images else None
                    if verbose:
                        print(
                            f"  VisualSearch: {total} matches "
                            f"in {len(result_images)} patches."
                        )

            elif action == "CROP":
                bbox = params.get("bounding_box")
                if bbox:
                    result = crop_image_action(
                        image=target_img,
                        bounding_box=bbox,
                        padding=params.get("padding", 0.05),
                    )
                    if result:
                        new_image = result[0]
                        images.append(new_image)
                        # FIX: was `bl` (undefined variable); corrected to `bbox`
                        obs_text = f"Cropped {bbox}. New image (Index: {len(images)-1})"
                    else:
                        obs_text = "Crop failed."
                else:
                    obs_text = "No bounding box for crop."

            elif action == "SEGMENT":
                bboxes = params.get("bounding_boxes", [])
                if bboxes:
                    result = segment_image(image=target_img, bounding_boxes=bboxes)
                    if result:
                        new_image = result[0]
                        images.append(new_image)
                        obs_text = (
                            f"Segmented {len(bboxes)} region(s). "
                            f"New image (Index: {len(images)-1})"
                        )
                    else:
                        obs_text = "Segmentation failed."
                else:
                    obs_text = "No bounding boxes for segmentation."

            elif action == "OCR":
                result   = ocr_extract_text(
                    image=target_img, engine=params.get("engine", "easyocr")
                )
                text_out = result.get("text", "")
                obs_text = f"OCR result:\n{text_out[:500]}"
                if verbose:
                    print(f"  OCR: {text_out[:100]}")

            elif action == "TEXT_TO_IMAGES_SIMILARITY":
                query   = params.get("text", "")
                indices = params.get("image_indices", list(range(len(images))))
                if query and indices:
                    selected = [images[i] for i in indices if i < len(images)]
                    result   = calculate_text_to_images_similarity(
                        text=query, images=selected
                    )
                    scores = result.get("similarity_scores", [])
                    best   = result.get("best_match_index", 0)
                    obs_text = (
                        f"Textâ†’Images similarity. Query: '{query}'\n"
                        f"Scores: {scores}\nBest match: Image {indices[best]}"
                    )
                    if verbose:
                        print(f"  TextToImages: best match Image {indices[best]}")
                else:
                    obs_text = "Missing text query or image indices."

            elif action == "IMAGE_TO_TEXTS_SIMILARITY":
                texts = params.get("texts", [])
                if texts:
                    result = calculate_image_to_texts_similarity(
                        image=target_img, texts=texts
                    )
                    scores = result.get("similarity_scores", [])
                    best   = result.get("best_match_index", 0)
                    obs_text = (
                        f"Imageâ†’Texts similarity.\n"
                        f"Scores: {scores}\nBest text: '{texts[best]}'"
                    )
                    if verbose:
                        print(f"  ImageToTexts: best '{texts[best]}'")
                else:
                    obs_text = "No text candidates provided."

            elif action == "IMAGE_TO_IMAGES_SIMILARITY":
                ref_idx = params.get("reference_image_index", 0)
                others  = params.get("other_image_indices", [])
                if others:
                    ref_img    = images[ref_idx] if ref_idx < len(images) else target_img
                    other_imgs = [images[i] for i in others if i < len(images)]
                    result     = calculate_image_to_images_similarity(
                        reference_image=ref_img, other_images=other_imgs
                    )
                    scores = result.get("similarity_scores", [])
                    best   = result.get("best_match_index", 0)
                    obs_text = (
                        f"Image to Images similarity.\n"
                        f"Scores: {scores}\nBest match: Image {others[best]}"
                    )
                    if verbose:
                        print(f"  ImageToImages: best Image {others[best]}")
                else:
                    obs_text = "Not enough images for similarity comparison."

            elif action == "OVERLAY":
                bg   = params.get("background_image_index", 0)
                ol   = params.get("overlay_image_index", 0)
                prop = params.get("overlay_proportion", {"x_min": 0, "y_min": 0,
                                                         "x_max": 1, "y_max": 1})
                if bg < len(images) and ol < len(images):
                    result = overlay(
                        background_image=images[bg],
                        overlay_image=images[ol],
                        overlay_proportion=prop,
                    )
                    if result:
                        new_image = result[0]
                        images.append(new_image)
                        obs_text = f"Overlay created. New image (Index: {len(images)-1})"
                    else:
                        obs_text = "Overlay failed."
                else:
                    obs_text = "Invalid image indices for overlay."

            elif action == "TEXT":
                text_content = prediction.get("text_args", [])
                obs_text = text_content[0] if text_content else "Empty text action."

            elif action == "TERMINATE":
                obs_text = f"Terminated. Answer: {params.get('final_response', '')}"

            else:
                obs_text = f"Unknown action: {action}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            obs_text = f"Tool execution error ({action}): {e}"

        return obs_text, new_image

    def run_task(
        self,
        image_path,
        prompt: str,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> str:
        """
        Run the full visual reasoning loop on a single task.

        Args:
            image_path: Path string or PIL Image.
            prompt:     The user's question / task.
            max_steps:  Maximum number of reasoning steps.
            verbose:    Print step-by-step progress.

        Returns:
            Final answer string.
        """
        img    = (
            Image.open(image_path).convert("RGB")
            if isinstance(image_path, str)
            else image_path
        )
        images = [img]

        system_prompt = load_vts_system_prompt() + "\n" + load_model_response_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": "Image Index: 0\n"},
                    {"type": "image", "image": img},
                    {"type": "text",  "text": prompt},
                ],
            },
        ]

        if verbose:
            print(f"\n{'='*70}\nTask: {prompt}\n{'='*70}\n")

        for step in range(max_steps):
            if verbose:
                print(f"--- Step {step + 1}/{max_steps} ---")

            if step == max_steps - 1:
                messages.append({
                    "role": "user",
                    "content": (
                        "Maximum steps reached. Provide the final answer "
                        "immediately using <action>Terminate</action>[EXECUTE]."
                    ),
                })

            prediction = self.predict_next_step(messages, images)

            # Force termination on the very last step
            if step == max_steps - 1 and prediction["action"] != "TERMINATE":
                prediction["action"]    = "TERMINATE"
                prediction["action_id"] = 12
                if not prediction["text_args"]:
                    # Extract a clean answer from JSON thought if possible,
                    # rather than using the raw full_text with special tokens.
                    clean_thought = _extract_thought_from_json(prediction["thought"])
                    prediction["text_args"] = [clean_thought]

            if verbose:
                print(f"  Thought: {prediction['thought'][:120]}...")
                print(f"  Action:  {prediction['action']}")

            if prediction["action"] == "TERMINATE":
                params = self._build_tool_parameters(prediction, images)
                answer = params.get("final_response",
                                    prediction.get("thought", ""))
                if verbose:
                    print(f"\nTask complete. Answer: {answer}")
                return answer

            obs_text, new_image = self._execute_tool(prediction, images, verbose)

            user_content: list = []
            if new_image is not None:
                user_content.append({
                    "type": "text",
                    "text": f"{obs_text}\nImage Index: {len(images)-1}\n",
                })
                user_content.append({"type": "image", "image": new_image})
            else:
                user_content.append({"type": "text", "text": obs_text})

            messages.append({"role": "assistant", "content": prediction["thought"]})
            messages.append({"role": "user",      "content": user_content})

        return "Task incomplete: maximum steps reached."

    # Batch inference

    def batch_inference(
        self,
        task_list: list[dict],
        max_steps: int = 10,
        verbose: bool = False,
    ) -> list[dict]:
        """
        Run inference on a list of tasks.

        Args:
            task_list:  List of dicts, each with 'image_path' and 'prompt'.
            max_steps:  Max reasoning steps per task.
            verbose:    Print progress.

        Returns:
            List of result dicts with 'task_id', 'answer', 'success'.
        """
        results = []
        for i, task in enumerate(task_list):
            if verbose:
                print(f"\n{'='*60}\nTask {i+1}/{len(task_list)}\n{'='*60}")
            try:
                answer = self.run_task(
                    image_path=task["image_path"],
                    prompt=task["prompt"],
                    max_steps=max_steps,
                    verbose=verbose,
                )
                results.append({"task_id": i, "answer": answer, "success": True})
            except Exception as e:
                print(f"Task {i} failed: {e}")
                results.append({
                    "task_id": i, "answer": None,
                    "success": False, "error": str(e),
                })
        return results