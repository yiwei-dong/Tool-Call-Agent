"""
Dataset Module for Qwen2.5-VL Agent Training.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from utils.heads import AgentConfig
import re

_default_cfg = AgentConfig()
MAX_BOXES   = _default_cfg.max_boxes
MAX_IMAGES  = _default_cfg.max_history_images

ID_TO_ACTION = {
    0: "Grounding", 1: "Depth", 2: "ZoomIn", 3: "VisualSearch",
    4: "Text", 5: "Overlay", 6: "Crop", 7: "Segment",
    8: "OCR", 9: "TextToImagesSimilarity", 10: "ImageToTextsSimilarity",
    11: "ImageToImagesSimilarity", 12: "Terminate",
}


class ToolHeadsDataset(Dataset):
    """Dataset for training the Qwen2.5-VL Agent with tool head labels."""

    def __init__(self, jsonl_file, processor, image_root=".", max_length=None,
                 execute_token: str = "[EXECUTE]"):
        self.data       = []
        self.processor  = processor
        self.image_root = image_root
        self.max_length = max_length or getattr(processor, "model_max_length", 2048)

        # ── Cache special token IDs ───────────────────────────────────────
        tok = self.processor.tokenizer

        # FIX: Set truncation_side to "left" so that when a sequence exceeds
        # max_length, the OLD context at the START is removed rather than the
        # END. [EXECUTE] lives at the very end of the last assistant turn;
        # default right-side truncation silently removes it, causing is_exec.any()
        # to return False and all head losses to be permanently 0.
        tok.truncation_side = "left"
        print(f"Tokenizer truncation_side set to 'left' (protects [EXECUTE] at end).")

        self._execute_token_id = tok.convert_tokens_to_ids(execute_token)
        if self._execute_token_id == tok.unk_token_id:
            raise ValueError(
                f"'{execute_token}' was not found in the tokenizer vocabulary. "
                "Call processor.tokenizer.add_tokens(['{execute_token}'], "
                "special_tokens=True) before constructing the dataset."
            )

        self._im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id   = tok.convert_tokens_to_ids("<|im_end|>")
        # Tokenise "assistant" to detect role-turn boundaries for label masking
        self._asst_ids    = tok.encode("assistant", add_special_tokens=False)
        # Newline token IDs (can be >1 token in some tokenizer versions)
        self._newline_ids = set(tok.encode("\n", add_special_tokens=False))

        self._skipped = 0   # tracks samples where [EXECUTE] was truncated away

        print(f"Loading dataset from {jsonl_file}...")
        self._load_data(jsonl_file)
        print(f"Loaded {len(self.data)} samples (max_length={self.max_length}).")

    # ─────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_data(self, jsonl_file):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    session = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                    continue

                images   = session.get("images", [])
                messages = session.get("messages", [])

                for i, msg in enumerate(messages):
                    if msg["role"] == "assistant" and "head_label" in msg:
                        self.data.append(
                            {
                                "messages":   messages[: i + 1],
                                "images":     images,
                                "head_label": msg["head_label"],
                                "session_id": line_num,
                            }
                        )

    def __len__(self):
        return len(self.data)

    # ─────────────────────────────────────────────────────────────────────
    # Image loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_images(self, image_paths):
        loaded, sizes = [], []
        for p in image_paths:
            # FIX: if path is already absolute, os.path.join leaves it intact.
            # If it's relative we join with image_root.
            full_path = p if os.path.isabs(p) else os.path.join(self.image_root, p)
            try:
                img = Image.open(full_path).convert("RGB")
                loaded.append(img)
                sizes.append(img.size)   # (width, height)
            except Exception as e:
                print(f"Warning: Cannot load {full_path}: {e}. Using blank fallback.")
                fallback = Image.new("RGB", (640, 480))
                loaded.append(fallback)
                sizes.append((640, 480))
        return loaded, sizes

    # ─────────────────────────────────────────────────────────────────────
    # Message preparation
    # ─────────────────────────────────────────────────────────────────────

    def _prepare_messages_with_images(self, messages, loaded_images):
        """Return a copy of messages with PIL images prepended to the first user turn."""
        messages_copy = [msg.copy() for msg in messages]
        if messages_copy and messages_copy[0]["role"] == "user":
            content = [{"type": "image", "image": img} for img in loaded_images]
            original = messages_copy[0]["content"]
            if isinstance(original, str):
                content.append({"type": "text", "text": original})
            elif isinstance(original, list):
                content.extend(original)
            messages_copy[0]["content"] = content
        return messages_copy

    def _inject_action_tag(self, messages, action_id):
        action_name = ID_TO_ACTION.get(action_id, "Terminate")
        correct_tag = f"<action>{action_name}</action>"
        
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return messages

        content = last_msg["content"]

        def apply_injection(text_str):
            if "[EXECUTE]" not in text_str:
                return f"{text_str}\n{correct_tag}[EXECUTE]"
            
            parts = text_str.split("[EXECUTE]", 1)
            before_exec = parts[0]
            after_exec = parts[1]
            
            before_exec_clean = re.sub(r"<action>.*?</action>", "", before_exec, flags=re.IGNORECASE).strip()
            
            return f"{before_exec_clean} {correct_tag}{after_exec}[EXECUTE]"

        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    part["text"] = apply_injection(part["text"])
        elif isinstance(content, str):
            last_msg["content"] = apply_injection(content)
            
        return messages

    # ─────────────────────────────────────────────────────────────────────
    # Label masking (critical SFT fix)
    # ─────────────────────────────────────────────────────────────────────

    def _mask_non_assistant_labels(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        For SFT we must only compute LM loss on the assistant's responses.

        Strategy: scan for  <|im_start|> assistant \\n … <|im_end|>  spans
        and copy labels only inside those regions. Everything else → -100.

        Robust to left-truncated sequences (which may start mid-turn): if the
        very first token is NOT <|im_start|> we skip ahead until we find a
        proper turn boundary, so we never accidentally label a partial user turn
        as an assistant turn.

        Also handles the edge case where a sample has no assistant turn at all
        (returns all -100 rather than training on garbage).
        """
        result = torch.full_like(labels, -100)
        ids    = input_ids.tolist()
        n      = len(ids)
        n_asst = len(self._asst_ids)

        found_any = False
        i = 0
        while i < n:
            # Look for <|im_start|> followed by the "assistant" token sequence
            if ids[i] != self._im_start_id:
                i += 1
                continue

            # Check the role tokens that follow
            role_end = i + 1 + n_asst
            if role_end > n:
                break
            if ids[i + 1: role_end] != self._asst_ids:
                # This is a non-assistant turn (<|im_start|>user, <|im_start|>system …)
                # Skip forward until we find the matching <|im_end|>
                j = role_end
                while j < n and ids[j] != self._im_end_id:
                    j += 1
                i = j + 1
                continue

            # ── Found an assistant turn ───────────────────────────────────
            found_any = True
            # Skip past "<|im_start|> assistant" + any leading newlines
            j = role_end
            while j < n and ids[j] in self._newline_ids:
                j += 1

            # Copy label tokens until the closing <|im_end|>
            while j < n and ids[j] != self._im_end_id:
                result[j] = labels[j]
                j += 1

            # Include <|im_end|> in the loss (teaches model when to stop)
            if j < n:
                result[j] = labels[j]
                j += 1

            i = j

        if not found_any:
            # This can happen if truncation removed all assistant turns.
            # Return all -100 (no LM loss for this sample, head loss only).
            pass

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Box target processing
    # ─────────────────────────────────────────────────────────────────────

    def _process_box_targets(self, head_label, img_sizes):
        target_boxes      = torch.zeros((MAX_BOXES, 4), dtype=torch.float)
        target_boxes_mask = torch.zeros(MAX_BOXES,       dtype=torch.float)

        raw_boxes = head_label.get("box_targets", [])
        if not raw_boxes and "box" in head_label:
            b = head_label["box"]
            if b and len(b) == 4 and sum(b) > 0:
                raw_boxes = [b]

        if not raw_boxes:
            return target_boxes, target_boxes_mask

        try:
            ref_idx = int(head_label.get("image_index", 0))
        except (ValueError, TypeError):
            ref_idx = 0
        ref_idx = max(0, min(ref_idx, len(img_sizes) - 1))
        img_w, img_h = img_sizes[ref_idx]

        for i, box in enumerate(raw_boxes):
            if i >= MAX_BOXES:
                break
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            w, h   = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            target_boxes[i] = torch.tensor(
                [
                    max(0.0, min(1.0, cx / img_w)),
                    max(0.0, min(1.0, cy / img_h)),
                    max(0.0, min(1.0, w  / img_w)),
                    max(0.0, min(1.0, h  / img_h)),
                ]
            )
            target_boxes_mask[i] = 1.0

        if head_label.get("box_mask", 1) == 0:
            target_boxes_mask[:] = 0.0

        return target_boxes, target_boxes_mask

    # ─────────────────────────────────────────────────────────────────────
    # __getitem__
    # ─────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        item        = self.data[idx]
        image_paths = item["images"]
        head_label  = item["head_label"]

        # 1. Load images
        loaded_images, img_sizes = self._load_images(image_paths)

        # 2. Prepare messages (deep copy, then inject action tag)
        messages  = self._prepare_messages_with_images(item["messages"], loaded_images)
        action_id = head_label.get("action_id", 12)
        if not (0 <= action_id < _default_cfg.num_actions):
            print(
                f"Warning: Invalid action_id {action_id} at idx {idx},"
                " defaulting to Terminate (12)."
            )
            action_id = 12
        messages = self._inject_action_tag(messages, action_id)

        # 3. Tokenise
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
            # truncation_side="left" is already set on the tokenizer in __init__
        )

        input_ids      = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)

        # 3b. Verify [EXECUTE] survived truncation.
        # Even with left-side truncation an extremely short max_length can
        # still lose [EXECUTE]. Log a warning so the user can detect it.
        if not (input_ids == self._execute_token_id).any():
            self._skipped += 1
            if self._skipped <= 10 or self._skipped % 500 == 0:
                print(
                    f"Warning: [EXECUTE] missing after truncation "
                    f"(idx={idx}, total so far: {self._skipped}). "
                    f"Head losses will be 0 for this sample. "
                    f"Consider increasing max_length (current={self.max_length})."
                )

        # 4. LM labels — pad positions → -100, then restrict to assistant turns.
        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        labels = self._mask_non_assistant_labels(input_ids, labels)

        # 5. Image index labels
        try:
            tgt_img1 = int(head_label.get("image_index", -100))
        except (ValueError, TypeError):
            tgt_img1 = -100
        if tgt_img1 >= len(loaded_images):
            tgt_img1 = -100

        try:
            tgt_img2 = int(head_label.get("image_index_2", -100))
        except (ValueError, TypeError):
            tgt_img2 = -100
        if tgt_img2 >= len(loaded_images):
            tgt_img2 = -100

        # 6. Multi-hot image label
        tgt_img_multi = torch.zeros(MAX_IMAGES, dtype=torch.float)
        for mi in head_label.get("image_indices", []):
            if 0 <= mi < MAX_IMAGES:
                tgt_img_multi[mi] = 1.0

        # 7. Box targets
        tgt_boxes, tgt_boxes_mask = self._process_box_targets(head_label, img_sizes)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "labels":          labels,
            "pixel_values":    inputs.get("pixel_values"),
            "image_grid_thw":  inputs.get("image_grid_thw"),
            "tgt_action":      torch.tensor(action_id, dtype=torch.long),
            "tgt_img1":        torch.tensor(tgt_img1, dtype=torch.long),
            "tgt_img2":        torch.tensor(tgt_img2, dtype=torch.long),
            "tgt_img_multi":   tgt_img_multi,
            "tgt_boxes":       tgt_boxes,
            "tgt_boxes_mask":  tgt_boxes_mask,
            "num_valid_images": len(loaded_images),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def custom_collate_fn(batch):
    """Collate a list of dataset items into a training batch."""
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"]  for b in batch])
    labels         = torch.stack([b["labels"]          for b in batch])

    pvs         = [b["pixel_values"]   for b in batch if b["pixel_values"]   is not None]
    pixel_values = torch.cat(pvs, dim=0) if pvs else None

    gthws           = [b["image_grid_thw"] for b in batch if b["image_grid_thw"] is not None]
    image_grid_thw  = torch.cat(gthws, dim=0) if gthws else None

    targets = {
        "action_ids":       torch.stack([b["tgt_action"]    for b in batch]),
        "img1_labels":      torch.stack([b["tgt_img1"]      for b in batch]),
        "img2_labels":      torch.stack([b["tgt_img2"]      for b in batch]),
        "img_multi_labels": torch.stack([b["tgt_img_multi"] for b in batch]),
        "box_targets":      torch.stack([b["tgt_boxes"]     for b in batch]),
        "box_masks":        torch.stack([b["tgt_boxes_mask"]for b in batch]),
        "num_valid_images": torch.tensor(
            [b["num_valid_images"] for b in batch], dtype=torch.long
        ),
    }

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "pixel_values":   pixel_values,
        "image_grid_thw": image_grid_thw,
        "targets":        targets,
    }