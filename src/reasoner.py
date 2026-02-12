import torch
import re
import os
import json
from copy import deepcopy
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from .agent_model import Qwen2_5_VL_Agent
from .utils.heads import AgentConfig
from .utils.tools import (
    grounded_segmentation, zoom_in_image_by_bbox, visual_search,
    ocr_extract_text, overlay, crop_image_action, segment_image,
    calculate_image_to_images_similarity
)

ACTION_ID_MAP = {
    0: "GROUNDING", 1: "DEPTH", 2: "ZOOMIN", 3: "VISUALSEARCH",
    4: "TEXT", 5: "OVERLAY", 6: "CROP", 7: "SEGMENT",
    8: "OCR", 9: "TEXT_TO_IMAGES_SIMILARITY", 10: "IMAGE_TO_TEXTS_SIMILARITY",
    11: "IMAGE_TO_IMAGES_SIMILARITY", 12: "TERMINATE"
}


class Reasoner:
    def __init__(self, model_path, head_path=None, device="cuda"):
        self.device = device

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        special_tokens = ["[EXECUTE]", "<textlist>", "</textlist>", "<text>", "</text>"]
        self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.execute_token_id = self.processor.tokenizer.convert_tokens_to_ids("[EXECUTE]")

        # Load Model
        config = AgentConfig()  # Should match training config
        self.model = Qwen2_5_VL_Agent(model_path, config, self.execute_token_id, lora_config=None)

        # Resize embeddings
        self.model.backbone.resize_token_embeddings(len(self.processor.tokenizer))

        # Load Head Weights if provided (trained separately)
        if head_path and os.path.exists(head_path):
            print(f"Loading Head weights from {head_path}")
            self.model.agent_head.load_state_dict(torch.load(head_path, map_location=device))

        self.model.to(device)
        self.model.eval()

    def predict_next_step(self, message_list, images):
        """Generates thought and predicts action"""
        # Prepare inputs
        text_prompt = self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message_list)

        inputs = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        # Generate Thought
        with torch.no_grad():
            generated_ids = self.model.backbone.generate(
                **inputs, max_new_tokens=512,
                tokenizer=self.processor.tokenizer,
                stop_strings=["<|im_end|>"],
                do_sample=False
            )

        # Decode text
        input_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_len:]
        full_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Check for [EXECUTE]
        if self.execute_token_id in new_tokens:
            # Run Head
            # Re-run forward pass on generated sequence up to [EXECUTE]
            exec_pos = (generated_ids == self.execute_token_id).nonzero(as_tuple=True)[1][0]
            valid_seq = generated_ids[:, :exec_pos + 1]

            with torch.no_grad():
                outputs = self.model.backbone(
                    input_ids=valid_seq,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,
                    output_hidden_states=True
                )
                feat = outputs.hidden_states[-1][:, -1:, :]  # Last token feature

                # Head Prediction
                act_logits, box_preds, img1, img2, _ = self.model.agent_head(
                    memory=feat, num_valid_images=len(images)
                )

                action_id = act_logits.argmax(-1).item()
                box = box_preds[0, 0].tolist()
                img_idx = img1.argmax(-1).item() + 1

                return {
                    "thought": full_text,
                    "action": ACTION_ID_MAP.get(action_id, "TEXT"),
                    "box": box,  # Normalized
                    "image_index": img_idx,
                    "text_args": self._parse_text(full_text)
                }

        return {"thought": full_text, "action": "TEXT", "text_args": []}

    def _parse_text(self, text):
        match = re.search(r'<text>(.*?)</text>', text, re.DOTALL)
        if match: return [match.group(1)]
        match = re.search(r'<textlist>(.*?)</textlist>', text, re.DOTALL)
        if match: return match.group(1).split(',')
        return []

    def run_task(self, image_path, prompt):
        """High level execution loop"""
        img = Image.open(image_path).convert("RGB")
        images = [img]

        messages = [
            {"role": "system", "content": "You are a visual reasoning agent."},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]}
        ]

        for step in range(10):  # Max steps
            print(f"--- Step {step} ---")
            prediction = self.predict_next_step(messages, images)
            print(f"Thought: {prediction['thought']}")
            print(f"Action: {prediction['action']}")

            if prediction['action'] == "TERMINATE":
                return prediction['thought']

            # Execute Tool (Simplified Logic)
            obs_text = "Action executed."
            if prediction['action'] == "CROP":
                # Convert Box
                w, h = images[prediction['image_index'] - 1].size
                bx, by, bw, bh = prediction['box']
                # Decode from center format to x1y1x2y2
                x1 = (bx - bw / 2) * w
                y1 = (by - bh / 2) * h
                x2 = (bx + bw / 2) * w
                y2 = (by + bh / 2) * h

                new_imgs = crop_image_action(images[prediction['image_index'] - 1],
                                             {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2})
                if new_imgs:
                    images.extend(new_imgs)
                    obs_text = "Image cropped and added to history."
                    # Add to history
                    messages.append({"role": "assistant", "content": prediction['thought']})
                    messages.append({"role": "user", "content": [
                        {"type": "text", "text": obs_text},
                        {"type": "image", "image": new_imgs[0]}
                    ]})
                    continue

            # Default: Just append thought if no tool logic matched
            messages.append({"role": "assistant", "content": prediction['thought']})
            if prediction['action'] != "TEXT":
                messages.append({"role": "user", "content": obs_text})

        return "Max steps reached."