"""
Reasoner Module - FULLY FIXED VERSION
Improved inference and bounds checking.
"""
import torch
import re
import os
import json
from copy import deepcopy
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from agent_model import Qwen2_5_VL_Agent
from utils.heads import AgentConfig
from utils.tools import (
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
    """Inference engine for the trained Qwen2.5-VL Agent."""
    
    def __init__(self, model_path, head_path=None, device="cuda", max_history_images=10):
        """
        Args:
            model_path: Path to trained model checkpoint
            head_path: Optional separate path to agent head weights
            device: Device to run on
            max_history_images: Maximum number of images to keep in history
        """
        self.device = device
        self.max_history_images = max_history_images

        # Load Processor
        print(f"Loading processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            min_pixels=256 * 28 * 28, 
            max_pixels=1280 * 28 * 28
        )
        
        special_tokens = ["[EXECUTE]", "<textlist>", "</textlist>", "<text>", "</text>"]
        num_added = self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.execute_token_id = self.processor.tokenizer.convert_tokens_to_ids("[EXECUTE]")
        
        print(f"Added {num_added} special tokens, [EXECUTE] ID: {self.execute_token_id}")

        # Load Model
        print(f"Loading model from {model_path}...")
        config = AgentConfig()
        self.model = Qwen2_5_VL_Agent(
            model_path, 
            config, 
            self.execute_token_id, 
            lora_config=None,
            freeze_backbone=False  # Inference mode
        )

        # Resize embeddings
        self.model.backbone.resize_token_embeddings(len(self.processor.tokenizer))

        # Load Head Weights if provided
        if head_path and os.path.exists(head_path):
            print(f"Loading agent head weights from {head_path}")
            state_dict = torch.load(head_path, map_location=device)
            self.model.agent_head.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()
        print("✅ Model loaded successfully!")

    def predict_next_step(self, message_list, images):
        """
        Generate thought and predict action for next step.
        
        Args:
            message_list: Conversation history
            images: List of PIL images
            
        Returns:
            Dict with 'thought', 'action', 'box', 'image_index', 'text_args'
        """
        # ✅ FIX: Validate image count
        if len(images) > self.max_history_images:
            print(f"⚠️  Warning: {len(images)} images > max {self.max_history_images}, truncating")
            images = images[-self.max_history_images:]
        
        # Prepare inputs
        text_prompt = self.processor.apply_chat_template(
            message_list, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message_list)

        try:
            inputs = self.processor(
                text=[text_prompt], 
                images=image_inputs, 
                videos=video_inputs,
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
        except Exception as e:
            print(f"❌ Processor failed: {e}")
            raise

        # Generate Thought
        with torch.no_grad():
            try:
                generated_ids = self.model.backbone.generate(
                    **inputs, 
                    max_new_tokens=512,
                    tokenizer=self.processor.tokenizer,
                    stop_strings=["<|im_end|>"],
                    do_sample=False
                )
            except Exception as e:
                print(f"❌ Generation failed: {e}")
                raise

        # Decode text
        input_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_len:]
        full_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Check for [EXECUTE]
        if self.execute_token_id in new_tokens:
            # Get position of [EXECUTE] token
            exec_positions = (generated_ids == self.execute_token_id).nonzero(as_tuple=True)
            if len(exec_positions[1]) > 0:
                exec_pos = exec_positions[1][0]
                valid_seq = generated_ids[:, :exec_pos + 1]

                # Run forward pass to get hidden states
                with torch.no_grad():
                    try:
                        outputs = self.model.backbone(
                            input_ids=valid_seq,
                            pixel_values=inputs.pixel_values,
                            image_grid_thw=inputs.image_grid_thw,
                            output_hidden_states=True
                        )
                        feat = outputs.hidden_states[-1][:, -1:, :]  # Last token feature

                        # Head Prediction
                        act_logits, box_preds, img1, img2, img_multi = self.model.agent_head(
                            memory=feat, 
                            num_valid_images=len(images)
                        )

                        action_id = act_logits.argmax(-1).item()
                        box = box_preds[0, 0].tolist()  # [cx, cy, w, h] normalized
                        
                        # ✅ FIX: Proper image index handling (0-indexed)
                        img_idx = img1.argmax(-1).item()
                        
                        # ✅ FIX: Bounds checking
                        if img_idx >= len(images):
                            print(f"⚠️  Warning: Predicted image index {img_idx} >= {len(images)}, "
                                  f"clamping to {len(images) - 1}")
                            img_idx = len(images) - 1
                        
                        # Validate box coordinates
                        box = [max(0.0, min(1.0, x)) for x in box]

                        return {
                            "thought": full_text,
                            "action": ACTION_ID_MAP.get(action_id, "TEXT"),
                            "action_id": action_id,
                            "box": box,  # Normalized [cx, cy, w, h]
                            "image_index": img_idx,  # 0-indexed
                            "text_args": self._parse_text(full_text)
                        }
                    
                    except Exception as e:
                        print(f"❌ Head prediction failed: {e}")
                        # Return text-only response
                        return {
                            "thought": full_text,
                            "action": "TEXT",
                            "action_id": 4,
                            "text_args": self._parse_text(full_text)
                        }

        # No [EXECUTE] token found
        return {
            "thought": full_text, 
            "action": "TEXT", 
            "action_id": 4,
            "text_args": []
        }

    def _parse_text(self, text):
        """Parse text arguments from generated response."""
        # Try <text>...</text> tag
        match = re.search(r'<text>(.*?)</text>', text, re.DOTALL)
        if match: 
            return [match.group(1).strip()]
        
        # Try <textlist>...</textlist> tag
        match = re.search(r'<textlist>(.*?)</textlist>', text, re.DOTALL)
        if match: 
            items = match.group(1).split(',')
            return [item.strip() for item in items if item.strip()]
        
        return []

    def _denormalize_box(self, norm_box, img_width, img_height):
        """
        Convert normalized box [cx, cy, w, h] to pixel coordinates [x1, y1, x2, y2].
        
        Args:
            norm_box: [cx, cy, w, h] in [0, 1]
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            [x1, y1, x2, y2] in pixels
        """
        cx, cy, w, h = norm_box
        
        # Convert to pixels
        cx_px = cx * img_width
        cy_px = cy * img_height
        w_px = w * img_width
        h_px = h * img_height
        
        # Convert to corners
        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        x2 = cx_px + w_px / 2
        y2 = cy_px + h_px / 2
        
        # Clamp to image bounds
        x1 = max(0, min(img_width, x1))
        y1 = max(0, min(img_height, y1))
        x2 = max(0, min(img_width, x2))
        y2 = max(0, min(img_height, y2))
        
        return [x1, y1, x2, y2]

    def run_task(self, image_path, prompt, max_steps=10, verbose=True):
        """
        High-level execution loop for visual reasoning tasks.
        
        Args:
            image_path: Path to input image
            prompt: User's question/task
            max_steps: Maximum reasoning steps
            verbose: Whether to print progress
            
        Returns:
            Final answer text
        """
        # Load initial image
        img = Image.open(image_path).convert("RGB")
        images = [img]

        # Initialize conversation
        messages = [
            {"role": "system", "content": "You are a visual reasoning agent."},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]}
        ]

        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 Task: {prompt}")
            print(f"{'='*70}\n")

        # Reasoning loop
        for step in range(max_steps):
            if verbose:
                print(f"--- Step {step + 1} ---")
            
            # Predict next action
            prediction = self.predict_next_step(messages, images)
            
            if verbose:
                print(f"💭 Thought: {prediction['thought'][:100]}...")
                print(f"🎯 Action: {prediction['action']}")
            
            # Check for termination
            if prediction['action'] == "TERMINATE":
                if verbose:
                    print(f"\n✅ Task completed!")
                return prediction.get('text_args', [prediction['thought']])[0] if prediction.get('text_args') else prediction['thought']

            # Execute tool (simplified)
            obs_text = "Action executed."
            new_image = None
            
            try:
                if prediction['action'] == "CROP" and 'box' in prediction:
                    # Get target image
                    img_idx = prediction.get('image_index', 0)
                    if img_idx < len(images):
                        target_img = images[img_idx]
                        w, h = target_img.size
                        
                        # Denormalize box
                        x1, y1, x2, y2 = self._denormalize_box(prediction['box'], w, h)
                        
                        # Crop
                        cropped = target_img.crop((int(x1), int(y1), int(x2), int(y2)))
                        images.append(cropped)
                        new_image = cropped
                        obs_text = f"Cropped region added to image history (now {len(images)} images)"
                        
                        if verbose:
                            print(f"   📸 Cropped box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                
                # Add more tool implementations here...
                
            except Exception as e:
                print(f"⚠️  Tool execution failed: {e}")
                obs_text = f"Tool execution failed: {str(e)}"

            # Update conversation
            messages.append({"role": "assistant", "content": prediction['thought']})
            
            if new_image is not None:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": obs_text},
                    {"type": "image", "image": new_image}
                ]})
            else:
                messages.append({"role": "user", "content": obs_text})

        if verbose:
            print(f"\n⚠️  Max steps ({max_steps}) reached without termination")
        
        return "Task incomplete - maximum steps reached."
