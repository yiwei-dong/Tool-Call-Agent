"""
Dataset Module - FULLY FIXED VERSION
All critical, strongly recommended, and nice-to-have fixes applied.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from utils.heads import AgentConfig

MAX_BOXES = AgentConfig.max_boxes
MAX_IMAGES = AgentConfig.max_history_images


class ToolHeadsDataset(Dataset):
    """Dataset for training Qwen2.5-VL Agent with tool head labels."""
    
    def __init__(self, jsonl_file, processor, image_root=".", max_length=None):
        """
        Args:
            jsonl_file: Path to JSONL file with training data
            processor: Qwen2.5-VL processor
            image_root: Root directory for images
            max_length: Maximum sequence length (defaults to processor's max length)
        """
        self.data = []
        self.processor = processor
        self.image_root = image_root
        # ✅ FIX: Make max_length configurable
        self.max_length = max_length or getattr(processor, 'model_max_length', 2048)
        
        # ✅ NEW: Statistics tracking
        self.stats = {
            'total_sessions': 0,
            'valid_samples': 0,
            'failed_parses': 0,
            'samples_without_head_label': 0
        }

        print(f"Loading dataset from {jsonl_file}...")
        self._load_data(jsonl_file)
        self._print_statistics()

    def _load_data(self, jsonl_file):
        """✅ NEW: Separated data loading logic"""
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip(): 
                        continue
                    
                    self.stats['total_sessions'] += 1
                    
                    try:
                        session = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.stats['failed_parses'] += 1
                        print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                        continue

                    images = session.get("images", [])
                    messages = session.get("messages", [])
                    
                    if not messages:
                        continue

                    # Extract training samples with head_label
                    for i, msg in enumerate(messages):
                        if msg['role'] == 'assistant' and 'head_label' in msg:
                            self.data.append({
                                "messages": messages[:i + 1],  # Include current assistant msg
                                "images": images,
                                "head_label": msg['head_label'],
                                "session_id": line_num  # ✅ NEW: Track source
                            })
                            self.stats['valid_samples'] += 1
                        elif msg['role'] == 'assistant':
                            self.stats['samples_without_head_label'] += 1

        except FileNotFoundError:
            print(f"❌ Error: Data file {jsonl_file} not found.")
            raise
        except Exception as e:
            print(f"❌ Unexpected error loading data: {e}")
            raise

    def _print_statistics(self):
        """✅ NEW: Print dataset statistics"""
        print(f"\n{'='*60}")
        print(f"Dataset Statistics:")
        print(f"  Total sessions parsed: {self.stats['total_sessions']}")
        print(f"  Valid training samples: {self.stats['valid_samples']}")
        print(f"  Failed JSON parses: {self.stats['failed_parses']}")
        print(f"  Assistant msgs without head_label: {self.stats['samples_without_head_label']}")
        print(f"  Max sequence length: {self.max_length}")
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.data)

    def _load_images(self, image_paths):
        """✅ NEW: Separated image loading logic"""
        loaded_images = []
        img_sizes = []

        for p in image_paths:
            full_path = os.path.join(self.image_root, p)
            try:
                img = Image.open(full_path).convert("RGB")
                loaded_images.append(img)
                img_sizes.append(img.size)  # (width, height)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load {full_path}: {e}")
                # ✅ FIX: More reasonable fallback size
                fallback = Image.new("RGB", (640, 480))
                loaded_images.append(fallback)
                img_sizes.append((640, 480))
        
        return loaded_images, img_sizes

    def _prepare_messages_with_images(self, messages, loaded_images):
        """✅ NEW: Separated message preparation logic"""
        # ✅ FIX: Use shallow copy instead of deep copy for performance
        messages_copy = [msg.copy() for msg in messages]
        
        if messages_copy and messages_copy[0]['role'] == 'user':
            content = []
            
            # Insert all images
            for img in loaded_images:
                content.append({"type": "image", "image": img})

            # Insert text
            original_content = messages_copy[0]['content']
            if isinstance(original_content, str):
                content.append({"type": "text", "text": original_content})
            elif isinstance(original_content, list):
                content.extend(original_content)

            messages_copy[0]['content'] = content
        
        return messages_copy

    def _process_box_targets(self, head_label, img_sizes):
        """✅ NEW: Separated box processing logic with improved validation"""
        target_boxes = torch.zeros((MAX_BOXES, 4), dtype=torch.float)
        target_boxes_mask = torch.zeros(MAX_BOXES, dtype=torch.float)

        # Get raw boxes
        raw_boxes = head_label.get('box_targets', [])
        
        # ✅ FIX: Backward compatibility for single 'box' field
        if not raw_boxes and 'box' in head_label:
            raw = head_label['box']
            if raw and len(raw) == 4 and sum(raw) > 0:
                raw_boxes = [raw]

        if not raw_boxes:
            return target_boxes, target_boxes_mask

        # Determine reference image for normalization
        target_img_idx1 = head_label.get('image_index', 0)
        ref_idx = target_img_idx1 if 0 <= target_img_idx1 < len(img_sizes) else 0
        
        # ✅ NEW: Safety check
        if ref_idx >= len(img_sizes):
            print(f"⚠️  Warning: Invalid image index {ref_idx}, defaulting to 0")
            ref_idx = 0
        
        img_w, img_h = img_sizes[ref_idx]

        # Process each box
        valid_box_count = 0
        for i, box in enumerate(raw_boxes):
            if i >= MAX_BOXES:
                print(f"⚠️  Warning: More than {MAX_BOXES} boxes, truncating")
                break

            if len(box) != 4:
                print(f"⚠️  Warning: Invalid box format: {box}")
                continue

            x1, y1, x2, y2 = box
            
            # ✅ NEW: Validate box coordinates
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️  Warning: Invalid box coordinates: {box}")
                continue

            # Convert to [cx, cy, w, h] and normalize
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            # ✅ FIX: Clamp to valid range [0, 1]
            norm_box = [
                max(0.0, min(1.0, cx / img_w)),
                max(0.0, min(1.0, cy / img_h)),
                max(0.0, min(1.0, w / img_w)),
                max(0.0, min(1.0, h / img_h))
            ]

            target_boxes[i] = torch.tensor(norm_box, dtype=torch.float)
            target_boxes_mask[i] = 1.0
            valid_box_count += 1

        # ✅ FIX: Changed default from 0 to 1 (boxes enabled by default)
        global_box_mask = head_label.get('box_mask', 1)
        if global_box_mask == 0:
            target_boxes_mask[:] = 0.0

        return target_boxes, target_boxes_mask

    def __getitem__(self, idx):
        """Get a single training sample."""
        try:
            item = self.data[idx]
            image_paths = item['images']
            head_label = item['head_label']

            # 1. Load images
            loaded_images, img_sizes = self._load_images(image_paths)

            # 2. Prepare messages with images
            messages = self._prepare_messages_with_images(item['messages'], loaded_images)

            # 3. Process text input
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding="max_length",
                max_length=self.max_length,  # ✅ FIX: Now configurable
                return_tensors="pt",
                truncation=True  # ✅ NEW: Explicit truncation
            )

            input_ids = inputs.input_ids.squeeze(0)
            attention_mask = inputs.attention_mask.squeeze(0)

            # Create labels for LM loss
            labels = input_ids.clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100

            # 4. Process Head Labels

            # A. Action ID
            action_id = head_label.get('action_id', 12)  # Default: Terminate
            if not (0 <= action_id < AgentConfig.num_actions):
                print(f"⚠️  Warning: Invalid action_id {action_id}, using TERMINATE")
                action_id = 12

            # B. Image Indices
            target_img_idx1 = head_label.get('image_index', -100)
            target_img_idx2 = head_label.get('image_index_2', -100)
            
            # ✅ NEW: Validate image indices
            if target_img_idx1 >= len(loaded_images):
                print(f"⚠️  Warning: image_index {target_img_idx1} >= {len(loaded_images)}")
                target_img_idx1 = -100

            # C. Multi-Image Indices -> Multi-hot Vector
            target_multi_hot = torch.zeros(MAX_IMAGES, dtype=torch.float)
            multi_indices = head_label.get('image_indices', [])
            for mi in multi_indices:
                if 0 <= mi < MAX_IMAGES:
                    target_multi_hot[mi] = 1.0
                else:
                    print(f"⚠️  Warning: Invalid multi_image_index {mi}")

            # D. Bounding Boxes
            target_boxes, target_boxes_mask = self._process_box_targets(head_label, img_sizes)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": inputs.pixel_values,
                "image_grid_thw": inputs.image_grid_thw,
                "labels": labels,
                # Head Targets
                "tgt_action": torch.tensor(action_id, dtype=torch.long),
                "tgt_img1": torch.tensor(target_img_idx1, dtype=torch.long),
                "tgt_img2": torch.tensor(target_img_idx2, dtype=torch.long),
                "tgt_img_multi": target_multi_hot,
                "tgt_boxes": target_boxes,
                "tgt_boxes_mask": target_boxes_mask,
                "num_valid_images": len(loaded_images),
                "session_id": item.get("session_id", -1)  # ✅ NEW: For debugging
            }

        except Exception as e:
            print(f"\n❌ Error processing sample {idx}: {e}")
            print(f"   Session ID: {item.get('session_id', 'unknown')}")
            raise


def custom_collate_fn(batch):
    """
    Custom collation function to handle variable-length sequences and images.
    
    ✅ IMPROVED: Better error handling and validation
    """
    try:
        # Basic tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        # Handle visual inputs (may be None or variable length)
        pixel_values = None
        image_grid_thw = None

        pvs = [item['pixel_values'] for item in batch if item['pixel_values'] is not None]
        if pvs:
            try:
                pixel_values = torch.cat(pvs, dim=0)
            except Exception as e:
                print(f"⚠️  Warning: Failed to concatenate pixel_values: {e}")
                print(f"   Shapes: {[pv.shape for pv in pvs]}")
                raise

        gthws = [item['image_grid_thw'] for item in batch if item['image_grid_thw'] is not None]
        if gthws:
            try:
                image_grid_thw = torch.cat(gthws, dim=0)
            except Exception as e:
                print(f"⚠️  Warning: Failed to concatenate image_grid_thw: {e}")
                raise

        # Stack Head Targets
        targets = {
            "action_ids": torch.stack([item['tgt_action'] for item in batch]),
            "img1_labels": torch.stack([item['tgt_img1'] for item in batch]),
            "img2_labels": torch.stack([item['tgt_img2'] for item in batch]),
            "img_multi_labels": torch.stack([item['tgt_img_multi'] for item in batch]),
            "box_targets": torch.stack([item['tgt_boxes'] for item in batch]),
            "box_masks": torch.stack([item['tgt_boxes_mask'] for item in batch]),
            "num_valid_images": torch.tensor([item['num_valid_images'] for item in batch], dtype=torch.long)
        }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "targets": targets
        }
    
    except Exception as e:
        print(f"\n❌ Collation failed: {e}")
        print(f"   Batch size: {len(batch)}")
        if batch:
            print(f"   First item keys: {batch[0].keys()}")
        raise
