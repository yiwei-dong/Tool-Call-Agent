import os
import json
import re
from copy import deepcopy
from shutil import rmtree
from PIL import Image

# 导入工具函数和常量
# 假设你的目录结构是:
# - vts_reasoner.py
# - tools.py
from .tools import (
    encode_image,
    get_depth,
    zoom_in_image_by_bbox,
    visual_search,
    grounded_segmentation,
    crop_image_action,
    segment_image,
    ocr_extract_text,
    overlay,
    calculate_text_to_images_similarity,
    calculate_image_to_texts_similarity,
    calculate_image_to_images_similarity,
    TERMINATE, GROUNDING, DEPTH, ZOOMIN, VISUALSEARCH,
    TEXT, OVERLAY, CROP, SEGMENT, OCR,
    IMAGE_TO_IMAGES_SIMILARITY, TEXT_TO_IMAGES_SIMILARITY, IMAGE_TO_TEXTS_SIMILARITY
)


# ==========================================
# 辅助函数：坐标转换
# ==========================================
def unnormalize_box(box, img_w, img_h):
    """
    将 Head 预测的归一化中心点坐标 [cx, cy, w, h] (范围 0.0-1.0)
    转换为 像素坐标 [x1, y1, x2, y2]
    """
    cx, cy, nw, nh = box

    # 计算像素宽和高
    box_w_pixel = nw * img_w
    box_h_pixel = nh * img_h

    # 计算左上角坐标 (Center - Width/2)
    x1 = (cx * img_w) - (box_w_pixel / 2)
    y1 = (cy * img_h) - (box_h_pixel / 2)

    x2 = x1 + box_w_pixel
    y2 = y1 + box_h_pixel

    # 确保不越界 (可选，但推荐)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return [int(x1), int(y1), int(x2), int(y2)]


def vts_reasoner(reasoner, image_path_list, task_prompt, system_prompt="", developer_prompt="", image_save_dir=None):
    """
    Args:
        reasoner: Reasoner 实例 (包含 Hybrid 模型)
        image_path_list: 图片路径列表或 PIL Image 对象列表
        task_prompt: 用户任务
        system_prompt: 系统提示词
    """

    # 1. 图片初始化
    if len(image_path_list) > 0 and isinstance(image_path_list[0], Image.Image):
        images = image_path_list
    else:
        images = [Image.open(image_path).convert("RGB") for image_path in image_path_list]

    # 2. 构建初始 Message List
    message_list = [
        {
            "role": "system",
            "content": system_prompt if len(system_prompt) else "You are a helpful assistant."
        }
    ]
    if len(developer_prompt):
        message_list[-1]['content'] += "\n" + developer_prompt

    # 构建 User Message
    user_content = []

    # 添加初始图片
    for i, image in enumerate(images):
        # 注意：这里我们构建用于 Trace 和 Hybrid 输入的通用格式
        # 具体 reasoner.predict_hybrid 会处理格式转换
        user_content.append({"type": "text", "text": f"Image Index: {i + 1}\n"})
        user_content.append({"type": "image", "image": image})  # 直接存 PIL Image

    user_content.append({"type": "text", "text": task_prompt})

    message_list.append({
        "role": "user",
        "content": user_content
    })

    call_stack = []
    completion_stack = []
    final_response = "No answer reached."

    print(f"Task Started: {task_prompt}")

    # 3. 推理循环
    while True:
        # ------------------------------------------------
        # Step A: 调用模型预测 (Think + Predict Action)
        # ------------------------------------------------
        # 传入 images 列表，供 Head 预测 image_index
        prediction = reasoner.predict_hybrid(message_list, images)

        action_name = prediction["action_name"]
        action_args = prediction["action_args"]
        thought = prediction["thought"]

        print(f"\n[Thought]: {thought}")
        print(f"[Action]: {action_name} | [Args]: {action_args}")

        # 记录 Trace
        call_stack.append({"name": action_name, "args": action_args, "thought": thought})
        completion_stack.append(thought)

        # 将 Thought 加入历史 (Assistant Role)
        # 注意：为了让模型知道自己之前的思考，必须加入历史
        message_list.append({
            "role": "assistant",
            "content": thought
        })

        # ------------------------------------------------
        # Step B: 终止条件检查
        # ------------------------------------------------
        if action_name == TERMINATE:
            # 尝试从 text args 获取最终回复，如果为空则使用 thought
            text_args = action_args.get("text", [])
            final_response = " ".join(text_args) if text_args else thought
            break

        # ------------------------------------------------
        # Step C: 执行工具 (Execute)
        # ------------------------------------------------
        return_images = []
        observation_content = []  # 本轮 Observation

        # 提取通用参数
        # image_index 来自 Head (1-based)，转换为 0-based
        img_idx = action_args.get("image_index", 1) - 1
        text_args = action_args.get("text", [])
        norm_box = action_args.get("box", [0, 0, 0, 0])

        # 确保图片索引有效
        if img_idx < 0 or img_idx >= len(images):
            observation_content.append({"type": "text", "text": f"Error: Image Index {img_idx + 1} is out of bounds."})
        else:
            current_image = images[img_idx]
            img_w, img_h = current_image.size

            # --- 1. Grounding ---
            if action_name == GROUNDING:
                ret_imgs, boxes, labels = grounded_segmentation(
                    image=current_image,
                    labels=text_args,
                    polygon_refinement=True
                )
                if boxes:
                    box_str = "\n".join([f"Label: {l}, Box: {b}" for l, b in zip(labels, boxes)])
                    observation_content.append({"type": "text", "text": f"Grounding Boxes:\n{box_str}\n"})

                # 处理返回的可视化图
                for i, img in enumerate(ret_imgs):
                    return_images.append(img)

            # --- 2. Zoom In (需要 Box 转换) ---
            elif action_name == ZOOMIN:
                pixel_box = unnormalize_box(norm_box, img_w, img_h)
                ret_imgs = zoom_in_image_by_bbox(image=current_image, bounding_box=pixel_box)
                for crop in ret_imgs:
                    return_images.append(crop)
                    observation_content.append({"type": "text", "text": "Zoomed in successfully."})

            # --- 3. Crop (需要 Box 转换) ---
            elif action_name == CROP:
                pixel_box = unnormalize_box(norm_box, img_w, img_h)
                ret_imgs = crop_image_action(image=current_image, bounding_box=pixel_box)
                for crop in ret_imgs:
                    return_images.append(crop)
                    observation_content.append({"type": "text", "text": "Image cropped successfully."})

            # --- 4. Visual Search ---
            elif action_name == VISUALSEARCH:
                query = text_args[0] if text_args else "object"
                ret_imgs, boxes_list, labels_list = visual_search(image=current_image, objects=query)
                if boxes_list:
                    count = len(boxes_list)
                    observation_content.append({"type": "text", "text": f"Found {count} matches for '{query}'.\n"})
                for patch in ret_imgs:
                    return_images.append(patch)

            # --- 5. OCR ---
            elif action_name == OCR:
                ocr_res = ocr_extract_text(image=current_image)
                observation_content.append({"type": "text", "text": f"OCR Result:\n{ocr_res['text']}\n"})

            # --- 6. Segment ---
            elif action_name == SEGMENT:
                # 假设 Segment 这里简单使用单个 box，或者你需要根据逻辑调整
                pixel_box = unnormalize_box(norm_box, img_w, img_h)
                # segment_image 通常需要 [[x1,y1,x2,y2]] 列表
                ret_imgs = segment_image(image=current_image, bounding_boxes=[pixel_box])
                for img in ret_imgs:
                    return_images.append(img)
                    observation_content.append({"type": "text", "text": "Segmentation completed."})

            # --- 7. Overlay (需要第二张图) ---
            elif action_name == OVERLAY:
                other_indices = action_args.get("other_image_indices", [])
                if other_indices:
                    idx2 = other_indices[0] - 1
                    if 0 <= idx2 < len(images):
                        ret_imgs = overlay(background_image=current_image, overlay_image=images[idx2])
                        for img in ret_imgs:
                            return_images.append(img)
                            observation_content.append({"type": "text", "text": "Overlay completed."})
                    else:
                        observation_content.append({"type": "text", "text": "Error: Overlay target index invalid."})

            # --- 8. Similarity ---
            elif action_name == IMAGE_TO_IMAGES_SIMILARITY:
                other_indices = action_args.get("other_image_indices", [])
                # 转换所有 indices
                valid_others = [images[i - 1] for i in other_indices if 0 <= (i - 1) < len(images)]
                if valid_others:
                    result = calculate_image_to_images_similarity(current_image, valid_others)
                    best_idx = other_indices[result['best_match_index']]
                    observation_content.append({
                        "type": "text",
                        "text": f"Similarity Scores: {result['similarity_scores']}\nBest Match Image Index: {best_idx}"
                    })

            # --- Fallback ---
            elif action_name == TEXT:
                # 纯文本动作，无额外 Observation，除非为了提示
                pass

            else:
                observation_content.append({"type": "text", "text": "Tool executed (No visual output)."})

        # ------------------------------------------------
        # Step D: 更新 Observation 到历史 (User Role)
        # ------------------------------------------------

        # 处理返回的新图片
        for new_img in return_images:
            images.append(new_img)
            # 添加图片到 Observation
            observation_content.append({"type": "text", "text": f"New Generated Image Index: {len(images)}\n"})
            observation_content.append({"type": "image", "image": new_img})

        # 如果没有任何输出，给一个默认反馈防止模型不知道发生了什么
        if not observation_content and action_name != TEXT:
            observation_content.append(
                {"type": "text", "text": "Action executed successfully with no specific output."})

        # 只有当有 Observation 时才添加 User Message
        if observation_content:
            message_list.append({
                "role": "user",
                "content": observation_content
            })

    # ==========================================
    # 4. 后处理与保存
    # ==========================================

    # 清理: 为了 JSON 序列化，将 message_list 里的 Image 对象替换为占位符
    message_list_trace = deepcopy(message_list)
    for ms in message_list_trace:
        if isinstance(ms["content"], list):
            for content in ms["content"]:
                if content.get("type") == "image":
                    content["image"] = "<image_object>"  # 替换 PIL 对象

    # 保存图片
    if image_save_dir:
        if os.path.exists(image_save_dir):
            rmtree(image_save_dir)
        os.makedirs(image_save_dir, exist_ok=True)

        image_save_paths = []
        for i, image in enumerate(images):
            path = os.path.join(image_save_dir, f'{i}.jpg')
            image.save(path)
            image_save_paths.append(path)
    else:
        image_save_paths = []

    traces = {
        "call_stack": call_stack,
        "completion_stack": completion_stack,
        "message_list": message_list_trace,
        "images_saved_paths": image_save_paths
    }

    return final_response, traces