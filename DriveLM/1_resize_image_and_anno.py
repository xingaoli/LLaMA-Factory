import json
from PIL import Image
import os
import re

input_path = "data/nuscenes_drivelm/v1_1_train_nus.json"  # 替换成你的输入文件路径
output_path = "data/nuscenes_drivelm/v1_1_train_nus_pixel_224.json"
resized_img_dir = "data/nuscenes_drivelm/samples_pixel_224"
os.makedirs(resized_img_dir, exist_ok=True)

# 等比例缩放图像
def resize_and_save_image_keep_aspect(input_path, output_path, max_width=400):
    img = Image.open(input_path).convert('RGB')
    orig_size = img.size  # (width, height)
    w, h = orig_size
    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size)
    img.save(output_path)
    return orig_size, new_size

# 固定尺寸缩放图像
def resize_and_save_image(input_path, output_path, width=224, height=224):
    img = Image.open(input_path).convert('RGB')
    orig_size = img.size  # (width, height)
    new_size = (width, height)
    img = img.resize((width, height))
    img.save(output_path)
    return orig_size, new_size


with open(input_path, 'r') as f:
    test_file = json.load(f)

for scene_id, scene_content in test_file.items():
    scene_description = scene_content.get("scene_description", "")
    key_frames = scene_content.get("key_frames", {})

    for frame_id, frame_data in key_frames.items():
        image_paths = frame_data.get("image_paths", {})
        cam_image_size = {}  # 存储各相机的 (原始尺寸, 缩放后尺寸)
        for cam_name, img_path in image_paths.items():
            img_path = img_path.replace('..', 'data').replace('nuscenes', 'nuscenes_drivelm')
            output_img_path = os.path.join(resized_img_dir, '/'.join(img_path.split('/')[-2:]))
            # orig_size, new_size = resize_and_save_image(img_path, output_img_path, 224, 224)
            orig_size = (int(1600), int(900))
            new_size = (int(224), int(224))
            cam_image_size[cam_name] = (orig_size, new_size)

        # 根据图像变换后的大小来缩放qa问答中的像素绝对坐标值
        key_object_infos = frame_data.get("key_object_infos", {})
        updated_obj_ids = {}  # 记录原始obj_id -> 更新后id 的映射
        updated_obj_2d_box = {} # 更新后的obj_id -> 更新后的2d box
        for obj_id, obj_info in key_object_infos.items():
            desc = obj_info.get("Visual_description", "Unknown object")
            category = obj_info.get("Category", "Unknown category")
            status = obj_info.get("Status", "Unknown status")
            bbox = obj_info.get("2d_bbox", None)
            try:
                parts = obj_id.strip("<>").split(",")
                cid, cam, cx, cy = parts[0], parts[1], float(parts[2]), float(parts[3])
            except:
                cam, cx, cy = None, None, None
            if cam in cam_image_size:
                orig_w, orig_h = cam_image_size[cam][0]
                new_w, new_h = cam_image_size[cam][1]
                x_scale = new_w / orig_w
                y_scale = new_h / orig_h
                if cx is not None and cy is not None:
                    cx_r = round(cx * x_scale, 1)
                    cy_r = round(cy * y_scale, 1)
                    new_obj_id = f"<{cid},{cam},{cx_r},{cy_r}>"
                    updated_obj_ids[obj_id] = new_obj_id
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x1_r = round(x1 * x_scale, 1)
                    y1_r = round(y1 * y_scale, 1)
                    x2_r = round(x2 * x_scale, 1)
                    y2_r = round(y2 * y_scale, 1)
                    updated_obj_2d_box[updated_obj_ids[obj_id]] = [x1_r, y1_r, x2_r, y2_r]

        resized_key_object_infos = {}
        for old_id, new_id in updated_obj_ids.items():
            resized_key_object_infos[new_id] = key_object_infos[old_id]
            resized_key_object_infos[new_id]['2d_bbox'] = updated_obj_2d_box[new_id]

        frame_data['key_object_infos'] = resized_key_object_infos

        # 提取所有 QA 对
        frame_data_qa = frame_data.get("QA", {})
        for k_fdq, v_fdq in frame_data_qa.items():
            for v_v_fdq in v_fdq:
                question = v_v_fdq.get("Q", "")
                answer = v_v_fdq.get("A", "")
                # 替换 question 和 answer 中的 obj_id
                for old_id, new_id in updated_obj_ids.items():
                    question = question.replace(old_id, new_id)
                    answer = answer.replace(old_id, new_id)
                    v_v_fdq["Q"] = question
                    v_v_fdq["A"] = answer

with open(output_path, 'w') as f:
    json.dump(test_file, f, indent=2)

print(f"✅ Done! Resized QA samples written to {output_path}")