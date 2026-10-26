import os
import time
import json
import tempfile
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import gradio as gr

from config import Config
from core import MultiScaleExtractor, compute_anomaly_map, postprocess_map, get_transforms, map_sensitivity_to_threshold

cfg = Config()
device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

# Load weights and threshold securely
checkpoint = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
base_threshold = float(checkpoint["threshold"])
border_margin = int(checkpoint.get("border_margin", cfg.border_margin))

print(f"Loaded Student Weights successfully!")
print(f"Calibrated Base Threshold loaded from model: {base_threshold:.4f}")

teacher = MultiScaleExtractor(pretrained=True).to(device).eval()
for p in teacher.parameters(): p.requires_grad = False

student = MultiScaleExtractor(pretrained=False).to(device).eval()
student.load_state_dict(checkpoint["state_dict"])

transform = get_transforms(cfg, is_train=False)

def inspect_single_image(image, sensitivity_val, min_area_input):
    if image is None:
        return ("<div style='text-align:center; padding:15px; font-weight:bold;'>Upload an image to inspect</div>", 
                "", "", "", None, None, None, None, pd.DataFrame(), None, None)

    start_time = time.time()
    orig_img = image.convert("RGB")
    orig_w, orig_h = orig_img.size

    effective_threshold = map_sensitivity_to_threshold(sensitivity_val, base_threshold)
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    with torch.inference_mode():
        t_feat, s_feat = teacher(input_tensor), student(input_tensor)
        raw_map = compute_anomaly_map(s_feat, t_feat, out_size=(orig_h, orig_w))

    scaled_margin = int(border_margin * (max(orig_w, orig_h) / cfg.input_size))
    smoothed_masked, clean_mask = postprocess_map(raw_map, effective_threshold, border_margin=scaled_margin, sigma=cfg.sigma)

    top_k = max(1, int(smoothed_masked.size * 0.001))
    anomaly_score = float(np.mean(np.partition(smoothed_masked.ravel(), -top_k)[-top_k:]))

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_list = []
    total_defect_area = 0
    contour_img_np = np.array(orig_img).copy()
    usable_area = max(1, (orig_w - 2 * scaled_margin) * (orig_h - 2 * scaled_margin))
    is_unreliable = False

    valid_idx = 1
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area_input:
            total_defect_area += area
            if (area / usable_area) > cfg.max_defect_coverage_ratio: 
                is_unreliable = True
            
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

            cv2.drawContours(contour_img_np, [c], -1, (255, 0, 0), 2)
            cv2.putText(contour_img_np, f"#{valid_idx}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            defect_list.append({
                "Defect ID": f"#{valid_idx}", 
                "Area (px)": int(area),
                "Area (%)": round((area / usable_area) * 100, 2), 
                "Centroid": f"({cx}, {cy})",
                "Bounding Box": f"[{x}, {y}, {w}, {h}]"
            })
            valid_idx += 1

    if is_unreliable:
        verdict, status_bg, sub_msg = "UNRELIABLE", "#F59E0B", "Single defect covers >35% area. Check lighting."
    elif len(defect_list) > 0 or anomaly_score > effective_threshold * 1.1:
        verdict, status_bg, sub_msg = "FAIL", "#EF4444", "Defect detected above calibrated threshold."
    elif anomaly_score >= effective_threshold * 0.9:
        verdict, status_bg, sub_msg = "BORDERLINE", "#F59E0B", "Slight surface variance near decision boundary."
    else:
        verdict, status_bg, sub_msg = "PASS", "#10B981", "Surface normal and free of defects."

    elapsed_time = round(time.time() - start_time, 3)
    scaled_map = np.clip(smoothed_masked / (2.0 * effective_threshold), 0.0, 1.0)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap((scaled_map * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay_img = cv2.addWeighted(np.array(orig_img), 0.6, heatmap_rgb, 0.4, 0)
    mask_img = Image.fromarray(clean_mask)

    status_html = f"<div style='background:{status_bg}; color:white; padding:15px; border-radius:10px; text-align:center;'><h2>{verdict}</h2><p>{sub_msg}</p></div>"
    gauge_html = f"<b>Anomaly Score: {anomaly_score:.4f}</b> / Effective Threshold: {effective_threshold:.4f}"
    stats_html = f"<b>Defects Found:</b> {len(defect_list)} | <b>Latency:</b> {elapsed_time}s"

    df_out = pd.DataFrame(defect_list) if defect_list else pd.DataFrame([{"Status": "No defects found."}])

    return (status_html, f"Effective Threshold: `{effective_threshold:.4f}`", gauge_html, stats_html, overlay_img, Image.fromarray(contour_img_np), Image.fromarray(heatmap_rgb), mask_img, df_out, None, None)

# Launch UI with Public Share Link
with gr.Blocks(title="Industrial Leather Inspection UI") as demo:
    gr.Markdown("# 🏭 Automated Leather QC Inspection Engine")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Leather Image")
            sensitivity_slider = gr.Slider(1.0, 10.0, value=5.0, label="Sensitivity (1 = Low, 5 = Normal, 10 = High)")
            min_area_slider = gr.Slider(10, 500, value=cfg.min_defect_area, label="Min Defect Area (px)")
            inspect_btn = gr.Button("Inspect Sample", variant="primary")
        with gr.Column():
            status_banner = gr.HTML()
            anomaly_gauge = gr.HTML()
            key_stats_box = gr.HTML()
            defect_dataframe = gr.Dataframe()

    with gr.Row():
        out_overlay = gr.Image(label="Thermal Overlay")
        out_contours = gr.Image(label="Defect Vector Boundaries")
        out_heatmap = gr.Image(label="Calibrated Heatmap")
        out_mask = gr.Image(label="Segmentation Mask")

    inspect_btn.click(
        inspect_single_image,
        inputs=[input_image, sensitivity_slider, min_area_slider],
        outputs=[status_banner, gr.Markdown(), anomaly_gauge, key_stats_box, out_overlay, out_contours, out_heatmap, out_mask, defect_dataframe, gr.File(), gr.File()]
    )

demo.launch(share=True)