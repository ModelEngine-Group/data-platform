import cv2
import numpy as np
import json
import os
import logging
from pathlib import Path

def export_geojson(di, params):
    #  获取当前切片的输出目录
    outdir = di.get("outdir", ".")
    # 获取切片名称 (HistoQC 通常以这个名字开头保存图片)
    sname = di.get("sname") or di.get("img_base")
    if not sname and "filename" in di:
        sname = Path(di["filename"]).stem
   
    if not sname:
        logging.error("无法获取切片名称，跳过 GeoJSON 生成。")
        return

    # 获取缩放因子 (WSI原图与处理图的倍率)
    scale_factor = di.get("log_factor", 1.0)
   
    # 定义想要识别的后缀和对应 QuPath 类别
    # 注意：这里务必和 output 文件夹里生成的文件名后缀一致
    ARTIFACT_CONFIG = {
        "_pen_markings.png": {"name": "Pen Marking", "color": -65536},
        "_coverslip_edge.png": {"name": "Coverslip Edge", "color": -16711936},
        "_flat.png": {"name": "Air Bubble", "color": -16776961}
    }

    all_features = []
    found_files = []

    logging.info(f" [切片: {sname}] 启动后期扫描提取 ")

    # 扫描文件夹中对应的 PNG
    for suffix, info in ARTIFACT_CONFIG.items():
        # HistoQC 保存的规律通常是 {sname}{suffix}
        # 例如: TCGA-XXX_pen_markings.png
        target_path = Path(outdir) / f"{sname}{suffix}"
      
        if not target_path.exists():
            # 兼容性扫描：如果文件名里有多余的点或空格，进行模糊匹配
            potential = list(Path(outdir).glob(f"*{suffix}"))
            if potential:
                target_path = potential[0]
            else:
                continue

        # 读取并转换
        mask = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or np.max(mask) == 0:
            continue

        found_files.append(target_path.name)

        # 提取轮廓
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            pts = (contour.reshape(-1, 2) * scale_factor).tolist()
            if len(pts) < 3: continue
            pts.append(pts[0]) # 闭合

            all_features.append({
                "type": "Feature",
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": info["name"], "colorRGB": info["color"]},
                    "isLocked": False
                },
                "geometry": {"type": "Polygon", "coordinates": [pts]}
            })
    #  保存 JSON
    if all_features:
        output_file = Path(outdir) / f"{sname}_artifacts.json"
        with open(output_file, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": all_features}, f)
        logging.info(f"[成功] 从 {found_files} 提取了 {len(all_features)} 个坐标点")
    else:
        logging.warning(f"[跳过] 在目录 {outdir} 中未找到有效伪影图像或图像为空")
    
    return