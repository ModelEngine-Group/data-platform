# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
import os
import pandas as pd
import subprocess
from pathlib import Path
from typing import Dict, Any
from datamate.core.base_op import Mapper

class HistoQCMapper(Mapper):
    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 动态路径与环境变量初始化
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 定位算子包内部的配置文件
        config_path = os.path.join(current_dir, "histoqc_src", "histoqc", "config", "config_v2.1.ini")
        # 定位 HistoQC 源代码目录
        histoqc_src_dir = os.path.join(current_dir, "histoqc_src")

        file_path = sample.get("filePath", "")
        out_dir = sample.get("export_path", os.path.dirname(file_path))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # HistoQC 在 TSV 和图片命名中通常保留完整文件名（含后缀）
        base_name = os.path.basename(file_path)

        # 注入环境变量，确保能找到包内 HistoQC 模块
        custom_env = os.environ.copy()
        custom_env["PYTHONPATH"] = f"{histoqc_src_dir}{os.pathsep}{custom_env.get('PYTHONPATH', '')}"
        
        #启动 HistoQC 算法
        cmd = [
            "python3", "-m", "histoqc",
            "-o", out_dir,
            "-c", config_path,
            "-n", "4",  
            file_path
        ]

        try:

            # 执行算法，设置 30 分钟超时
            result = subprocess.run(cmd, capture_output=True, text=True, env=custom_env, timeout=1800)

            # 如果被系统 Kill (返回码 -9) 或其他错误，记录在 text 字段
            if result.returncode != 0:
                sample["text"] = json.dumps({"error": f"HistoQC failed with return code {result.returncode}", "stderr": result.stderr})
                return sample

        except Exception as e:
            sample["text"] = json.dumps({"error": f"Subprocess exception: {str(e)}"})
            return sample

        #结果解析与归档
        final_report = {
            "slide_name": base_name,
            "metrics": {},
            "annotations": [],
            "generated_images": []
        }

        #  解析 TSV 质量指标

        tsv_path = Path(out_dir) / "results.tsv"

        if tsv_path.exists():
            try:
                # comment='#' 跳过 HistoQC 的元数据行，on_bad_lines='skip' 增强容错
                df = pd.read_csv(tsv_path, sep='\t', comment='#', on_bad_lines='skip')
                # 匹配第一列文件名
                row = df[df.iloc[:, 0].astype(str) == base_name]
                if not row.empty:
                    final_report["metrics"] = row.iloc[-1].to_dict()
            except Exception as e:
                final_report["metrics_error"] = str(e)

        # 提取伪影坐标并转换为 GeoJSON (适应多种路径结构)
        scale_factor = float(getattr(self, 'scaleFactor', 1.0))
        artifact_types = {
            "pen_markings": {"name": "Pen Marking", "color": -65536},
            "coverslip_edge": {"name": "Coverslip Edge", "color": -16711936}
        }

        # 考虑到 HistoQC 可能会创建子文件夹，使用 glob 进行深度搜索
        search_dir = Path(out_dir) / base_name
        if not search_dir.exists():
            search_dir = Path(out_dir)

        for key, info in artifact_types.items():
            # 搜索包含文件名和伪影类型的图片
            found_files = list(search_dir.glob(f"*{base_name}*{key}*.png"))
            if found_files:
                mask_path = found_files[0]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.max(mask) > 0:
                    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        pts = (contour.reshape(-1, 2) * scale_factor).tolist()
                        if len(pts) < 3: continue
                        pts.append(pts[0])
                        final_report["annotations"].append({
                            "type": "Feature",
                            "properties": {"classification": {"name": info["name"], "colorRGB": info["color"]}},
                            "geometry": {"type": "Polygon", "coordinates": [pts]}
                        })
                final_report["generated_images"].append(mask_path.name)
        #   写回 DataMate 样本数据
        sample["text"] = json.dumps(final_report,ensure_ascii=False,indent=2)
        if final_report["metrics"]:
            sample["extra_info"] = final_report["metrics"]
        return sample