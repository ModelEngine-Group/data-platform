# HistoQC Mapper for DataMate

[Original Project](https://github.com/choosehappy/HistoQC)

 **说明**: 本算子是 [HistoQC]在 DataMate 算子管线中的标准集成实现。包含了针对 WSI 质量控制产物的结构化提取逻辑。


##  主要改动 (Modifications)

本算子在集成过程中，针对 DataMate 的 `Mapper` 架构进行了以下逻辑增强：

### 1. 空间坐标提取 (Spatial Coordinates Extraction)
**掩码转坐标**：新增了对 HistoQC 输出掩码（Mask PNG）的后处理逻辑。
**GeoJSON 产出**：通过 OpenCV 轮廓检测算法，将伪影区域（如笔迹、盖玻片边缘）转换为标准的 GeoJSON 多边形数据，并封装于 `sample["text"]` 中。
**缩放对齐**：支持通过 `scaleFactor` 参数对检测坐标进行重采样映射，确保坐标体系与原图一致。

### 2. 性能与耗时监控 (Performance & Timing)
**执行计时**：在算子执行周期内引入了计时器，记录每个切片任务的实际处理时长。
**指标关联**：将处理耗时（Processing Time）与输入文件大小（File Size）作为元数据存入输出结果，便于后续进行算法性能评估。
