# 摔倒检测成果与对比台账

更新日期：2026-08-13

本文件记录可横向比较的冻结结果。开发集成绩、最终测试成绩和外部测试
严格分开；除非表格明确注明，否则不要把开发集数字作为产品准确率。
这是一套工程基准，不是医疗或人身安全认证。

## 评测协议

- 数据集：GMDCSA-24 v2.1。
- Subject 1–2：训练时序 MLP。
- Subject 3：选择特征 mask、隐藏层宽度、正则、阈值和连续确认次数。
- Subject 1–3：配置冻结后重新拟合权重。
- Subject 4：最终测试，排除此前使用过的 10 个 pipeline smoke 片段，剩余
  27 段包含 12 个 Fall 和 15 个 ADL。
- 外部测试：RealBiomFall testing subset，34 段全部为 Fall，因此只能报告
  Recall、提前报警和延迟，不能报告 Accuracy、Specificity、Precision 或 F1。
- 视频统一按 15 FPS 采样；每段视频开始前重置 tracker 和时序状态。
- 比标注跌倒起点早 0.5 秒以上的报警按提前误报处理，不算 TP。
- `temporal gate` 是模型第一次连续满足概率门限的时刻；`deployed alert`
  是 MQTT 实际可见的 `fallen/recovering` 状态。

## 最终部署对比：GMDCSA Subject 4

下面是完整多人 tracker 加实际状态机的结果，是当前最适合用于版本比较的
主表。

| 平台/版本 | Pose 前端 | TP | FN | TN | FP | Accuracy | Recall | Specificity | Precision | F1 | 平均延迟 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reCamera v0.2 temporal baseline | YOLO11n INT8 CVI | 10 | 2 | 10 | 5 | 74.1% | 83.3% | 66.7% | 66.7% | 74.1% | 1.75 s |
| Jetson 优化前 deployed | YOLO11s FP16 TRT | 8 | 4 | 11 | 4 | 70.4% | 66.7% | 73.3% | 66.7% | 66.7% | 1.69 s |
| Jetson 优化后 deployed | YOLO11s FP16 TRT | 10 | 2 | 12 | 3 | 81.5% | 83.3% | 80.0% | 76.9% | 80.0% | 1.47 s |
| Jetson 优化前 deployed | YOLO11m FP16 TRT | 9 | 3 | 11 | 4 | 74.1% | 75.0% | 73.3% | 69.2% | 72.0% | 1.70 s |
| Jetson 优化后 deployed | YOLO11m FP16 TRT | 12 | 0 | 11 | 4 | **85.2%** | **100%** | 73.3% | **75.0%** | **85.7%** | **1.26 s** |

优化后 YOLO11m 相对其旧 deployed 版本：Accuracy +11.1 个百分点、Recall
+25.0 个百分点、F1 +13.7 个百分点，并消除了本测试集中的全部 Fall 漏报；
代价是多 1 个 ADL 误报。

## 时序模型开发集记录

两个 profile 都使用 48 帧/3.2 秒窗口、504 维聚合特征、骨盆中心化 pose
mask、32 个隐藏单元、0.8 概率阈值和连续 3 次确认。以下仅为 Subject 3
开发验证，不作为最终准确率。

| Profile | 样本 | TP/FN | TN/FP | Accuracy | Recall | F1 | 中位延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `yolo11s-pose` | 43 | 21/0 | 22/0 | 100% | 100% | 100% | 0.90 s |
| `yolo11m-pose` | 43 | 21/0 | 21/1 | 97.7% | 100% | 97.7% | 0.90 s |

运行时 `temporal_profile=auto` 根据 engine 文件名选择相应的冻结权重；权重
直接编译进 C++ 动态库，部署不依赖 Torch、Ultralytics、scikit-learn 或
ONNX Runtime。

## 外部泛化：RealBiomFall testing

| 版本/输出 | TP | FN | Recall | 提前报警 | 平均延迟 | Pose coverage |
|---|---:|---:|---:|---:|---:|---:|
| reCamera v0.2 temporal baseline | 20 | 14 | 58.8% | 9 | 1.18 s | — |
| Jetson YOLO11m 优化前 temporal gate | 18 | 16 | 52.9% | 6 | 0.96 s | 70.6% |
| Jetson YOLO11m 优化前 deployed | 17 | 17 | 50.0% | 6 | 0.95 s | 70.6% |
| Jetson YOLO11m 优化后 temporal gate | 21 | 13 | **61.8%** | 9 | 0.51 s | 70.6% |
| Jetson YOLO11m 优化后 deployed | 18 | 16 | 52.9% | 7 | 0.99 s | 70.6% |

外部集的主要限制仍是远景/遮挡情况下的 pose coverage，部分视频几乎检测不到
人。不能用这 34 段 testing 数据继续选阈值；下一轮应增加独立开发数据改善
远景人体检测和 track continuity。

## Orin 性能记录

设备均为 SM87、TensorRT 10.3、CUDA 12.6、FP16、固定 1x3x640x640。
`trtexec` 数字只表示推理核心，不包含 RTSP、NVDEC/VIC、预处理、NMS、跟踪
和 MQTT。

| Engine | Orin Nano `trtexec` | Orin NX `trtexec` | Nano 完整评测桥接均值 |
|---|---:|---:|---:|
| YOLO11s-Pose | 12.20 ms / 81.97 FPS | 10.73 ms / 93.18 FPS | 14.87 ms |
| YOLO11m-Pose | 21.30 ms / 46.95 FPS | 18.58 ms / 53.83 FPS | 24.31 ms |

当前保守路数建议仍是 Orin Nano 4 路 YOLO11s、Orin NX 3 路 YOLO11m，按
15 FPS 输入起步；最终路数必须用实际 RTSP 编码、场景人数、功耗模式和 MQTT
负载做持续压力测试。

## reCamera / reCamera Pro 状态

| 平台 | 多人独立轨迹/状态 | 优化时序权重 | 可比较最终测试 |
|---|---|---|---|
| reCamera SG2002 | 已实现 | 仍为原 v0.2/CVI profile | 有原 v0.2 baseline |
| reCamera Pro | 已实现 | 尚未按 Pro pose 输出重训 | 尚无同协议最终数据 |
| Jetson Orin | 已实现 | YOLO11s/m 独立 profile 已冻结 | 已完成 |

状态机结构和训练流程可以复用到 reCamera/Pro，但 Jetson FP16 TensorRT 权重
不能直接冒充 CVI INT8 或 Pro profile。后续应分别用两端真实 pose 输出抽取
Subjects 1–3 traces、重新训练，然后只在冻结后读取 Subject 4。

## 可审计产物

- 完整解释与历史基线：[`EVALUATION.md`](EVALUATION.md)
- 逐片段 JSON：[`evaluation/`](evaluation/)
- 文件校验和：[`evaluation/SHA256SUMS`](evaluation/SHA256SUMS)
- 训练工具：[`tools/train_temporal_for_pose.py`](tools/train_temporal_for_pose.py)
- 视频评测工具：[`tools/evaluate_videos.py`](tools/evaluate_videos.py)

新增结果时必须追加协议、平台、engine/profile、混淆矩阵和原始 JSON；不要只
记录一个 Accuracy 百分比。
