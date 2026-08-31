# 摔倒检测成果与对比台账

更新日期：2026-08-30

本文件记录可横向比较的冻结结果。开发集成绩、最终测试成绩和外部测试
严格分开；除非表格明确注明，否则不要把开发集数字作为产品准确率。
这是一套工程基准，不是医疗或人身安全认证。

所有运行时结果必须符合 [`../contracts/MQTT.md`](../contracts/MQTT.md)；只有
通过同一 MQTT contract、同一数据划分和同一事件匹配规则的结果才能横向比较。

## 跨平台完成度总表

`pending` 表示实现或真机评测仍在进行；`blocked` 必须同时写明外部阻塞，不能
用推测值补齐。性能与准确性是两个独立门槛：只有性能 smoke 不能宣称准确率，
模型自己的 COCO pose mAP 也不能当作跌倒 Accuracy。

| 平台 | 原生加速后端 | 可复现 Compose | MQTT contract | 真机性能 | GMDCSA S4 | RealBiomFall |
|---|---|---|---|---|---|---|
| reCamera SG2002 | CVI Runtime INT8 | N/A（固件/appMgr） | 已实现，待 schema fixture | 已有单路现场基线 | 74.1% Accuracy / 83.3% Recall | 58.8% Recall |
| reCamera Pro | RKNN | N/A（Pro app packaging） | 已实现；WS 真机字段已采集，MQTT schema fixture 待真机 broker | 已完成单路 live camera/NPU/WebSocket | production fallback 81.5% Accuracy / 91.7% Recall；native experiment 70.4% / 75.0% | N/A（未执行外部集评测） |
| Jetson Orin Nano/NX | TensorRT 10.3 FP16 | 已有 | 已实现，fixture + 真机多路 RTSP/MQTT payload | 已完成早期 1/2/3/4/6 context、单路 E2E，以及当前 Nano 8 路/NX 9 路 RTSP 边界 | 已完成 | 已完成 |
| RK3576 | RKNN Runtime | 已有 | 已实现，fixture test | 已完成 | 88.9% Accuracy / 100% Recall（native temporal gate） | N/A（未执行外部集评测） |
| RK3588 | RKNN Runtime | 已有 | 已实现，fixture test | 已完成（含既有 NPU 负载） | 88.9% Accuracy / 100% Recall（native temporal gate） | N/A（未执行外部集评测） |
| Raspberry Pi + Hailo-8 | HailoRT/GStreamer | 已有 | 已实现，fixture + 2,602 条 RTSP 实时消息 | 已完成 synthetic、RTSP 单/双路、S 16 路/M 5 路最大路数与有人跌倒正向链路 | 88.9% Accuracy / 100% Recall（native temporal gate） | N/A（未执行外部集评测） |

## 统一性能表

最终行必须同时记录设备、功耗模式、模型与量化、输入、并发路数、纯推理延迟、
完整 pipeline 延迟/吞吐、CPU、RSS、加速器利用率和功耗。暂时无法可靠读取的
指标写 `N/A（原因）`，不能省略测量口径。

| 平台/设备 | 模型/精度 | 输入 | 路数 | 纯推理 ms / FPS | 完整 pipeline FPS / P95 | CPU | RSS | 加速器利用率 | 功耗 | Runtime 镜像 | 状态 |
|---|---|---:|---:|---|---|---|---|---|---|---|---|
| reCamera SG2002 / OS 0.2.2 | YOLO11n-Pose CVI INT8 | 640² | 1 | 52.96 mean / 53 P95 | 10.00 FPS / pending | pending | 11.6 MB | pending | pending | N/A（appMgr deb） | 无标注现场 200 帧 |
| reCamera Pro / firmware V1.0.4 | YOLO11n-Pose RKNN INT8 | 640²（720p camera frame） | 1 | 35.89 mean / 39.36 P95 | 13.05 FPS；77.80 / 85.99 ms mean/P95 | N/A（appMgr未暴露进程CPU） | 836.6–839.9 MB system used | NPU 19–21% | N/A（无可靠口径） | N/A（signed appMgr package） | 783帧/60秒；温度51.2–52.5°C；现场仅3帧有人，不作Accuracy |
| Orin Nano（2026-08-13 历史基线） | YOLO11s-Pose TRT FP16 | 640² | 1 E2E；1/2/3/4/6 inference context | 14.76/14.79 ms E2E infer；context aggregate 69.66/71.12/70.07/70.15/70.85 FPS | 单路 RTSP 14.94 FPS / 69.63 ms 输出间隔 P95 | 11.08% E2E | 211.3 MiB E2E | 24.82% mean | 9.07 W mean | 206 MB disk / 51.8 MB content | `--infStreams` 为推理-only；4 路是吞吐推算，不是 RTSP 容量实测 |
| Orin NX（2026-08-13 历史基线） | YOLO11m-Pose TRT FP16 | 640² | 1 E2E；1/2/3/4/6 inference context | 20.98/21.03 ms E2E infer；context aggregate 53.83/54.27/53.60/53.60/53.54 FPS | 单路 RTSP 15.02 FPS / 69.74 ms 输出间隔 P95 | 10.43% E2E | 126.7 MiB E2E | 28.48% mean | 13.85 W mean | 206 MB disk / 51.8 MB content | `--infStreams` 为推理-only；3 路是吞吐推算，不是 RTSP 容量实测 |
| Orin Nano Super（2026-08-20 当前边界） | YOLO11s-Pose TRT FP16 | 640² | 8 路通过；9 路失败 | 6.44 ms mean / 155.3 FPS `trtexec` core | 8 路各 14.95 FPS；9 路各 13.36 FPS | N/A（报告未记录） | 9 路 711 MiB | GR3D 91%@8 路 / 98%@9 路 | N/A（报告未记录） | N/A（当前报告未记录） | RTSP+MQTT 实测最大通过 8 路，阈值 14.5 FPS/路 |
| Orin NX Super（2026-08-20 当前边界） | YOLO11s-Pose TRT FP16 | 640² | 9 路通过；10 路失败 | 5.75 ms mean / 173.8 FPS `trtexec` core | 9 路各 14.93 FPS；10 路各 13.05 FPS | N/A（报告未记录） | 8 路 664 MiB；9/10 路 N/A | GR3D 95%@9 路 / 99%@10 路 | N/A（报告未记录） | N/A（当前报告未记录） | RTSP+MQTT 实测最大通过 9 路，阈值 14.5 FPS/路 |
| RK3576 | YOLO11n-Pose RKNN FP16 | 640² | 1/2 context | 63.03 mean / 74.44 P95 | 15.15 FPS；65.73 / 77.98 ms | 63.1% snapshot | 174.4 MiB | Core0/1 36%/0% snapshot | N/A（无可靠口径） | 257,793,213 B | 4 人 bus 图；2ctx blank 29.15 FPS |
| RK3588 | YOLO11n-Pose RKNN FP16 | 640² | 1/2/3 context | 51.41 mean / 58.92 P95（1ctx） | 19.25 / 38.13 / 51.40 aggregate FPS | N/A（争用） | 189.9/217.6/294.8 MiB | 100%@1GHz（含既有负载） | N/A（无可靠口径） | 257,793,213 B | 现有 voice/RKLLM 争用；无 failed submit |
| Raspberry Pi 5 + Hailo-8 | YOLOv8s-Pose HEF INT8 | 640² | 1/2；16 路 RTSP | 6.87 ms / 393.3 FPS（HailoRT） | 旧基线：单路 14.32 FPS、双路 14.33+14.30、有人流 14.72；本轮 16 路 14.5215–14.5715 FPS、probe 36.16–40.93 ms | 有人流 12.5% final；B17 246% | 有人流 130,784 KiB max；B17 1,366,592 KiB | N/A（CLI未给利用率） | N/A（无可靠板级遥测） | 143,442,009 B | 既有 2,602/2,602 MQTT contract 通过；本轮 ENABLE_MQTT=OFF，16 路为受控 RTSP 当前最大通过路数 |

## 统一准确性表

以下待填行必须使用本文件开头的冻结协议。若某平台 pose frontend 不同，应先用
Subjects 1–3 重训/冻结其时序 profile，再只读 Subject 4 和外部集。

| 平台/Profile | 数据集/划分 | TP | FN | TN | FP | Accuracy | Recall | Specificity | Precision | F1 | 平均报警延迟 | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| reCamera v0.2 CVI | GMDCSA S4 | 10 | 2 | 10 | 5 | 74.1% | 83.3% | 66.7% | 66.7% | 74.1% | 1.75 s | frozen baseline |
| Jetson YOLO11s optimized | GMDCSA S4 | 10 | 2 | 12 | 3 | 81.5% | 83.3% | 80.0% | 76.9% | 80.0% | 1.47 s | frozen |
| Jetson YOLO11m optimized | GMDCSA S4 | 12 | 0 | 11 | 4 | 85.2% | 100% | 73.3% | 75.0% | 85.7% | 1.26 s | frozen |
| reCamera Pro production fallback on Pro traces | GMDCSA S4 | 11 | 1 | 11 | 4 | 81.5% | 91.7% | 73.3% | 73.3% | 81.5% | 1.22 s | frozen comparator；production default；early alert 1 |
| reCamera Pro native profile | GMDCSA S4 | 9 | 3 | 10 | 5 | 70.4% | 75.0% | 66.7% | 64.3% | 69.2% | 1.47 s | frozen experiment；early alerts 3；未promote |
| RK3576 native profile | GMDCSA S4 | 12 | 0 | 12 | 3 | 88.9% | 100% | 80.0% | 80.0% | 88.9% | 1.49 s | frozen；独立 RK3576 traces |
| RK3588 native profile | GMDCSA S4 | 12 | 0 | 12 | 3 | 88.9% | 100% | 80.0% | 80.0% | 88.9% | 1.53 s | frozen；独立 RK3588 traces |
| Hailo-8 native temporal gate | GMDCSA S4 | 12 | 0 | 12 | 3 | 88.9% | 100% | 80.0% | 80.0% | 88.9% | 1.61 s | frozen；pose coverage 92.02% |

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

### Jetson Spark LAN RTSP 完整链路（2026-08-13）

统一控制流为 H.264 Constrained Baseline 640×640@15 FPS、约 1.2 Mbps，经
Spark LAN MediaMTX 输入生产 Python 应用。链路包含 NVDEC/VIC、TensorRT、
多人 tracker、冻结时序模型、状态机和 MQTT JSON 序列化。MQTT publisher 边界被
本地 wrapper 截获，用于逐条验证实际发布 payload；未测 broker 网络传输延迟。

| 设备/Profile | 消息/时长 | 输出 FPS / 间隔 P95 | infer mean/P95 | CPU/RSS | GPU | 板功耗 | Contract |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nano / YOLO11s | 918 / 61.36 s | 14.94 / 69.63 ms | 14.76/14.79 ms | 11.08% / 211.3 MiB | 24.82% mean | 9.07 W mean | 918/918 pass |
| NX / YOLO11m | 932 / 61.98 s | 15.02 / 69.74 ms | 20.98/21.03 ms | 10.43% / 126.7 MiB | 28.48% mean | 13.85 W mean | 932/932 pass |

`fall-person` 使用循环 GMDCSA S4 Fall/01 验证正向完整路径。Nano 的 867 条消息
全部通过 contract，663 帧有可见人、866 帧保留 track/pose17、573 帧
temporal-positive、产生 2 个 stream-global fall event；NX 的 861 条消息全部通过，
819 帧有可见人、861 帧保留 track/pose17、788 帧 temporal-positive、产生 1 个
stream-global fall event。循环事件数只证明 RTSP→TRT→tracker→temporal→状态机
→payload 路径，不是 Accuracy/Recall。输出间隔 P95 也不是 source-to-MQTT 延迟。
本轮 NX 开始前和结束后 Docker 业务集合均为空，没有停止任何 NX 业务。

## 跨平台同模型对比：YOLO11n-Pose 640²（2026-08-14）

此前 Jetson 只测过 YOLO11s/11m，而 RK 与 reCamera 跑的是 YOLO11n，横向数字比的是
「平台+模型」而不是平台。这一轮把模型固定成 YOLO11n-Pose 640² 重测。方法与既有行一致：
Jetson 用 `trtexec --useCudaGraph --noDataTransfers --infStreams=N`，RK 用
`platforms/rknn/benchmark.py` 空白 640 输入、迭代数与 FP16 基线相同。单帧均为加速器
推理耗时，不含 RTSP 解码、letterbox、跟踪、时序 MLP 与 MQTT。

| 平台 | 精度 | 单帧 | 1 ctx | 3 ctx | 6 ctx | 备注 |
|---|---|---:|---:|---:|---:|---|
| Orin Nano | FP16 | 3.69 ms | 270.7 | 274.9 | 263.0 | 停业务 |
| Orin NX | FP16 | 3.26 ms | 306.2 | 283.3 | 287.8 | 停业务 |
| Orin Nano | INT8＊ | 2.75 ms | 363.9 | 387.5 | 351.8 | 停业务；未标定 |
| Orin NX | INT8＊ | 2.45 ms | 408.0 | 425.4 | 393.6 | 停业务；未标定 |
| Orin Nano | FP16 | 3.69 ms | 270.5 | 270.3 | 264.9 | 业务共存（对照） |
| Orin NX | FP16 | 3.77 ms | 264.9 | 245.2 | 245.2 | 业务共存（对照） |
| RK3588 | FP16 | 51.41 ms | 19.25 | 51.40 | — | 既有基线 |
| RK3588 | INT8 | 29.78 ms | 32.43 | 90.36 | — | w8a8，240 帧标定 |
| RK3576 | FP16 | 56.13 ms | 17.50 | — | — | 既有基线；2 ctx 29.15 |
| RK3576 | INT8 | 36.23 ms | 26.97 | — | — | w8a8；2 ctx 42.05 |

＊ Jetson 的 INT8 引擎由 `trtexec --int8` 直接构建，**没有标定器也没有标定集**，动态范围
是随意取的：只能用于看内核速度，检测结果不可用，不是可部署配置。`build_engine.sh` 只传
`--fp16`；要做真正的 INT8 需要在项目里实现标定器与标定集，并在 INT8 姿态输出上重新冻结
时序权重。

**只有占用加速器的共存业务才会影响数字，一旦影响就可能把排序颠倒。** Orin NX 在自身跑着
GPU 推理业务时 FP16 测得 264.9 FPS，低于 Orin Nano，与两者算力关系相反；停掉后是
306.2 FPS。Orin Nano 停与不停完全一致（270.5 对 270.7），因为它上面的 openclaw /
warehouse / face_rec 都不碰 GPU。同一效应让 Orin NX 上 INT8 单 context 首次测得
4.55 ms——比自己的 FP16 还慢，看上去像 INT8 在 Jetson 上退化；空闲下是 2.45 ms。
结论：跨板对比必须在停掉占用加速器的业务后进行。

**RKNN INT8 是净收益。** 两块板都快 1.4–1.8 倍，且在 8 张**未参与标定**的 GMDCSA 帧上，
每帧检测数与 FP16 完全一致（1.12 vs 1.12），框相差约 1 px、分数相差 ±0.04。唯一可测的
退化是一个画面边缘的部分人体（0.391→0.358，可见关键点 2/17→0/17），它在 FP16 下也几乎
没有可用姿态。注意：这只是 8 帧的逐帧一致性，**没有算 mAP，也没有在 INT8 轨迹上重跑冻结
的 Subject 4 门限**，因此不能直接沿用 FP16 的准确率数字。

固定模型且停掉占用加速器的业务后，边缘板卡的排序为：Orin NX 3.26 ms ＞ Orin Nano
3.69 ms ＞ RK3588 51.4 ms ＞ reCamera 2002 53.0 ms，Jetson 单帧约为 RK3588 的 1/15。
Jetson 的总吞吐在 1–6 个 context 下几乎不变，说明并发共享同一份 GPU 预算而不是线性倍增。

**Hailo-8 是唯一一块「固定成 11n 反而没法比」的板子。** Model Zoo v2.15 的 hailo8 目录没有
任何 n 尺寸姿态模型，我们用 Dataflow Compiler 3.31.0 自行编译了一份 YOLO11n-Pose（640²、
INT8、64 帧 GMDCSA 标定、raw head），在 `harvest-pi` 上停掉 `mcp_face_rec` 后实测：

| HEF | context 数 | FPS (hw_only) | HW 延迟 |
|---|---:|---:|---:|
| YOLO11n-Pose（自编译） | 3 | 92.20 | 9.01 ms |
| YOLOv8s-Pose（Model Zoo v2.15） | 1 | 393.90 | 6.87 ms |

两个数要分开看：**单帧延迟同量级（9.01 vs 6.87 ms，+31%），差 4.3 倍的是吞吐**。按单路 15 FPS 的部署需求，n 的 92.2 FPS 仍有约 6 倍余量，差距只在多路密度上体现。`hailortcli parse-hef` 给出原因：11n 被切成 3 个 context，每帧换一次
权重；s 尺寸是 single context，权重常驻。编译期已经有征兆——总算力占用只有 16.6%，
cluster_2 的 control 利用率却已 100%。所以这个数字反映的是该 HEF 的资源分配，不是
Hailo-8 跑 11n 的上限。部署配置维持 YOLOv8s：又快、模型又大。

根因是**编译期 control 资源顶满，不是算力不够**。三个 context 的 control 利用率
60.2% / 60.2% / 30.5%，对应 compute 只有 29.5% / 36.9% / 16.6%；cluster_4（ctx0）、
cluster_2 与 cluster_7（ctx1）、cluster_2（ctx2）都已 100% control。control 预算按层和
按输出流消耗，与算多少无关——nano 的层数并不比 s 少（YOLO11 的 C3k2/C2PSA 分支更碎），
raw head 还多出 9 条输出流。每帧为此付 25.09 Mbps 的跨 context 搬运。

而且那次是**裸编译**：model script 只有一行输入归一化，没有 `performance_param` 或
`resources_param`，allocator 跑的是默认档；被对比的 Model Zoo `yolov8s_pose` 是带调优
脚本编的。所以这是「调过的 s」对「没调的 n」。

加 batch 能验证这个判断，但修不好它：

| batch | n FPS | n 延迟（每批） | s FPS | s 延迟 |
|---:|---:|---:|---:|---:|
| 1 | 92.20 | 9.01 ms | 393.90 | 6.87 ms |
| 4 | 160.28 | 19.41 ms | — | — |
| 8 | 183.40 | 33.20 ms | 393.76 | 7.29 ms |
| 16 | 198.26 | 60.56 ms | — | — |

n 回收了 2.15 倍（92.2→198.3 FPS），但 batch=16 仍只有 s 在 batch=1 时的一半。s 则在各
batch 下持平（393.9→393.8）——single context、权重常驻就是这个形状。剩下的差距就是多
context 切分本身，只能在编译期解决。

可选路径：① 加 `performance_param(compiler_optimization_level=max)` 与放宽的
`resources_param` 重编，争取并掉只有 16.6% compute 的 ctx2；② 把 9 个 raw head 按尺度
concat 成 3 个，降输出流与 control 压力（要改 ONNX end node 和 host 解码）；③ 加 batch，
只提吞吐且封顶在 s 的一半。

方法：`hailortcli benchmark --time-to-run 15`，只测加速器，与 trtexec / rknn 各行同口径。

原始数据：[`reports/yolo11n-crossplatform-20260814.json`](reports/yolo11n-crossplatform-20260814.json)

## reCamera / reCamera Pro 状态

| 平台 | 多人独立轨迹/状态 | 优化时序权重 | 可比较最终测试 |
|---|---|---|---|
| reCamera SG2002 | 已实现 | 仍为原 v0.2/CVI profile | 有原 v0.2 baseline |
| reCamera Pro | 已实现 | 原生 MLP 已冻结/测试；同一 Pro traces 上 fallback 更优，故继续作为生产默认 | S4 native/fallback 均已完成 |
| Jetson Orin | 已实现 | YOLO11s/m 独立 profile 已冻结 | 已完成 |
| Raspberry Pi 5 + Hailo-8 | 已实现原生 HailoRT 运行时 | Hailo YOLOv8s-Pose 独立 profile 已冻结并编译 | S4 temporal gate 已完成；deployed state-machine accuracy 未单独评估 |
| RK3576 | 已实现原生 RKNN Lite 多路运行时 | RK3576 独立 profile 已冻结并部署 | S4 temporal gate 已完成；deployed state-machine accuracy 未单独评估 |
| RK3588 | 已实现原生 RKNN Lite 多路运行时 | RK3588 独立 profile 已冻结并部署 | S4 temporal gate 已完成；deployed state-machine accuracy 未单独评估 |

## Raspberry Pi 5 + Hailo-8 状态（2026-08-30）

Fleet `harvest-pi` 已验证 Hailo-8、HailoRT/driver/firmware 4.21.0、Debian 13、
Pi 5 16 KB page 与 `force_desc_page_size=4096`。官方 Hailo-8
`yolov8s_pose.hef`（640×640，SHA256
`e19856699ed47cf866d23265827f960b263f287dab5e54e82c7ce37e12525a2d`）可被
`hailortcli parse-hef` 正确解析为 9 个量化输出。原生 C++ 服务与 payload
contract 共 2 项测试全部通过。

| 模型/口径 | 单路 | 双路 | CPU/RSS/温度/功耗 | 跌倒 Accuracy | Runtime 镜像 |
|---|---:|---:|---:|---:|---:|
| YOLOv8s-Pose Hailo-8，纯 NPU / synthetic pipeline | 393.3 FPS、6.87 ms / 29.77 FPS、7.81 ms probe | 每路 29.86/29.66 FPS，probe 7.87/7.81 ms | 双路 CPU 38.2%、RSS 118,480 KiB、60.05°C；功耗 N/A | S4 temporal gate 88.89% | 143,442,009 B |
| Mac RTSP 640²@15 低码率控制流 | 14.32 app FPS；15.03 steady MQTT FPS；probe 7.77/8.36 ms mean/P95 | 每路 14.33/14.30 app FPS；15.07/15.04 steady MQTT FPS；probe P95 15.37/15.43 ms | 单/双路 CPU final 11.5%/22.9%；RSS max 127,760/182,080 KiB；温度 max 60.4/60.9°C；功耗 N/A | 无人物检测，不能用于 Accuracy | 143,406,131 B |
| Mac RTSP 1280x720@15 对照 | 13.41 app FPS；14.17 steady MQTT FPS；probe 7.88/8.50 ms mean/P95 | — | CPU final 24.8%；RSS max 133,184 KiB；温度 max 60.4°C | 无人物检测，不能用于 Accuracy | 143,406,131 B |
| Spark LAN `fall-person`，GMDCSA S4 Fall/01 循环 | 14.72 app FPS；15.02 steady MQTT FPS；probe 7.48/8.53 ms mean/P95 | — | CPU final/max 12.5%/15.0%；RSS max 130,784 KiB；温度 max 62.0°C | 循环正样本功能测试，不能用于 Accuracy | 143,406,131 B |

经用户明确许可，短暂停止独占 `/dev/hailo0` 的 `mcp_face_rec` 完成了基准；测试后
已 `docker start` 恢复并确认容器重新为 `healthy`。Synthetic 数字使用
`test://ball` 隔离 RTSP 网络；RTSP 数字使用 Mac 经 Tailscale 发布的统一控制流。
控制流单/双路共 1,311 条、720p 对照 407 条 MQTT 消息全部通过统一 contract。
两个 640² stream id 与 frame counter 独立。720p 服务端观察到 reader slow/discard，
其丢帧不能归因于 Hailo。前三组控制素材未产生人物检测，因此验证的是空检测
分支。随后 Spark `fall-person` 运行 60.05 秒：884/884 消息通过 contract，615 条
有可见有效人物和 pose17，813 条保留 track，462 条 person temporal-positive，
并覆盖 normal/suspected/fallen/recovering 四状态。5 个事件把 stream-global
event ID 从 0 单调推进到 5；每个事件当前人物均 `features.valid=true`、pose17 为
17 点、temporal probability=1.0。循环正样本产生多次事件只证明完整
decode→Hailo→pose→tracker→temporal→状态机→MQTT 路径，不是准确率样本。
`pipeline_ms` 是 hailonet 前缓冲到 source pad 的 probe 延迟，
不是源端到 MQTT 的总延迟，也不冒充纯 NPU call。原始压缩 MQTT、资源 CSV、日志和
汇总见 `reports/rpi-hailo8-rtsp-*20260813*`。Hailo Model Zoo 报告的 59.2/56.36
是 COCO pose full/HW mAP，只能描述量化 pose 精度，不能填入跌倒 Accuracy。
功耗仍无可靠实测值。Pi 原生最终 runtime 镜像已构建，content size
143,442,009 bytes，digest
`sha256:7e7d81503ed94160d8a9f64d5caa388a806a51a0e801889c40aeb85f591c86e2`；
镜像不包含 HEF、HailoRT、编译器或源码，模型与 ABI 锁定库由 compose 只读挂载。

Spark LAN 后续控制流 `rtsp://192.168.3.42:8554/fall-e2e-low` 和有人跌倒流
`rtsp://192.168.3.42:8554/fall-person` 已从 Pi ffprobe 为 H.264 Constrained
Baseline 640²@15，Pi→Spark RTT 平均 3.96 ms。有人正样本 E2E 与 Hailo 原生
pose trace/时序 profile 全量冻结评估均已完成。
GMDCSA 160 段总时长已从 Spark 素材实测为 1,289.173 秒；Subjects 1-3 为
999.657 秒，Subject 4 为 289.515 秒。计划拆成约 25 分钟开发 trace 窗口和约
10 分钟冻结测试窗口，总计预留 45 分钟含每段 RTSP 启停/状态重置开销，CPU 训练
期间不占 Hailo。160/160 clips 在 307 秒内提取完成、失败 0，随后服务恢复
healthy。S1-2 fit、S3 select、S1-3 refit 后生成 freeze manifest，首次读取 S4
得到 TP/FN/TN/FP=12/0/12/3，Accuracy/F1=88.89%、Recall=100%、Specificity/
Precision=80%、平均/中位延迟=1.608/1.25 秒、早报 0。S4 clean 的 valid pose17
coverage 为 2743/2981=92.02%。这是 temporal gate 指标，不冒充完整 deployed
state-machine Accuracy。profile header 175,450 B，SHA256
`dec7237a1204cd2d9d54aa6810ca941ef82b83a18e65bb06a4eb7893bb55faf9`，已切换
native runtime 并在 Pi 上重新编译，逻辑/contract 2/2 通过。采集转换工具与完整冻结步骤见
[`../platforms/rpi-hailo/TEMPORAL_TRAINING.md`](../platforms/rpi-hailo/TEMPORAL_TRAINING.md)。

状态机结构和训练流程可以复用到 reCamera/Pro，但 Jetson FP16 TensorRT 权重
不能直接冒充 CVI INT8 或 Pro profile。后续应分别用两端真实 pose 输出抽取
Subjects 1–3 traces、重新训练，然后只在冻结后读取 Subject 4。

## reCamera Pro 频率档位对性能的影响（2026-08-14）

复测 35.89 ms 基线时先测出 43.12 ms。原因是频率档位，不是争用。

Pro 的 NPU 用 `rknpu_ondemand`，30 秒内 60/60 采样都停在 **800 MHz**（上限 950）；CPU
`interactive`，跑在 1008 MHz（上限 1608）。把 `min_freq` 顶到 950 MHz、CPU 切
`performance` 后复测：

| 状态 | 帧数 | infer mean | median | p95 | pre/infer/post | pipeline |
|---|---:|---:|---:|---:|---|---:|
| 默认 800 MHz | 379 | 43.12 ms | 41.30 | 51.82 | 0.02 / 42.93 / 2.07 | 45.01 ms |
| 锁定 950 MHz | 382 | 35.18 ms | 34.75 | 38.11 | 0.00 / 35.21 / 1.34 | 36.56 ms |

**同一块板仅因频率档位就差 23%**，锁频后与冻结基线 35.89 一致（差 2%）；按频率折算的预测值
43.12 × 800/950 = 36.3 ms 也落在同一点。因此 Pro 的任何数字不写明 NPU 频率就不可复现。
测完已还原为 `min_freq=396000000` + `interactive`。

两块 RK 不受影响：RK3588 40/40 采样稳在 1000 MHz、RK3576 40/40 稳在 950 MHz，都是各自上限
——尽管 `platforms/rknn/benchmark.py` 从未设置过 governor。Jetson 记录的是 MAXN_SUPER，
Hailo 没有这一层。

场景说明：两轮都近乎空场（有人帧 2/379 与 3/382），所以 post 一列偏低。Pro 与 RK 同属
RKNN 路径，后处理随人数增长（RK3576 4 人画面 2.7 ms、空白帧 0.4 ms）。

附：该板 load average 约 13（四核）**不是负载**。十几个 Rockchip 媒体线程（`vvi_thread`、
`venc`、`vpss`、`vrga`、`valloc`、`vlog`）常驻不可中断 D 态，Linux 把 D 态计入 load average。
实际 CPU 总占用约 23%（应用 55.7%、`rkipc` 28.7%，均为单核百分比）。

## reCamera Pro 原生时序 MLP 状态（2026-08-13）

这里的“前端重训”不是重新训练 YOLO Pose，而是固定 Pro 的
`yolo11n_pose_rawhead_int8.rknn`，用它导出的关键点 trace 重新训练小型时序 MLP。
不同 pose 前端会改变关键点置信度、缺点率和空间误差分布，所以 Jetson YOLO11s
MLP 只能作为功能 fallback，不能冒充 Pro 原生准确率。

已交付：RV1126B RKNNLite/ffmpeg 15 FPS trace extractor、S1-2 fit、S3 select、
S1-3 refit、冻结 checkpoint/哈希、S4 只读 test 和 NumPy-only gzip JSON exporter。
部署侧依旧只有 NumPy，不需要 Torch、Ultralytics 或 sklearn；这些库只在训练主机
拟合 MLP 时使用。

Fleet 现已有临时 RV1126B Pro 测试机。固件 V1.0.4 上通过官方 appMgr 单活切换、
签名校验和安装了完整四文件 0.2.0 包；60 秒 live camera→RKNN→tracker/temporal→
WebSocket 采集得到 783 帧、13.05 FPS，推理 mean/P95 35.89/39.36 ms，完整 pipeline
mean/P95 77.80/85.99 ms，NPU 19–21%，温度 51.2–52.5°C。现场仅 3 帧检出人物且
没有跌倒边沿，所以这只是真机性能/功能证据，不是 Accuracy。原始证据与口径见
[`reports/recamera-pro-live-v0.2.0-20260813.json`](reports/recamera-pro-live-v0.2.0-20260813.json)。

严格协议已完成 160/160 clips。原生 profile 在 clean S4 为 TP9/FN3/TN10/FP5，
Accuracy 70.37%、Recall 75%、F1 69.23%，early alerts 3；同一 Pro traces 上现有
fallback 为 TP11/FN1/TN11/FP4，Accuracy/F1 81.48%、Recall 91.67%，early alert 1。
因此生产默认保持 fallback，原生 profile 仅保留作审计实验。全体 pose coverage
87.27%，S4 Fall 72.84%，明显低于 ADL 91.77%，是下一轮前端/RGA预处理优化重点。

官方 frame broker 的性能优化已完成：原 1280×720 NV12→全尺寸 RGB→Python
letterbox 改为 NV12 dma-buf 经 RGA 等比例缩到 640×360、再由 RGA 转 RGB，
Python 只填充灰值 114 的边框。原图尺寸和独立 letterbox metadata 继续用于把
pose/box 映回原图，RGA 任一步失败会永久降级到旧路径。12.02 秒真机 A/B 中，
direct 路径为 18.13 FPS、主循环 preprocess 0.0 ms；旧路径为 12.14 FPS、
preprocess 38.2–43.1 ms，吞吐提升 49.3%。完整结果见
[`reports/recamera-pro-rga-direct-ab-20260813.md`](reports/recamera-pro-rga-direct-ab-20260813.md)
和
[`reports/recamera-pro-rga-direct-ab-20260813.json`](reports/recamera-pro-rga-direct-ab-20260813.json)。
测试全程通过 appMgr 单活切换，结束后已恢复 `retail-vision` 并确认持续输出检测。

## reCamera 0.2.2 真机运行基线

2026-08-13 在测试机 `recamera-one`（SG2002、reCamera OS 0.2.2）上通过
supervisor `appMgr/installApp` 和 `appMgr/stop → switch` 部署，未绕过应用切换器，
避免与其他相机应用争用 VPSS。模型为固件内置
`yolo11n_pose_cv181x_int8.cvimodel`，输入 640×640@15 FPS；结果从本机 MQTT
`recamera/fall-detection/results` 连续采集。应用包版本仍为 0.2.0；这里的 0.2.2
指设备固件版本。

| 现场样本 | 消息数 | 输出 FPS | 推理均值 / P95 | RSS | 人数范围 | 有效 person-track 帧 | temporal-positive person-track 帧 | fall event |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 修复前 | 200 | 10.00 | 53.00 / 53 ms | 11.4 MB | 0–2 | 22 / 247 | 150 / 247 | 2 |
| valid-observation gate 修复后 | 200 | 10.00 | 52.96 / 53 ms | 11.6 MB | 1 | 0 / 200 | 4 / 200 | 0 |

修复前两个事件都发生在 `features.valid=false` 的轨迹上：一个已经丢失 7 帧，
另一个虽仍有检测框，但肩/髋关键点不足以形成有效几何观测。根因是时序窗口的
高概率会在当前姿态无效时继续把 `normal/suspected` 推进为 `fallen`。当前状态机
要求进入 `fallen` 的确认帧必须有有效当前观测；短时遮挡仍可保持既有状态，但
不能凭陈旧时序分数创建新事件。修复后二次采样没有再出现该类事件，且推理性能
没有可测退化。

这两段是未人工标注、场景也未按测试协议控制的现场运行数据，只能证明吞吐、
资源占用和上述状态机不变量，**不能计算或宣称 Accuracy/Recall/F1**。修复后样本
的 200 个 person-track 帧全部 `features.valid=false`，同时也暴露出当前镜头下
COCO 肩/髋关键点覆盖不足；下一步应保存视频和 CVI pose trace，在带标注回放上
评估阈值与重新训练时序 profile，而不能因“0 次报警”得出准确率更高的结论。

统一 MQTT contract 包随后仍通过同一 `appMgr install → stop → switch` 流程部署；
最终 strict-temporal 包（含同步文档）SHA256 为
`5c9d6dcd664d948efeeba82d99d144605ebd3560e9fcbb99c17d4af3622e060c`。
真机实时消息已验证顶层与每人字段、数组类型、`features.valid`、
`stream_id=camera-0` 及 `event_id == global_event_id`，未绕过 supervisor 直接启动。
最终重启后连续 20 条消息均为 `normal`、`event_id=0`，没有无效姿态或绕过几何
候选的 temporal 直接事件；这仍是运行不变量检查，不是准确率样本。

原始 MQTT 采样暂存于开发 Mac：

- `/private/tmp/recamera-fall-0.2.2-live.ndjson`
- `/private/tmp/recamera-fall-0.2.2-visible-gate.ndjson`

它们是临时调试素材，不纳入源码仓库。若要作为长期可审计产物，应按
[`../assets/ASSET_LOCATIONS.md`](../assets/ASSET_LOCATIONS.md) 的约定备份到 Spark，
记录 SHA256 后再更新本台账。

## 丢帧对准确率的影响：GMDCSA Subject 4（2026-08-15）

MPP 路径的 appsink 会在应用忙时丢帧，需要确认这对判定有没有代价。做法是拿已冻结的 RK3588
逐帧轨迹，把丢帧作为**唯一变量**注入，用同一份冻结权重与阈值重跑 Subject 4，排除网络、编码、
场景等其他差异。先复现基线：88.89% / 100% recall / 80% specificity / 1.525 s，与冻结值逐项
一致，误报片段同为 ADL/06、07、16。

| 场景 | 实际丢帧 | Accuracy | Recall | Specificity | 误报 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 基线 | 0% | 88.9% | 100% | 80% | 3 | 1.525 s |
| 6.5% 随机 | 6.01% | 88.9% | 100% | 80% | 3 | 1.49 s |
| 6.5% 周期 | 6.31% | 88.9% | 100% | 80% | 3 | 1.43 s |
| 13% 随机 | 12.62% | 88.9% | 100% | 80% | 3 | 1.22 s |
| 13% 周期 | 12.06% | 88.9% | 100% | 80% | 3 | 1.28 s |

6.5% 对应 appsink 队列深度 3（rc5 默认，实测 13.98 fps），13% 对应深度 1（rc4 及之前）。
**两档都没有任何片段翻转分类**，12 段跌倒全部命中、误报恒为同样 3 段、提前误报 0 次；随机与
周期两种模式结论一致，说明不敏感于丢帧分布。

延迟随丢帧下降（1.525→1.22 s）**不是改进而是副作用**：时序窗口固定 48 帧，丢帧后这 48 帧
覆盖的真实时长变长，确认条件更早满足，等于隐性拉长了判定窗口。本测试集上未换来误报增加，
但窗口时长被丢帧率间接改变是一处不应有的耦合；按时间而非帧数定义窗口可以解耦，**未改动**。

局限：Subject 4 共 27 段，分辨率 3.7 个百分点，指标完全相同意味着无片段翻转，只能回答有无
明显退化，回答不了 1 个百分点级别的差异。Subject 4 仍是只读测试集，此处仅测鲁棒性，未用于
选择任何阈值。

## RK3588 硬件解码（MPP）与 CPU 解码对照（2026-08-14）

用 cgroup 累计 CPU 计时（`cpu.stat` 的 `usage_usec`）而非采样 `docker stats`——后者对这种
突发负载不可用，同一组里样本从 60% 跳到 352%。各预热 60 秒跳过启动突发，再测 60 秒：

| backend | 核数 | 吞吐 | 单帧 CPU |
|---|---:|---:|---:|
| gstreamer_mpp | 2.61 | 12.83 fps | **203.5 ms** |
| opencv_ffmpeg | 3.00 | 14.75 fps | **203.7 ms** |

**整机单帧成本差 0.1%**，但这**不等于**硬件解码没有收益。把解码单独隔离出来测
（用应用自己的 `GStreamerMPP` / `FFmpegRTSP` 两个类，读帧后不做任何下游处理，40 秒）：

| 路径 | 帧数 | 吞吐 | CPU | 单帧 CPU |
|---|---:|---:|---:|---:|
| MPP 硬件解码 | 573 | 14.32 fps | 3.67 s | **6.4 ms** |
| FFmpeg 软件解码 | 628 | 15.52 fps | 17.78 s | **28.3 ms** |

**隔离测量下硬件解码单帧便宜 4.4 倍，省 21.9 ms。** 解码并不廉价。

但这个节省**没有全部兑现到整机**。MPP 的 appsink 原本是 `max-buffers=1 drop=true`，零深度
缓冲吸收不了抖动——应用早处理完就干等下一帧、晚处理完那帧已被丢弃——所以它的吞吐低于
ffmpeg，直接比较核数会把"少处理帧"算成"省 CPU"。把缓冲深度做成可配置后扫描：

| 配置 | 核数 | fps | 单帧 CPU |
|---|---:|---:|---:|
| MPP queue=1 | 2.44 | 12.98 | 187.9 ms |
| MPP queue=3 | 2.64 | 13.98 | 188.6 ms |
| MPP queue=5 | 2.70 | 14.38 | 188.0 ms |
| MPP queue=8 | 2.80 | 14.95 | 187.3 ms |
| opencv_ffmpeg | 2.93 | 14.84 | 197.4 ms |

单帧成本在各深度下恒定（187–189 ms），说明加缓冲不引入开销，只改变处理帧数。

**同吞吐对照（queue=8 的 14.95 fps 对 ffmpeg 的 14.84 fps）：2.80 核对 2.93 核，净省
0.13 核（4.4%），单帧省 10.1 ms（5%）。** 早先记录的 0.43 核差值是在吞吐不等的条件下测的，
其中大部分来自少处理 13% 的帧，不是解码节省。

隔离测量给出的 21.9 ms/帧（0.33 核）**高估了一倍**，差额不是被别处吃掉。给应用加
`read_ms` 计时后，在同吞吐、各 400 帧下两条路径的下游完全相同：

| backend | read 均值/中位 | inference | pipeline |
|---|---:|---:|---:|
| gstreamer_mpp | 6.69 / 1.82 ms | 58.24 ms | 59.68 ms |
| opencv_ffmpeg | 6.70 / 2.51 ms | 58.41 ms | 59.72 ms |

推理与 pipeline 一致，说明不存在"省下的 CPU 在下游被还回去"。此前观察到的
inference 62.0 对 58.1 ms 是两次抓取的画面差异（平均人数 0.623 对 0.600）造成的噪声，
据此提出的 DRAM 带宽争用假设**不成立**。`read()` 中位数 MPP 更低（1.82 对 2.51 ms），
均值相同——均值被等待下一帧的长尾拉平，那是墙钟不是 CPU。

差异出在隔离 harness 本身：`read()` 在空循环里跑、CPU 完全空闲，cv2 的 FFmpeg 解码器会用
更多线程，成本被放大。**以生产条件下的同吞吐对照为准：净省 0.13 核。**

缓冲深度是吞吐换延迟：queue=8 在 14.95 fps 下最深积压约 0.54 秒，而本方案告警延迟本身是
1.4 秒量级。默认取 3（补回约七成吞吐缺口，最坏延迟约 0.2 秒）。

检出不受解码路径影响：各 300 帧下平均人数 0.623（MPP）对 0.600（ffmpeg），有人帧
187 对 180，落在循环片段采样差异内。RGA 缩放与 cv2 缩放不改变检出。

两组测量对不上：若解码省 21.9 ms/帧，整机单帧成本应差出约 20 ms，实测却只差 0.2 ms。
反推非解码部分的单帧成本，MPP 侧为 196.5 ms、ffmpeg 侧为 173.6 ms，两者本该相同却差 13%，
**模型未闭合**。因此 0.39 核的差距中，解码节省与少处理 13% 帧各占多少，用现有数据无法分离
——两种解释算出的数值恰好接近。要分开需要在固定帧率下重测（例如两侧都限速到同一 fps），
这项**未做**。

两条路径的差异是缓冲策略而非解码性能：MPP 走 appsink `max-buffers=1 drop=true`，应用忙时
丢帧、保低延迟；ffmpeg 走 `cv2.VideoCapture` 的内部队列，每帧都处理、延迟累积。对本方案的
含义是**换到 MPP 会少占约一个核，代价是丢约 13% 的帧**，不是净赚。时序模型用 48 帧 /
3.2 秒窗口，帧密度下降对准确率的影响**未测**。

解码器自身稳态不丢帧：`gst-launch` 直连 `fakesink` 测得 20 秒内 delivered 263 + out-dated
discards 19 = 282，与源在该窗口的 284 帧相符；19 次丢弃全部集中在 0:00:02.560 一个时间点，
是启动时清积压追实时的一次性行为。源速率实测 14.2 fps（在解码器之前的 parser 出口测）。

抖动缓冲不是因素：`latency=100/500`、`drop-on-latency=true/false` 三种组合分别为
12.5 / 12.8 / 12.9 fps。

## Rockchip RKNN 真机性能（2026-08-13）

两平台共用 `platforms/rknn`：Python 仅做多流编排、MQTT、多人 tracker、
冻结 NumPy 时序 MLP 与每轨状态机；OpenCV/FFmpeg backend 做 RTSP/resize，
原生 RKNN Lite/librknnrt 做 pose。部署不依赖 Torch、Ultralytics 或 ONNX
Runtime。MQTT 保持 SG2002 的 Unix-ms `timestamp`、stream-global event id、
可见人数/保留 fallen 轨迹和每人 `pose17`/`features` contract。

模型为固定 1x640x640 RGB YOLO11n-Pose raw nine-head FP16，RKNN Toolkit
2.3.2。以下 `inference` 只计 `rknnlite.inference()`；`pipeline` 额外包含
DFL/keypoint decode 与 NMS，不含 RTSP、tracker、时序模型和 MQTT。

| 平台 / 口径 | 单上下文吞吐 | 单上下文 infer mean/P95 | 单上下文 pipeline mean/P95 | 多上下文吞吐 | CPU/RSS/NPU | 跌倒 Accuracy |
|---|---:|---:|---:|---:|---|---|
| RK3576 `cat-remote`, bus 图 4 人, 500 帧 | 15.15 FPS | 63.03 / 74.44 ms | 65.73 / 77.98 ms | 2ctx blank 29.15 FPS | 快照 CPU 63.1%，RSS 174.4 MiB，Core0/1=36%/0% | frozen temporal gate：88.9%（独立 S4） |
| RK3588 `radxa`, blank 640, 100 帧 | 19.25 FPS | 51.41 / 58.92 ms | 51.77 / 59.28 ms | 2ctx 38.13；3ctx 51.40 FPS | 1/2/3ctx RSS 189.9/217.6/294.8 MiB；NPU 快照 100%@1GHz，受既有 voice/RKLLM 争用 | frozen temporal gate：88.9%（独立 S4） |

RK3576 环境：kernel 6.1.99-rk3576、driver 0.9.8、librknnrt 2.3.2、
`rknpu_ondemand`，采样频率 950 MHz。模型 10,532,939 bytes，SHA256
`659519ae8179749925c3f15d978b760f6040a00ae01666c9050036077bac8bbd`。
bus 图稳定输出 4 人；完整 RKNN→pose→tracker→时序 MLP→状态机一帧 smoke
通过。RK3576 独立 profile 已按 S1–3 冻结协议训练并在 clean S4 获得 88.9%
Accuracy / 100% Recall；该值是 temporal gate，不是 bus smoke 或 Jetson 复用值。

RK3588 环境：ROCK 5T、kernel 6.1.84-8-rk2410、driver 0.9.8、librknnrt
2.3.2、`rknpu_ondemand`@1 GHz。模型 7,647,051 bytes，SHA256
`22f00270870b25dc013e4e8e39aed98f82bcabb69b9272302280a6b8b8f48d5c`。
1/2/3 context 均 exit 0、无 failed submit；既有 openvoicestream/RKLLM/
agent 服务未停止、NPU 未 reset，因此数据明确属于“争用下可用吞吐”，不是独占峰值。

共享 `linux/arm64` multi-stage runtime 镜像实测 144,142,723 bytes；流式
`docker save | gzip -1` 为 143,348,208 bytes。模型、RKNN Toolkit、Torch/CUDA
不 baked；RKNN Lite 与 librknnrt 由 compose 只读挂载。目标板首次 build 的
Docker base layer 出现 `unexpected EOF`，但宿主原生 RKNN 数据和本机 ARM64
镜像审计均已完成。详见
[`../platforms/rk3576/RESULTS.md`](../platforms/rk3576/RESULTS.md)。
Radxa 本机同镜像为 144,143,406 bytes、gzip 143,317,758 bytes；compose config
和容器 `--validate` 均通过。详见
[`../platforms/rk3588/RESULTS.md`](../platforms/rk3588/RESULTS.md)。

### Rockchip Spark LAN RTSP 完整链路

统一 Spark H.264 Baseline 640²@15 控制流进入生产 app；链路包含
OpenCV/FFmpeg decode+letterbox、原生 RKNN、后处理、多人 tracker、冻结 NumPy
temporal、状态机及远端 MQTT broker。QoS0 broker 观察存在 frame-id gaps，因此
下表将“broker 实收 FPS”和 payload frame-id rate 分开，不能把前者或后者冒充
纯 NPU 吞吐。

| 平台/路数 | Broker 实收 FPS | Frame-id rate | infer P95 | pipeline P95 | CPU/RSS/NPU 快照 |
|---|---:|---:|---:|---:|---|
| RK3576 / 1 | 4.88 | 10.82 | 94.05 ms | 96.16 ms | 64.19% / 178.8 MiB / 30%,0% |
| RK3576 / 2 | 3.82 + 3.80 | 9.15 + 9.17 | 111.0/109.3 ms | 112.49/110.97 ms | 94.06% / 277.6 MiB / 35%,26% |
| RK3588 / 1 | 8.56 | 14.81 | 76.0 ms | 76.45 ms | 34.73% / 169.8 MiB / 47%,0%,0% |
| RK3588 / 3 | 5.19 + 5.41 + 5.65 | 10.64 + 10.60 + 11.00 | 136.9/134.1/137.6 ms | 137.86/135.11/138.50 ms | 101.89% / 366.4 MiB / 64%,48%,29% |

`fall-person` 循环 GMDCSA S4 Fall/01 的两板各 500 条消息均逐条通过统一
contract。RK3576 有 289 条 visible+pose17、410 条 retained temporal track、
4 个 stream-global fall event，并覆盖四种状态含 recovering；RK3588 分别为
302、398、5，覆盖 normal/suspected/fallen。这证明真实正向路径闭合，不是
Accuracy/Recall。原始 NDJSON/summary 已备份到
`spark:/home/harvest/datasets/fall-detection/evaluation/rk-e2e/20260813/`，聚合证据
为 [`reports/rknn-e2e-spark-20260813.json`](reports/rknn-e2e-spark-20260813.json)。

RK 专属 temporal trace 已在两板完成 Subjects 1–4 的独立、可恢复抽取；数据树
160 MP4/8 CSV 在 Spark/RK3576/RK3588 的内容聚合 SHA256 均为
`007219878d9782391ad455590a76268866687c72c2aa2ead16ad39060f88ba9d`。
每板 160/160 clips、失败 0。每片 trace 原子落盘并写
`extraction-manifest.json`；Subject 4 是在 S1–3 配置冻结后才以显式
`--allow-holdout` 解锁。清洁 S4（排除 10 个 smoke clips）上，两板均为
TP=12/FN=0/TN=12/FP=3；RK3576/RK3588 mean latency 分别为 1.492/1.525 s。
完整命令见
[`../platforms/rknn/TEMPORAL_TRAINING.md`](../platforms/rknn/TEMPORAL_TRAINING.md)。

### RK MPP 等比例 letterbox 与 C++ GIL 优化

生产视频链现在由 `mppvideodec`/RGA 等比例缩放，再把映射缓冲区一次复制到
灰值 114 的 640×640 letterbox；非方形画面不再拉伸。1280×720 真机流在旧
强制 640² 路径上两板均协商失败，新路径在 RK3576/RK3588 均正确协商为
640×360，并各自通过 200/200 MQTT contract。RK3576 的 E2E
inference/pipeline 均值为 62.47/63.09 ms；RK3588（保留 voice/LLM 业务）为
54.18/54.45 ms。

C++ pose decode/NMS 仅在数值热段释放 GIL。两 context 吞吐在 RK3576 基本
持平（29.60→29.42 FPS），RK3588 提升 2.2%（37.34→38.16 FPS），说明当前
主要瓶颈仍是 NPU 而不是 Python GIL。完整方法和逐板 JSON 见
[`reports/rknn-letterbox-gil-ab-20260813.md`](reports/rknn-letterbox-gil-ab-20260813.md)、
[`../platforms/rk3576/results/rk3576-letterbox-gil-ab-20260813.json`](../platforms/rk3576/results/rk3576-letterbox-gil-ab-20260813.json)
与
[`../platforms/rk3588/results/rk3588-letterbox-gil-ab-20260813.json`](../platforms/rk3588/results/rk3588-letterbox-gil-ab-20260813.json)。

## 可审计产物

- 完整解释与历史基线：[`EVALUATION.md`](EVALUATION.md)
- 逐片段 JSON：[`reports/`](reports/)
- 文件校验和：[`reports/SHA256SUMS`](reports/SHA256SUMS)
- 素材位置及公开下载源：[`../assets/ASSET_LOCATIONS.md`](../assets/ASSET_LOCATIONS.md)
- 训练工具：[`../platforms/jetson/tools/train_temporal_for_pose.py`](../platforms/jetson/tools/train_temporal_for_pose.py)
- Pro 原生 MLP 工具：canonical sibling checkout
  `recamera/recamera_pro/apps/fall-detection/tools/train_freeze_temporal_mlp.py`
- Pro 原生 trace 工具：canonical sibling checkout
  `recamera/recamera_pro/apps/fall-detection/tools/extract_pose_traces.py`
- Pro 阻塞清单：canonical sibling checkout
  `recamera/recamera_pro/apps/fall-detection/evaluation/native-profile-status.json`
- 视频评测工具：[`../platforms/jetson/tools/evaluate_videos.py`](../platforms/jetson/tools/evaluate_videos.py)

新增结果时必须追加协议、平台、engine/profile、混淆矩阵和原始 JSON；不要只
记录一个 Accuracy 百分比。

## Rockchip GStreamer MPP/RGA + C++ 后处理复验（2026-08-13）

两板真实 inventory 均只注册 `rockchipmpp:mppvideodec`，没有独立 RGA
GStreamer element；decoder plugin 实际动态链接 MPP 与 RGA，支持 parsed
AU-aligned H.264/H.265 输入及 RGB/NV12/DMABuf 输出。生产链路因此按实机能力实现为
`rtspsrc -> depay -> parse -> mppvideodec(640² RGB) -> appsink uint8 ->
RKNNLite -> C++ decode/NMS`，Python tracker/MLP/FSM/MQTT 不变。硬件链路失败回退
OpenCV/FFmpeg，C++ 导入/运行失败回退 NumPy；strict 模式可禁用回退。

| 平台/流 | schema | Broker FPS / frame-id rate | infer mean/P95 | pipeline mean/P95 | CPU/RSS/NPU 快照 |
|---|---:|---:|---:|---:|---|
| RK3576 low | 200/200 | 14.90 / 14.85 | 57.93/65.43 ms | 58.42/65.74 ms | 56.57% / 192.5 MiB / 52%,10% |
| RK3576 fall-person | 500/500 | 12.55 / 12.55 | 64.37/91.12 ms | 66.70/96.07 ms | 35.92% / 159.9 MiB / 33%,0% |
| RK3588 low | 200/200 | 14.67 / 14.24 | 53.65/83.00 ms | 54.12/83.20 ms | 27.78% / 190.8 MiB / 47% |
| RK3588 fall-person | 500/500 | 12.96 / 12.93 | 61.52/105.17 ms | 63.00/107.13 ms | 289.59% / 164.5 MiB / 39% |

全部 1,400 条 payload 逐条通过统一 contract，且逐条记录
`source_backend=gstreamer_mpp`、`postprocess_backend=cpp`。两板 positive loop
各产生 8 个 fall event，分别有 RK3576 325/459、RK3588 305/430 条 visible/
tracked 消息；这是 E2E 功能证据，不是 Accuracy。RK3576 low 快照受并行 trace
容器争用；RK3588 全程保留 voice/RKLLM/agent 业务，均未重置 NPU 或停止业务。

最终 ARM64 multi-stage 镜像 ID 为 `sha256:5ceaf23a7370...`，两板内容一致。
`docker image inspect` 为 257,793,213 B，`docker save | gzip -1` 为
255,849,560 B（SHA256
`c3e26ce8340e7a560a077d46661aa3040e848eb8d723f5196f35b525ef89a0a3`）。首个
可工作但含完整 Debian `plugins-bad` 的版本为 342,271,154 B；改为只读挂载宿主
parser plugin/codecparser ABI 后减少 84,477,941 B。runtime audit 确认无
compiler、pybind11 headers/module/cache，C++ `.so` 已 strip（200,232 B）。保留的
Python GI 与 GStreamer tools/base/good 是 RTSP/depay/appsink 动态运行依赖。
原始 NDJSON/summary 位于两平台 `results/*-mpp-cpp-*-20260813.*`。

## Jetson 多 context 与静态 batch 对比（2026-08-13）

两机均为 MAXN_SUPER、SM87、TensorRT 10.3.0.30、FP16、640×640，测试命令
使用 `trtexec --useCudaGraph --noDataTransfers`。多 context 固定 batch=1，
`--infStreams=N` 让一个反序列化 engine 创建 N 个独立 execution context / CUDA
stream / buffer；每路吞吐按 aggregate/N 计算。Nano 测试时另有一套真实单路
fall-detection RTSP 容器运行，NX 保留 edge-LLM 和 voice 服务，未停止业务、未做
GPU reset，因此这是共存容量，不冒充空闲峰值。

| 设备/Profile | context | aggregate FPS | FPS/路 | mean/P95 | CPU | trtexec RSS mean/max | GPU | 板功耗 mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Orin Nano / YOLO11s | 1 | 69.66 | 69.66 | 14.35/22.02 ms | 8.9% | 249.5/269.7 MiB | 94.8% | 16.81/17.85 W |
| Orin Nano / YOLO11s | 2 | 71.12 | 35.56 | 28.11/34.14 ms | 8.6% | 261.9/283.0 MiB | 93.1% | 17.57/18.77 W |
| Orin Nano / YOLO11s | 3 | 70.07 | 23.36 | 42.75/48.49 ms | 11.3% | 273.2/295.3 MiB | 91.3% | 17.47/18.53 W |
| Orin Nano / YOLO11s | 4 | 70.15 | 17.54 | 56.78/64.11 ms | 12.7% | 284.0/307.0 MiB | 91.2% | 17.70/18.73 W |
| Orin Nano / YOLO11s | 6 | 70.85 | 11.81 | 84.53/92.77 ms | 15.7% | 306.0/330.7 MiB | 91.2% | 17.63/18.53 W |
| Orin NX / YOLO11m | 1 | 53.83 | 53.83 | 18.57/18.60 ms | 6.9% | 203.3/219.7 MiB | 91.4% | 23.87/25.16 W |
| Orin NX / YOLO11m | 2 | 54.27 | 27.14 | 36.83/37.04 ms | 8.2% | 214.6/231.9 MiB | 91.4% | 24.97/26.39 W |
| Orin NX / YOLO11m | 3 | 53.60 | 17.87 | 55.80/60.34 ms | 9.6% | 226.4/244.6 MiB | 91.4% | 24.85/26.23 W |
| Orin NX / YOLO11m | 4 | 53.60 | 13.40 | 74.31/89.40 ms | 10.7% | 237.4/256.6 MiB | 91.4% | 25.04/26.38 W |
| Orin NX / YOLO11m | 6 | 53.54 | 8.92 | 110.85/127.91 ms | 13.7% | 260.3/281.4 MiB | 91.4% | 25.33/26.68 W |

15 FPS 的纯推理边界为 Nano 4 路 YOLO11s、NX 3 路 YOLO11m；6 context
都稳定完成，但单路已低于 15 FPS。总吞吐随 context 数几乎不变，说明并发
context 是分配同一 GPU 预算，而不是让吞吐线性倍增。实际产品还要为 RTSP、
NVDEC/VIC、H2D、预处理、NMS、tracker 与 MQTT 留余量。

| 设备/Profile | 调度 | engine | batch/s | image/s | batch mean/P95 | 结论 |
|---|---|---:|---:|---:|---:|---|
| Nano / YOLO11s | batch1，4 context | 28,217,060 B | — | 70.15 aggregate | 56.78/64.11 ms（每 context） | 17.54 FPS/路，低延迟独立流 |
| Nano / YOLO11s | static batch4 | 23,033,340 B | 44.99 | **179.94** | 22.23/22.79 ms | 吞吐约空闲 batch1 的 2.20×；需凑批 |
| NX / YOLO11m | batch1，3 context | 51,571,548 B | — | 53.60 aggregate | 55.80/60.34 ms（每 context） | 17.87 FPS/路，低延迟独立流 |
| NX / YOLO11m | static batch4 | 44,795,132 B | 26.49 | **105.96** | 37.75/38.02 ms | 吞吐为 batch1 的 1.97×；需凑批 |

batch4 当前只是 benchmark artifact，**不是已部署 ABI**。现有 native bridge
强制 batch=1，且每个 `TrtBridge` 会完整反序列化一次 engine；它并发安全但比
“进程级共享 `ICudaEngine` + 每路独立 context/stream/buffer”更占内存。若启用
static batch，需要新增跨流 bounded-wait micro-batcher、按 batch 维拆分输出，并把
每个 slot 返回原 `stream_id` 的独立 tracker/时序状态。单相机 15 FPS 累积 batch4
理论上先增加约 200 ms 等待；跨相机凑批可降低等待，但受最慢
RTSP 流和抖动影响。实时报警默认继续使用 batch1 多 context。

构建固定 shape 的完整参数为
`--minShapes/--optShapes/--maxShapes=images:Bx3x640x640 --fp16
--builderOptimizationLevel=3 --memPoolSize=workspace:1024`，没有修改 ONNX 图。
可审计汇总为
[`orin-nano-yolo11s-multicontext-trt10.3.json`](../platforms/jetson/evaluation/orin-nano-yolo11s-multicontext-trt10.3.json)
和
[`orin-nx-yolo11m-multicontext-trt10.3.json`](../platforms/jetson/evaluation/orin-nx-yolo11m-multicontext-trt10.3.json)。

## Raspberry Pi + Hailo-8 多路 RTSP 压测（2026-08-30）

受控源为 Spark LAN MediaMTX：H.264 Constrained Baseline，640x640，15 FPS，约
1.2 Mbps，GOP 30。每组 warmup 10 秒、测量 60 秒，目标为每路至少 14.5 FPS。
本轮 `ENABLE_MQTT=OFF`（Pi 缺少 mosquitto development headers），吞吐口径为
RTSP → 软件解码 → Hailo → pose decode/tracker → payload construction，不包含 broker
publish，也不能替换既有 MQTT contract 证据。

| 组别 | 路数 | FPS min/max | probe latency range | 判定 |
|---|---:|---:|---:|---|
| A queue=2/drop=false | 15 | 15.0207/15.1040 | 53.27–57.42 ms | pass |
| A queue=2/drop=false | 16 | 14.6152/14.6652 | 44.30–51.75 ms | pass |
| A queue=2/drop=false | 17 | 13.0236/13.0570 | 44.7402–51.3382 ms | fail |
| B queue=1/drop=false | 15 | 14.9320/14.9987 | 40.53–43.72 ms | pass |
| B queue=1/drop=false | 16 | 14.5215/14.5715 | 36.16–40.93 ms | pass |
| B queue=1/drop=false | 17 | 13.2264/13.2597 | 39.4761–46.0263 ms | fail |
| B queue=1/drop=false | 18 | 11.5100/11.5600 | 39.3743–51.4715 ms | fail |
| C queue=1/drop=true | 15 | 14.6652/14.8485 | 37.03–40.78 ms | pass |
| C queue=1/drop=true | 16 | 14.3982/14.5815 | 41.99–45.85 ms | fail |

当前目标阈值下最大通过配置为 B 的 16 路。synthetic `test://ball` 在 15 路达到
最高 15 FPS，16 路失败，不能代替真实 RTSP 结果。queue=1 的主要作用是减少在途旧帧
和陈旧帧延迟，不宣称提升吞吐。C 的 drop-on-latency=true 在 16 路最低 FPS 为
14.3982，低于目标，因此生产默认保持关闭。

最终源码在 Pi 上以 `BUILD_APP=ON`、`ENABLE_MQTT=OFF` 做 Release 构建并通过 4/4
CTest。只设置 `STREAMS`、10 秒 warmup 和 60 秒测量时间，保留 queue/drop 默认值的
16 路复验为 14.5828–14.6328 FPS/路；CPU 239%、RSS 1,258,256 KiB、70.8°C。

B17 资源快照为 CPU 246%、RSS 1,366,592 KiB、温度 78.5°C、`get_throttled=0x80000`。
该值表示启动以来曾触发 soft temperature limit；当前状态位 `0x8` 未置位，因此这份快照
不表示采样时仍处于 soft temperature limit。原始结构化记录见
[`rpi-hailo8-multistream-20260830.json`](reports/rpi-hailo8-multistream-20260830.json)。

## 2026-08-30 官方 YOLOv8m-Pose HEF

本轮采用 Hailo Model Zoo v2.19.0 官方 Hailo-8 HEF：
[`yolov8m_pose.hef`](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov8m_pose.hef)。
文件大小 31,608,992 B，SHA256 为
`fa0bfbf83dba494f4d75ec2fd0ef497ca9d402a65c324afc9865ffc327a53514`，解析为
HAILO8、3 contexts、9 raw outputs。DFC2.19 PCIe Gen3x4 官方表为 batch1 68.6
FPS、batch8 145 FPS，不是 Pi 端到端数字。

| 测试 | 路数 | FPS | 延迟 | 结果 |
|---|---:|---|---|---|
| HailoRT 裸 benchmark | 1 | 30.87–30.98 | HW 26.92–26.97 ms | 稳定采样 |
| HailoRT 裸 benchmark，batch 4 | 1 network group | 69.38 total | HW 45.67 ms/batch | throughput probe |
| HailoRT 裸 benchmark，batch 8 | 1 network group | 86.91 total | HW 71.08 ms/batch | throughput probe |
| synthetic app | 1 | 30.0 | 26.7402 ms | pass |
| synthetic app | 2 | 15.4862 / 15.4695 | 60.9354 / 60.3006 ms | pass |
| synthetic app | 3 | 10.3143 / 10.3143 / 10.3309 | 87.557 / 89.0384 / 89.2127 ms | fail |
| RTSP 640²@15 | 2 | 15.0098 / 14.9932 | 40.7077 / 43.8206 ms | pass |
| RTSP 640²@15 | 3 | 10.3278 / 10.3278 / 10.3111 | 90.737 / 90.1096 / 88.6653 ms | fail |

目标为每路至少 14.5 FPS，官方 YOLOv8m-Pose 的 Pi RTSP 最大通过路数为
**2 路**。测试使用 `ENABLE_MQTT=OFF`（Pi 缺 mosquitto development headers），
口径为 RTSP→软解→Hailo→decode/tracker→payload construction，不含 broker publish，
不能替换旧 MQTT contract 证据。三路 synthetic CPU/RSS 为 45.9% / 263920 KiB，
三路 RTSP 为 35.2% / 328032 KiB，温度 59.8–61.5°C。

裸 HEF 的 batch 4/8 相比 batch 1 分别为 2.25/2.81 倍吞吐，说明 batch 能摊薄
multi-context 切换成本；它没有把三个 context 变成并行执行。当前每个 GStreamer
pipeline 独立提交 batch 1，因此该数据不提高上表已实测的 2 路 RTSP 上限。若要利用
batch，需要增加跨流 frame collector/batcher，并重新测试组批等待、尾延迟和公平性。

自编译 m 模型不计入结果：64 张标定图使优化等级降为 1，GPU noise-analysis
收到损坏的 TensorFlow device name；跳过该诊断后 QAT 完成。allocator 仍在执行
multi-context 搜索时发现官方 HEF，搜索约 3.5 小时后被主动停止，并非 allocator
报错或超时。完整结构化证据见
[`reports/rpi-hailo8-yolov8m-pose-20260830.json`](reports/rpi-hailo8-yolov8m-pose-20260830.json)。
