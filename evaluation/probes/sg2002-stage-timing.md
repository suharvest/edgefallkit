# reCamera 2002 分段计时探针

一次性的测量改动，**没有合入发布版本**，本文件保存它的做法与结果，供复现与追溯。

## 为什么需要

出厂应用只上报一个 `inference_time_ms`：起点在 `retrieveFrame` 之后、终点在 `detectAll`
之后，也就是 letterbox + CVI 推理 + raw-head 解码 + 应用后处理的总和，而且
`duration_cast<milliseconds>` 在**测量时**就截断成整数（实测只出现 52/53/54 三个值）。

后果是跨平台性能表里 2002 那一行的「加速器推理」列填不了，只能写「不可分离」。

## 改了什么

四处，全部为新增，不改变既有字段的语义：

1. `main.cpp` — 计时改用 `microseconds` 再除 1000；毫秒转换发生在测量之后而非之中
2. `pose_detector.{h,cpp}` — 在 `model_->run()` 两侧和其后的应用处理两侧各打一对时间戳，
   通过 `modelRunMs()` / `postprocessMs()` 暴露
3. `result_payload.{h,cpp}` — 新增 `model_run_ms` / `postprocess_ms` / `pipeline_ms`，
   并用 `inference_metric` 标注 `model_run_ms` 的范围
4. `inference_time_ms` **保持原样**（整数、detectAll 全程），以便与冻结基线对比

## 实测结果（2026-08-20，固件出厂配置，640² INT8 YOLO11n-Pose）

两组各 300 帧，一组无人、一组画面中恒有 1 人：

| | 无人 | 有人（1 人） |
|---|---:|---:|
| `model_run_ms` | 52.650 ms | 52.739 ms |
| `postprocess_ms` | 0.002 ms | 0.012 ms |
| `pipeline_ms` | 53.127 ms | 53.227 ms |
| `inference_time_ms`（原字段） | 53.017 | 53.033 |

原字段两组均值与冻结基线 52.96 ms 一致，因此本次测量与历史数据可比。

## 结论

- **应用层后处理只占 0.002–0.012 ms**。此前文档推测"整数毫秒分辨不出约 2 ms 的增量"，
  那 2 ms 并不存在。
- **`model_run_ms` 仍不是纯加速器**。letterbox、NPU 推理、raw-head 解码三步封装在
  sscma-micro 的单次 `model_->run()` 内，应用层没有接缝可打点；要再拆需改上游库。
  它是该平台可测的最窄推理区间，不等同于 Jetson 的 `trtexec` 或 RK 的
  `rknnlite.inference()`。
- 有人时后处理涨到 6 倍但绝对值仍极小——与 Jetson、Hailo 同类，因为解码遍历固定数量的
  anchor，与画面里有几个人无关。

## 复现

补丁见同目录 `sg2002-stage-timing.patch`。应用到
`sscma-example-sg200x/solutions/fall-detection/`，在 SG200X SDK 容器里构建
（`cmake -B build -DCMAKE_BUILD_TYPE=Release . && cmake --build build && cd build && cpack`），
安装生成的 deb 即可。测完请重装官方包还原设备：

    https://sensecraft-statics.seeed.cc/solution-app/recamera_ecosystem/packages/fall-detection_0.2.0_riscv64.deb
    sha256 a7e3347a706f8045767f4e10768db88a316cbecb1cc879480c9edcedfb872e8d
