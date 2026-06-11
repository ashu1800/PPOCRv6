# 项目概述

PPOCRv6 是一个 C++ 无 UI OCR 服务项目，目标产物为单个 `PPOCRv6.exe`。程序启动时读取或生成运行目录下的 `config.json`，按配置加载 `models/medium`、`models/small` 或 `models/tiny` 中的 PPOCR ncnn 模型，并通过 HTTP REST 接口接收 base64 图片进行文字识别。

服务优先使用 ncnn Vulkan GPU 推理；如果 Vulkan 初始化或模型 GPU 加载失败，自动退回 CPU 推理，并在启动日志和 `/health` 响应中显示当前推理模式。

## 模块结构

- `CMakeLists.txt`：C++17 构建入口，配置 ncnn、OpenCV、nlohmann/json 和测试目标。
- `include/ppocr`：公共头文件，包含配置、日志、HTTP 服务、OCR engine、base64 和队列接口。
- `src/config.cpp`：首次启动生成 `config.json`，并校验模型级别、端口、API Key、并发、识别分段宽度、直识别宽度和方向重试参数。
- `src/http_server.cpp`：HTTP 服务实现，提供 `GET /health` 和 `POST /ocr`，使用 API Key 鉴权与有界队列控制并发，并在 OCR worker 内解码 base64。
- `src/ocr_engine.cpp`：OCR runtime 初始化、模型文件校验、ncnn GPU 优先和 CPU fallback、四点透视裁剪、长文本切分和低置信度 180 度重试逻辑。
- `src/ocr_postprocess.cpp`：CTC 解码、稳定文本框四点排序、动态阅读顺序排序、UTF-8 安全文本段合并和方向重试选择等 OCR 后处理工具。
- `tests/test_main.cpp`：基础单元测试，覆盖配置、base64、队列和后处理。
- `models`：外置 PPOCR ncnn 模型目录，每个级别包含 `det.ncnn.param/bin`、`rec.ncnn.param/bin`、`keys.txt`。

## 变更历史

[2026-06-11] 日志 - 启动后输出 `models` 支持级别说明，列出 medium/small/tiny 的速度和精度取舍。
[2026-06-11] 测试 - 将单元测试从 `assert` 迁移为 Release 下生效的异常式 `check`。
[2026-06-11] OCR识别 - 加固四点排序退化场景、方向重试选择保护和超长识别宽度配置语义。
[2026-06-11] HTTP服务 - 提前释放 OCR 请求大字符串和 worker 解码后的 base64 缓冲，降低并发大图内存峰值。
[2026-06-11] 配置 - 新增 `rec_direct_max_width`，用于控制直识别与分段识别阈值。
[2026-06-11] OCR识别 - 优化四点透视裁剪、动态 padding、长文本分段、低置信度 180 度重试和动态阅读顺序排序。
[2026-06-11] HTTP服务 - 将 OCR base64 解码移动到 worker 内，队列满时避免无效大图解码。
[2026-06-11] 配置 - 新增 `rec_max_width` 和 `enable_orientation_retry`，并记录 `items[].box` 稳定点序。
[2026-06-11] HTTP服务 - 加固请求体大小限制、socket 超时、完整响应发送、线程生命周期和端口绑定失败处理。
[2026-06-11] 日志 - 每次 OCR 请求完成或失败时输出识别耗时。
[2026-06-11] OCR识别 - 修复 GPU FP16 导致的空识别结果，并为宽截图增加行级文本分割 fallback。
[2026-06-11] 日志 - 启动后输出 HTTP 接口调用方法、鉴权 Header、请求 JSON 和返回 JSON 格式。
[2026-06-11] 依赖 - 安装并适配 ncnn 20260526 Windows VS2022 x64，本地支持真实 ncnn Vulkan 构建。
[2026-06-11] OCR服务 - 新增 C++ ncnn HTTP OCR 服务骨架，支持配置生成、API Key、有界并发、GPU 优先与 CPU fallback 日志。
