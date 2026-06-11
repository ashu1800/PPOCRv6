# PPOCRv6

PPOCRv6 是一个 Windows C++ OCR HTTP 服务，使用 PPOCR ncnn 模型进行文字识别。

## 构建

需要安装或提供以下依赖：

- CMake 3.24+
- C++17 编译器
- nlohmann/json
- OpenCV `core`、`imgcodecs`、`imgproc`
- 如果需要 GPU 推理，需要启用 Vulkan 的 ncnn

构建示例：

```powershell
cmake -S . -B build-msvc-ncnn -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:\Users\Ashu\Desktop\PPOCRv6\third_party\ncnn-20260526-windows-vs2022\x64;C:\ProgramData\miniconda3\Library"
cmake --build build-msvc-ncnn --config Release
```

如果只想在没有 ncnn 的情况下做依赖冒烟测试：

```powershell
cmake -S . -B build -G Ninja -DPPOCR_WITH_NCNN=OFF -DCMAKE_PREFIX_PATH="C:\ProgramData\miniconda3\Library"
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

## 运行

将 `PPOCRv6.exe` 放在已有 `models` 目录旁边，并从该运行目录启动程序。

首次启动时，服务会在运行目录下创建 `config.json`：

```json
{
  "model_level": "small",
  "listen_host": "0.0.0.0",
  "port": 8080,
  "api_key": "<自动生成>",
  "prefer_gpu": true,
  "gpu_device": 0,
  "max_concurrent_requests": 2,
  "queue_size": 8,
  "ncnn_threads_per_request": 1,
  "max_request_body_bytes": 16777216,
  "rec_max_width": 960,
  "rec_direct_max_width": 4096,
  "enable_orientation_retry": true
}
```

启动日志会显示当前推理模式：

- `Inference mode: GPU(Vulkan)` 表示 ncnn Vulkan 初始化成功，正在使用 GPU 推理。
- `Inference mode: CPU` 加 `CPU fallback reason: ...` 表示 GPU 不可用，服务已退回 CPU 推理。
- `Model levels` 会列出 `medium`、`small`、`tiny` 的速度和精度取舍。

## API

健康检查：

```http
GET /health
```

OCR 识别：

```http
POST /ocr
X-API-Key: <config api_key>
Content-Type: application/json

{"image_base64":"..."}
```

返回格式：

```json
{
  "full_text": "",
  "items": [
    {
      "text": "示例",
      "confidence": 0.98,
      "box": [
        {"x": 0, "y": 0},
        {"x": 100, "y": 0},
        {"x": 100, "y": 30},
        {"x": 0, "y": 30}
      ]
    }
  ]
}
```

`items[].box` 按左上、右上、右下、左下的顺序返回。非常长的识别裁剪图会自动分段并合并文本；`rec_direct_max_width` 控制从直接识别切换到分段识别的阈值，`rec_max_width` 控制每个识别分段的宽度。`enable_orientation_retry` 为 `true` 时，低置信度的横向文本裁剪图会额外尝试旋转 180 度识别，并且只有新结果通过置信度和长度保护条件时才会被采用。

`max_concurrent_requests`、`queue_size` 和 `max_request_body_bytes` 在配置校验阶段有上限限制，用于避免意外耗尽资源。`ncnn_threads_per_request` 用于独立控制每次 ncnn 推理使用的 CPU 线程数，不再和 HTTP 请求并发数绑定。
