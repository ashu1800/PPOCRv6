# PPOCRv6

Windows C++ OCR HTTP service for PPOCR ncnn models.

## Build

Install or provide these dependencies:

- CMake 3.24+
- C++17 compiler
- nlohmann/json
- OpenCV `core`, `imgcodecs`, `imgproc`
- ncnn built with Vulkan if GPU inference is required

Example:

```powershell
cmake -S . -B build-msvc-ncnn -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:\Users\Ashu\Desktop\PPOCRv6\third_party\ncnn-20260526-windows-vs2022\x64;C:\ProgramData\miniconda3\Library"
cmake --build build-msvc-ncnn --config Release
```

For dependency smoke tests without ncnn:

```powershell
cmake -S . -B build -G Ninja -DPPOCR_WITH_NCNN=OFF -DCMAKE_PREFIX_PATH="C:\ProgramData\miniconda3\Library"
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

## Runtime

Place `PPOCRv6.exe` beside the existing `models` directory and start it from that runtime directory.

On first start the service creates `config.json`:

```json
{
  "model_level": "small",
  "listen_host": "0.0.0.0",
  "port": 8080,
  "api_key": "<generated>",
  "prefer_gpu": true,
  "gpu_device": 0,
  "max_concurrent_requests": 2,
  "queue_size": 8,
  "max_request_body_bytes": 16777216,
  "rec_max_width": 960,
  "enable_orientation_retry": true
}
```

Startup logs include the selected inference mode:

- `Inference mode: GPU(Vulkan)` when ncnn Vulkan initializes successfully.
- `Inference mode: CPU` plus `CPU fallback reason: ...` when GPU is unavailable.

## API

Health:

```http
GET /health
```

OCR:

```http
POST /ocr
X-API-Key: <config api_key>
Content-Type: application/json

{"image_base64":"..."}
```

Response shape:

```json
{
  "full_text": "",
  "items": [
    {
      "text": "example",
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

`items[].box` is ordered as top-left, top-right, bottom-right, bottom-left. Very long recognition crops are split and merged automatically; `rec_max_width` controls the segment width after the crop is too wide for direct recognition. When `enable_orientation_retry` is true, low-confidence horizontal crops are retried after a 180 degree rotation and the higher-confidence result is returned.
