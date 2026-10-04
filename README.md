# Personal Lens Atlas

> Windows 11 上的本地 CPU 眼神校正原型：使用个人“看镜头”参考眼睛库，探索比小瞳孔贴图更自然的实时 Eye Contact 预览。

`Python 3.11` · `MediaPipe Tasks` · `OpenCV` · `NumPy` · `Windows 11` · `CPU-only`

## English summary

Personal Lens Atlas is a CPU-only Windows MVP for personalized eye-contact preview. It captures a person's real camera-lens eye references across nine head poses, matches them to live MediaPipe landmarks, and composites only inside the current eyelid aperture. It is an experimental local preview, not a virtual camera or an NVIDIA Broadcast replacement.

## 项目状态

这是一个**工程验证 MVP**，不是 NVIDIA Broadcast 的复刻，也不是通用的三维/生成式眼神重建模型。

当前已实现：本地摄像头预览、九姿态个人参考采集、离线参考库构建、实时 Original / Lens Reference 对比，以及眼皮保护的眼区合成。

当前未实现：虚拟摄像头输出、会议软件接入、任意头部角度的自然重建、跨用户通用模型。

## 需求定义与工程推进

这个项目从一开始就不是泛泛的“让瞳孔移动一下”。项目发起阶段把目标、约束和验收顺序写成了可执行的工程协议：

- **真实使用场景**：面试或视频沟通时，使用者看屏幕内容，画面仍尽量呈现为看向物理摄像头镜头；明确不做“眼动追踪鼠标”。
- **硬约束**：Windows 11、Python 3.11、普通 USB/内置摄像头、CPU、MediaPipe、OpenCV；不改动现有 Python，不依赖 RTX、CUDA 或 NVIDIA Broadcast。
- **分阶段验收**：先在本地确认 `Original / Corrected` 的真实视觉效果；只有第一阶段稳定，才进入干净输出、虚拟摄像头和会议软件适配。
- **质量信号可观察**：黑点、双瞳孔、瞳孔越出眼皮、滑条改变但画面无实际改善，都被当作算法失败而不是“再调大一点强度”的问题。
- **隐私边界**：参考视频、眼区图片和姿态数据必须保留在本机，公开代码不能带入任何可识别的人脸素材。

这些约束直接改变了技术路线：早期的局部瞳孔/眼区实验无法通过大幅转眼时的视觉检查，因此当前默认路线改为个人化“看镜头”参考图集，并让眼皮遮罩优先于任何眼睛替换操作。完整的项目背景、决策依据和接手说明见 [PROJECT_CONTEXT_CN.md](PROJECT_CONTEXT_CN.md)。

## 为什么做这个项目

常见的轻量级眼神校正方案只在瞳孔附近移动或绘制少量像素。位移较大时，容易留下黑点、双瞳孔，或让瞳孔和眼皮脱离。

本项目改用一个个人化思路：录制使用者在不同头部姿态中**直视物理摄像头镜头**时的真实眼区。实时运行时，系统为当前姿态挑选最接近的真实参考眼睛，以眼角和眼睑几何对齐后，仅在当前眼皮开口内部融合。这样不会凭空“画”一个瞳孔，也能在不可靠时主动回退到原图。

## 核心能力

- Windows 11、普通内置/USB 摄像头、Python 3.11、CPU 本地运行；不依赖 RTX、CUDA 或 NVIDIA Broadcast。
- 使用当前 MediaPipe Tasks `FaceLandmarker` 获取 478 点面部/虹膜地标，并以眼睛局部坐标和头部姿态信号进行匹配。
- 引导录制 9 个舒适头部姿态；记录 JSON 帧号而非依赖摄像头不可靠的容器 FPS。
- 从本机视频中自动筛选清晰、睁眼、亮度合理且虹膜未贴近眼皮的参考帧。
- 使用仿射对齐、实时眼睑开口遮罩、羽化融合和亮度匹配；闭眼、追踪不稳、姿态超范围时保留原图。
- 包含 Windows 批处理启动脚本、隔离 Python 环境和无摄像头单元测试。

## 架构

```text
                         ┌───────────────────────────┐
Camera ──> MediaPipe ───>│ eye landmarks + head pose  │
                         └─────────────┬─────────────┘
                                       │
reference capture ─> JSON frames ─> atlas builder ─> local eye-reference atlas
                                       │                         │
                                       └── pose / openness match ┘
                                                    │
current eyelid aperture mask ─> affine alignment + safe blend ─> preview
```

## 快速开始（Windows 11）

先安装 64 位 Python 3.11。`setup.bat` 会在项目目录创建独立的 `gaze-env`，不会卸载、替换或降级系统 Python。

```powershell
.\setup.bat
.\capture_reference.bat
.\build_reference_atlas.bat
.\atlas_preview.bat
```

### 使用方式

1. `capture_reference.bat`：按窗口提示，在 9 个自然头部姿态下始终看着物理摄像头镜头。
2. `build_reference_atlas.bat`：按 JSON 帧号处理视频，并为每种姿态的两只眼睛选择可靠的真实参考图。
3. `atlas_preview.bat`：显示 `Original` 与 `Lens Reference` 对比；顶部滑条默认 100%。
4. 保持头部居中，故意看屏幕左、右、上、下方，检查右侧是否仍像在看物理摄像头，而不是出现黑点或双瞳孔。

快捷键：`S` 切换双栏/单栏，`D` 显示调试标记，`Q` 或 `Esc` 退出。

> `main.py` / `run.bat` 保留了早期 ONNX 神经眼区实验，用于历史对照；当前推荐验证路径是 `atlas_preview.bat`。默认安装不会下载该实验模型；如需自行研究，请先阅读 `THIRD_PARTY_NOTICES.md`，再显式运行 `setup.bat --experimental-neural`。

## 隐私与安全边界

- 原始摄像头帧和个人参考视频只在本机处理。
- `reference_samples/` 和 `reference_atlas/` 默认被 `.gitignore` 排除；它们含有可识别的人脸、眼区和姿态数据，请勿使用 `git add -f` 强制提交。
- 本项目不上传摄像头视频，也不捆绑使用者人脸素材。公开发布前请执行 `git status`，确认没有个人录制文件。
- 对闭眼、低光、虹膜过近眼皮、姿态不在参考覆盖范围等情况，渲染器会回退原图。

## 验证

```powershell
.\gaze-env\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖眼睛局部几何、目标/参考仿射关系、眼皮遮罩范围、镜像输入一致性、姿态校准与安全回退等核心逻辑。

## 项目结构

```text
.
├── atlas_preview.py/.bat          # 当前实时个人参考库预览
├── capture_reference.py/.bat      # 九姿态“看镜头”本地采集
├── build_reference_atlas.py/.bat  # 参考帧筛选与小型本机图集构建
├── reference_atlas.py             # 姿态匹配、仿射对齐、眼皮保护与融合
├── face_tracker.py                # MediaPipe Tasks FaceLandmarker 包装
├── gaze_estimator.py              # 局部眼睛/虹膜几何估计
├── camera.py                      # Windows 摄像头探测与回退
├── tests/                         # 无摄像头单元测试
├── requirements.txt               # 经过验证的 Python 3.11 依赖
├── setup_neural_experimental.bat  # 明确确认后才启用的旧 ONNX 实验安装
├── THIRD_PARTY_NOTICES.md         # 第三方模型与许可证说明
├── PROJECT_CONTEXT_CN.md          # 项目背景、决策依据与协作者接手说明
└── docs/PROJECT_NARRATIVE_CN.md   # 公开作品集叙述与面试复盘
```

## 技术要点

### 个人参考图集，而非固定瞳孔贴图

每个参考记录包含头部 yaw / pitch / roll、眼睛开合度、局部眼睑轮廓、虹膜位置、清晰度和亮度。运行时按姿态距离和开合度选择参考，减少“一个固定眼睛贴在脸上”的感觉。

### 镜像坐标处理

摄像头预览通常镜像显示，而采集素材保存为原始方向。项目先把实时镜像画面规范到原始坐标系，按眼睛位置稳定匹配参考图，再转换回显示坐标，避免镜像时左右眼交换造成错误合成。

### 眼皮优先的安全合成

参考眼区不会覆盖整块矩形区域。渲染器同时计算当前眼睑开口和参考眼睑开口，只在二者交集内融合，并在边缘保留保护带与羽化区域。这样眨眼或半闭眼时优先保留真实当前画面。

## 局限与下一步

个人参考库覆盖范围由录制姿态决定，极端侧脸、强遮挡、闭眼和明显光照变化会回退原图。下一阶段需要在视觉效果验证通过后，再增加无覆盖层的干净输出和虚拟摄像头适配，并在会议平台允许的前提下测试。

## 给协作者与 AI Agent 的接手说明

在改代码前请先读 [PROJECT_CONTEXT_CN.md](PROJECT_CONTEXT_CN.md)。当前产品主线是 `atlas_preview.py` + `reference_atlas.py`，而非旧的 ONNX 实验。接手时请保持以下边界：

- 不把项目描述为已实现虚拟摄像头、会议软件接入或 NVIDIA Broadcast 级效果。
- 不以“固定瞳孔贴图”回退当前方案；当前优先级是眼皮安全、真实参考眼区和不可靠时回退原图。
- 不提交 `reference_samples/`、`reference_atlas/`、模型权重或虚拟环境；这些目录包含生物特征或本机生成物。
- 视觉效果必须在真实摄像头画面验证，单元测试只能验证几何、匹配和回退逻辑。

## 项目叙述与面试复盘

可参考 [docs/PROJECT_NARRATIVE_CN.md](docs/PROJECT_NARRATIVE_CN.md)，了解项目的需求定义、实机反馈、技术取舍与可诚实陈述的能力。简历中建议定位为“计算机视觉 / 实时视频处理 MVP”，不要表述为已达到 NVIDIA Broadcast 的质量，或已实现会议软件虚拟摄像头。

## 许可证与第三方组件

本仓库代码采用 [MIT License](LICENSE)。MediaPipe、OpenCV、NumPy、ONNX Runtime 和可选的研究模型遵循各自许可证；详细说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。研究模型只在本机按需下载，不会作为仓库内容重新分发。

