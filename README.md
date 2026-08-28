# YOLO 引体向上计数

这个小程序使用 Ultralytics YOLO 姿态模型检测视频中的人体，叠加人体框和骨架，并根据头部、肩部与单杠的相对位置完成引体向上计数。输出画面包含英文计数 HUD、动作状态、人体框和 17 点姿态骨架。

## 功能

- YOLO11 Pose 人体姿态检测；
- 自动选择画面中置信度最高的人物；
- 根据双手腕位置估计单杠高度；
- 鼻子关键点优先、肩膀关键点遮挡补偿；
- 状态机滞回和连续帧防抖，避免重复计数；
- 输出带标注的 MP4 和逐帧 CSV；
- 默认支持 CPU 推理，无需自行训练模型。

## 运行

在项目目录中执行：

```powershell
python -m pip install -r requirements.txt
python pullup_counter.py --source video/input/test.mp4
```

需要查看单杠和头部相对位置时，可以添加 `--debug`。

默认输出：

- `video/output/test_counted.mp4`：带人体框、骨架、动作状态和计数的演示视频
- `video/output/test_counted.csv`：逐帧记录，方便检查计数阈值

首次运行会自动下载 `yolo11n-pose.pt`。如果希望提高姿态精度，可以使用更大的模型：

```powershell
python pullup_counter.py --model yolo11s-pose.pt --source video/input/test.mp4
```

CPU 环境下建议先使用默认的 `yolo11n-pose.pt`。如果人物较小或关键点不稳定，可把 `--imgsz` 调到 `960`，或者把 `--keypoint-conf` 调低到 `0.25`。

## 计数逻辑

程序会自动用双手腕位置估计单杠高度，然后计算关键点相对单杠和人体框高度的位置：

- 鼻子接近/越过单杠：进入最高点并计数；
- 横杠遮挡面部时，使用肩膀位置作为最高点补偿信号；
- 鼻子明显低于单杠：回到下放状态；
- 上下使用不同阈值，并要求连续多帧稳定，避免抖动重复计数。

如果视频的动作标准与默认阈值不同，直接修改 `PullUpCounter.update()` 里的 `top_threshold` 和 `bottom_threshold` 即可。调试运行时，画面左上角会显示 `nose/bar` 比值，CSV 里也会记录该值。

## 仓库说明

出于隐私和仓库体积考虑，人物测试视频、推理输出和模型权重不会提交。请把自己的视频放入 `video/input/`；YOLO 权重会在首次运行时自动下载。
