# 复盘与 demo 视频

## 复盘合成视频

主脚本加 `--record` 时，运行结束后会自动生成 `first_person.mp4`（960×800：相机 + VLM 面板 + 俯视图 / 场景图 / 语义场）。也可以单独重做：

```bash
python scripts/make_cgfm_composite_video.py RUN_DIR --fps 1.5
```

**注意**：脚本会改名已有的 `--out` 文件（`first_person.mp4` → `first_person_orig.mp4`，仅当后者不存在时）。对已经整理好的运行目录重做时请用 `--out /别的路径.mp4`，不要覆盖原文件。

## demo 视频（带第三人称录像）

1920×1080，左侧第三人称录像 + VLM 问答卡，右侧第一人称相机、事件流和三张地图，底部时间线。

```bash
# 1) 每个 tick 的面板图（--out 指到别处）
python scripts/make_cgfm_composite_video.py RUN_DIR --out /tmp/x.mp4 --fps 1.5 --panels-dir PANELS_DIR
# 2) 单目标 demo（这里的 --front-start 是第三人称视频第一帧的墙钟时间）
python scripts/make_cgfm_demo_front.py RUN_DIR --front RUN_DIR/third_person.MOV --front-start HH:MM:SS.sss --panels PANELS_DIR --out demo.mp4
# 3) 终身（多目标）demo：标题 "Life-long Object Navigation"、目标队列进度条、回忆阶段、按目标分段的时间线
python scripts/make_cgfm_demo_lifelong.py RUN_DIR --front RUN_DIR/third_person.MOV \
    --front-start HH:MM:SS.sss --front-hold SECONDS --post-speed 1.5 --panels PANELS_DIR --out demo.mp4
```

关键参数（终身版）：`--front-start` 是**面板开始的墙钟时间**（配合 `--front-hold`，第三人称第一帧定格这么多秒后才开始播放）；`--front-rate` 是第三人称视频相对墙钟的速率（默认 1.0）；`--post-speed 1.5` 让 360° 扫描之后的所有内容（第三人称和面板一起）按 1.5 倍速播放；`--scan-speed`（默认 2）是扫描阶段的倍速；`--fp-lag` 是帧文件存盘相对拍摄的延迟；`--blend` 在 tick 之间做淡入淡出（实测很多人觉得眩晕，默认关闭）。

运行目录要用 `cp -a` 复制：每个 tick 的墙钟时间就是 `frames/tick_N.jpg` 的修改时间。

## 如何对齐第三人称视频（经验）

第三人称视频是手持录的，没有可用的时钟，必须靠画面里的事件对齐：

1. **用开头最明确的画面**。例如“第三人称开始录制时，机器人已经在 360° 扫描里转了约 90°”，对应日志里 `rotate to +90deg ok` 之后的 tick（航向 84°）。这是最可靠的锚点。
2. 再用中段和结尾的事件交叉验证：走到目标 0.5 m 内的墙钟时间（日志里 `dist=` 降到 ≤ 0.5 m 的 tick），在视频里找机器人停在目标旁的画面。
3. **不要只用一个端点、也不要用肉眼估计的“到达”拟合速率**：这会把起点和速率一起带偏。我们试过 “视频时长 = 最后一个 tick 到视频结尾” 估起点，再用三次到达事件拟合 “起点 + 速率”，得到的结果在不同事件处差了 3–4 s，后来按开头的姿态线索重新对齐才对上。
4. 视频文件自带的创建时间和音轨与机器人运动的互相关都不可靠（相关系数只有 0.1–0.4）。
5. tick 之间平均相隔约 1.3 s（不是 1 s），第一人称画面取“拍摄时间最近的 tick”，不要取“已经过去的最近一个”，否则第一人称平均落后约 0.7 s。
