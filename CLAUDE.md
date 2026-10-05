# CLAUDE.md — 在新电脑上部署和使用 CGFM_G1_STACK

给 Claude Code 的说明。目标：在一台新开发机上**一步一步部署、逐级验证**，并按这个项目已经形成的流程协助日常实验。先读 `README.md`（总览）和 `docs/ARCHITECTURE.md`（原理），部署时按本文第 2 节执行。和维护者沟通用中文。

## 1. 这是什么、文件在哪

Unitree Go2 上的 CGFM 实机导航：CLIP 场景图 + 语义 frontier + Qwen2.5-VL 选目标，支持单目标和终身（多目标、已找到的目标直接回忆）。本仓库只含**上层决策代码**；底层 ROS 导航栈、Jetson 端服务、`vlfm` 包都是外部依赖（README 依赖表）。

| 路径 | 作用 |
|---|---|
| `scripts/run_cgfm_lifelong.py` / `run_cgfm_pipeline.py` | 终身 / 单目标主脚本（两个互相独立，改一个别连带另一个） |
| `cgfm/` | 核心库，两个主脚本共用；改动必须对单目标行为无副作用 |
| `scripts/make_cgfm_*.py` | 复盘 / demo 视频 |
| `scripts/*_udp_relay.py` | ROS ↔ UDP 中转，主脚本自动拉起，不要手动启动 |
| `tools/` | 拉栈 / 关栈 / 健康检查 / 服务启动；所有地址和路径在 `tools/env.sh` |
| `configs/paths.example.env` | 复制成 `configs/paths.env`（已 gitignore）后填写 |

## 2. 铁律（先看这一节）

1. **永远不要替用户 ARM 机器人。** 真机运行由用户在自己的终端输入 `ARM`、手持遥控器启动；你只给命令。你可以自己跑 `--dry-run`（机器人不会收到任何指令）。除非用户明确要求，不要调用 `/hello`、`/standup`、`/liedown` 等会让机器人动的服务。
2. **给命令前先检查导航栈**（第 4 节清单）。用户说“检查导航栈”就是做这件事。
3. **开跑前挪动过机器人 → 必须重启整个导航栈**（SLAM 在站立处初始化；挪动后 z 可能漂到 -10 m）。重启前先问用户机器人是否已放好、站稳。
4. **shell 坑**：`teardown_nav_stack.sh` 会杀进程组，可能把调用它的 shell 一起带走（退出码 144）→ `(setsid bash tools/teardown_nav_stack.sh > /tmp/td.log 2>&1 < /dev/null &)` 后台启动，**下一条命令再**读日志，等到最后一行 `teardown: dev stack procs left=0 ...`。`pkill -f` / `pgrep -f` 会匹配到你自己的命令行，用方括号写法 `pkill -f "[v]llm.entrypoints"` 或按 PID 杀。不要对带变量的路径 `rm -r "$VAR/*"`。
5. 用户在手动调整机器人或打断你的长命令时，**检查要轻量**，别连续跑 30 s 以上的采样；先问。
6. 公开仓库：**不要提交**个人路径（`/home/<user>`）、邮箱、令牌、实验结果、视频、权重；`LICENSE` 保持标准 MIT 文本（多一句话 GitHub 就识别不出）；提交用 noreply 邮箱（`git config user.email`）。除非用户要求，不要 push。
7. 改完 Python 先 `python -m py_compile`，再用 `--dry-run` 短跑（20–30 s）才算验证；改 shell 脚本先 `bash -n`。

## 3. 新机器部署步骤

每一步都有验证命令，**通过再进下一步**。需要用户提供的信息见第 6 节，先问清楚再动手。

### 3.1 前置条件
Ubuntu + ROS 2 Jazzy（底层栈用）、NVIDIA 驱动 + CUDA、至少一块 24 GB 显卡（Qwen 建议单独占一块卡；只有一块卡时设 `CGFM_VLM_GPU=0` 并调低 `CGFM_VLM_GPU_UTIL`）、conda。验证：`nvidia-smi`、`ls /opt/ros`（要有 `jazzy`）。

### 3.2 克隆与配置
```bash
git clone https://github.com/KentLiu-robo/CGFM_G1_STACK.git && cd CGFM_G1_STACK
cp configs/paths.example.env configs/paths.env    # 按第 6 节填写
set -a; . configs/paths.env; set +a                # Python 脚本读环境变量（每个新终端都要执行）
```

### 3.3 Python 环境（主脚本 / YOLO-World 用，Python 3.9）
```bash
conda create -n vlfm python=3.9 -y && conda activate vlfm
pip install -r requirements.txt                   # torch 请装与驱动匹配的 CUDA 版本
git clone https://github.com/bdaiinstitute/vlfm.git "$VLFM_REPO"   # 按其 README 安装 vlfm 和 frontier_exploration
```
验证（应打印 `imports ok; cuda: True`）：
```bash
PYTHONPATH=$VLFM_REPO $VLFM_PYTHON - <<'EOF'
import torch, open_clip, open3d, scipy, cv2, requests, vlfm, frontier_exploration
print("imports ok; cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
EOF
```
主脚本启动时需要 `vlfm` 可导入：把 `$VLFM_REPO` 加进 `PYTHONPATH`，或 `pip install -e`。

### 3.4 模型权重
- **CLIP ViT-H-14**（`laion2b_s32b_b79k`，主脚本首次启动时由 open_clip 自动下载到 HF 缓存）。
- **YOLO-World**：`$VLFM_REPO/data/yolov8s-worldv2.pt`（首次使用自动下载）。**注意**：首次 `set_classes()` 会触发 `ultralytics` 的 AutoUpdate，安装 `clip` 包、下载约 340 MB 权重，并可能把 Pillow 升到 11.x；这是一次性的。
- **Qwen2.5-VL-3B-Instruct-AWQ**：`tools/launch_cgfm_vlm_server.sh` 设了 `HF_HUB_OFFLINE=1`，所以**必须提前下载**：`huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct-AWQ`（约 3.4 GB）。
验证：`ls ~/.cache/huggingface/hub | grep -iE "qwen2.5-vl|CLIP-ViT-H-14"` 应看到 `models--Qwen--Qwen2.5-VL-3B-Instruct-AWQ` 和 `models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K`。

### 3.5 Qwen 服务（单独的 Python 环境）
需要 **vLLM 0.7.2 + transformers 4.49.0** 的环境，不要装进 `vlfm` 环境（版本会冲突）。把它的解释器写进 `CGFM_VLM_PYTHON`。`tools/vllm_compat/sitecustomize.py` 是一个导入补丁（别名 `Qwen2_5_VLImageProcessor`），启动脚本会自动加进 `PYTHONPATH`。
```bash
CGFM_VLM_PYTHON=/path/to/vllm_env/bin/python setsid bash tools/launch_cgfm_vlm_server.sh > /tmp/cgfm_vlm.log 2>&1 &
curl -s http://127.0.0.1:8101/v1/models | grep qwen     # 约 40 s 后应有输出
```
验证环境版本：`$CGFM_VLM_PYTHON -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"` → `0.7.2 4.49.0`。

### 3.6 YOLO-World 检测服务
```bash
bash tools/launch_yolo_world_server.sh        # 端口 12185；日志 /tmp/cgfm_logs/yolo_world.log
```

### 3.7 底层栈和 Jetson（外部，向用户确认后再动）
本仓库不含这部分。需要：ROS 2 工作空间（SLAM、terrain、localPlanner、pathFollower、`unitree_webrtc_ros`；一份快照见 https://github.com/KentLiu-robo/VLFM_GO2_STACK 的 `dev_machine/`）、Jetson 上的 Livox 驱动和 RealSense 推流服务（同仓库 `jetson/`）、Jetson 与开发机的网络和时间同步（Livox 驱动要求 NTP 已同步）。把路径填进 `configs/paths.env`（`GO2_WS_SETUP`、`GO2_JETSON_IP` 等）。**没有机器人时跳过这一节，只做 3.8 的前三级。**

### 3.8 验证阶梯（逐级，通过标准明确）

| 级 | 做什么 | 通过标准 | 需要机器人 |
|---|---|---|---|
| 1 | `python -m py_compile cgfm/*.py scripts/*.py`；`bash -n tools/*.sh` | 无输出 | 否 |
| 2 | `PYTHONPATH=$VLFM_REPO python scripts/run_cgfm_lifelong.py --help` | 打印参数说明 | 否 |
| 3 | 3.5 / 3.6 的服务 | `curl :8101/v1/models` 有 qwen；`ss -ltn` 里 `:12185` 在监听 | 否 |
| 4 | `bash tools/bringup_nav_stack.sh` | 最后 `VERDICT: READY` | 是 |
| 5 | 第 4 节检查清单 | 全部通过 | 是 |
| 6 | `run_cgfm_lifelong.py ... --dry-run`（20–30 s） | 日志有 tick、无 Traceback；能看到 `[VLM] asked` | 是 |
| 7 | 真机运行 | **由用户 ARM** | 是 |

## 4. 日常流程

**检查导航栈**（用户说“检查导航栈 / 重启并检查”时做）：
1. `bash tools/health_check.sh` → `VERDICT: READY`。
2. SLAM：`/tmp/go2_stack/slam.log` 里 `failureDetected` 应为 0、没有 `process has died`；静止时 `/state_estimation` 的 x、y 波动 < 0.02 m、z ≈ -0.02。
3. **相机画面真的在更新**（`health_check.sh` 只看端口）：数 10 s 内不同帧，彩色约 10 fps、深度约 8 fps；或 `tools/camera_view_check.sh`。
4. 到 Jetson 的 ping 0% 丢包；Qwen 和 YOLO-World 在线；没有残留的 `run_cgfm_*`、relay、`pathFollower`。
5. 机器人周围 0.6 m 内没有低矮障碍（椅子底座、桌腿、纸箱）；可以抓一帧 `/terrain_map` 按方向统计最近障碍距离。

**重启导航栈**：teardown（第 2 节第 4 条的写法）→ 单独确认进程都停了 → `bringup_nav_stack.sh`（同样放后台，等 `stack LEFT RUNNING`）→ 做上面的检查。teardown 默认保留检测服务；Qwen 要自己 `kill -INT <PID>`。

**跑完后分析**（用户说“跑完了”时）：读运行目录的 `log.txt`（`FOUND` / `LOCKED` / `RECALL` / `STALL` / `STOPPING`）、`lifelong_summary.json`、`vlm_events.jsonl`（选中、复核结果、投票）、`go2_obstacle_map_final.png`；对照下表判断原因：

| 现象 | 多半是 |
|---|---|
| `picked #N ... crop check named something else: 'box'/'table'` 反复出现，目标一直没锁 | 复核误拒小而糊的目标（`docs/KNOWN_ISSUES.md` 第 5 条）；投票规则会在同一节点被选 3 次后接受 |
| 锁定后 `Walking toward the goal` 的位置长时间不变，localPlanner 日志 `PATH NOT FOUND ... all directions blocked` | 被低矮障碍夹住（第 4 条），不是 SLAM 发散；先看 `failureDetected` 是否为 0 |
| `frontiers=0` 后只剩 tick 日志 | 探索耗尽，目标不在该区域（第 6 条，尚未自动放弃） |
| `failureDetected` 反复出现 | WiFi 下 IMU 丢包，或挪动机器人后 SLAM 发散（第 1、2 条），需要重启导航栈 |
| `NOT STARTING` / `NOT ARMING` / `STOPPING` | 主脚本的安全检查触发，先查 SLAM 日志 |

## 5. 改代码的约定

- 单目标 `run_cgfm_pipeline.py` 的行为要保持稳定；终身相关的新行为只放进 `run_cgfm_lifelong.py`；`cgfm/` 里的改动必须**默认关闭或不影响单目标**。
- 新阈值写成文件顶部的常量并加注释说明来由（实测数据或日志依据），别散落在逻辑里。
- 视频和 demo 工具：第三人称视频**必须用开头的明确画面**（如“录制开始时机器人已转了约 90°”）加中后段多个事件交叉对齐，不要只用一个端点或肉眼估计的到达时间拟合速率，详见 `docs/DEMO_VIDEOS.md`。
- 对已经整理好的运行目录重做复盘视频时，`make_cgfm_composite_video.py` 加 `--out` 指到别处，别覆盖 `first_person.mp4`。

## 6. 向用户确认的信息（部署前）

1. 有没有真机？机器人型号和相机朝向（本代码按 Go2 + 前向 RealSense 调过：`CAMERA_HEIGHT_M = 0.44`）。
2. Jetson 的 IP 和用户名（`GO2_JETSON_IP`、`GO2_JETSON_USER`）；Jetson 上有没有 `go2-relay` 这类 systemd 服务（没有就设 `GO2_RELAY_UNIT=none`）。
3. 底层导航栈 ROS 工作空间的 `install/setup.bash` 路径（`GO2_WS_SETUP`）、`unitree_webrtc_ros` 的工作空间和虚拟环境、是否需要额外的 DDS 环境脚本（`GO2_ROS_ENV_SCRIPT`）。
4. vlfm 仓库路径（`VLFM_REPO`）和两个 Python 解释器（`VLFM_PYTHON`、`CGFM_VLM_PYTHON`）。
5. GPU 布局：几块卡、Qwen 放哪一块（`CGFM_VLM_GPU`）。
6. 结果目录放哪（`CGFM_RES_DIR`，默认仓库内 `results/`，目录会很大）。
7. 开发机和路由器是有线还是 WiFi（WiFi 下 IMU 丢包更严重，见 `docs/KNOWN_ISSUES.md` 第 1 条）。
