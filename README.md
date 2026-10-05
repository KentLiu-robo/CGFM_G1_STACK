# CGFM_G1_STACK

**CGFM on a real legged robot: open-vocabulary ObjectNav and life-long (multi-target) navigation with a CLIP scene graph, a semantic frontier map and a VLM gatekeeper.**

> English summary — The robot builds a lightweight 3-D scene graph online (YOLO-World boxes → depth point clouds → CLIP ViT-H-14 features), diffuses the CLIP similarity of every node to the target text over the explored free space to score frontiers, and lets a small VLM (Qwen2.5-VL-3B) decide which scene-graph node is the target (with a crop check). In life-long mode the target queue advances after every find, the scene graph/maps persist, and a target that was already seen is recalled by sending its coordinates straight to the base navigation stack instead of exploring. The code is the real-robot deployment on a Unitree Go2 (Livox Mid-360 + RealSense), derived from VLFM's real-robot loop and MSGNav's CGFM components. See [`docs/`](docs/) for details. Status: research code, validated in a handful of indoor real-robot runs; no benchmark numbers.

> 仓库名里的 “G1” 沿用建仓时的命名；目前代码实际部署和验证的平台是 **Unitree Go2**。

---

## 特性

- **在线场景图**（`cgfm/scene_graph.py`）：YOLO-World 检测框 → 深度反投影（去地面 + DBSCAN）→ CLIP ViT-H-14 特征；按点云重叠 + CLIP 相似度关联和合并，同帧同时出现的物体之间建立边。
- **语义场 + frontier 打分**（`cgfm/semantic_map.py`、`cgfm/frontier_scorer.py`）：把每个节点对目标文字的 CLIP 相似度沿已探索自由空间做 Dijkstra 扩散，frontier 得分 = 语义邻域均值 / (1 + λ·测地距离)，带最小距离、迟滞和停滞黑名单。
- **VLM 守门**（`cgfm/vlm_selector.py`）：Qwen2.5-VL 看到场景图（节点 id + 类别、同帧关系、证据图、当前视图）后只回答目标的 id，**不输出坐标**；被选中的节点还要过一次裁剪图选择题复核，或被同一节点反复选中（投票规则）才会锁定。
- **终身（多目标）导航**（`scripts/run_cgfm_lifelong.py`）：目标队列 `--targets "a,b,c"`，找到一个后自动切到下一个，场景图和地图保留；切换时做记忆检查——**已经找到过的目标直接取它的坐标发给底层导航栈走回去**，见过没找到过的交给 VLM 判断，没见过的继续探索。
- **实机安全保护**：扫描前后检查 SLAM 健康；运行中位姿跳变、位姿停发或冻结、SLAM 进程崩溃都会停止；Ctrl-C / `stop` 先杀 `pathFollower` 再尽力 `/liedown`；`--dry-run` 完全不发运动指令。
- **复盘与 demo 工具**：复盘合成视频（相机 + VLM 面板 + 俯视图 / 场景图 / 语义场）和带第三人称录像的 demo 视频。

## 系统结构

```
Jetson (机载)                      开发机 (PC, ROS 2 Jazzy)
┌────────────────────┐  WiFi      ┌───────────────────────────────────────────────────────────┐
│ Livox Mid-360 驱动  │──DDS─────▶│ 底层导航栈（不在本仓库）：SLAM → scan/terrain →            │
│ RealSense 推流服务  │──TCP:6000─▶│  localPlanner → pathFollower → /cmd_vel → WebRTC → Go2    │
└────────────────────┘            │                     ▲  UDP relay (pose / waypoint / cmdvel)│
                                  │  ┌──────────────────┴───────────────────────────────┐     │
                                  │  │ CGFM（本仓库）scripts/run_cgfm_*.py + cgfm/       │     │
                                  │  │ YOLO-World(:12185) → 场景图 → 语义场 → frontier    │     │
                                  │  │ Qwen2.5-VL(:8101) 选目标 → /way_point              │     │
                                  │  └───────────────────────────────────────────────────┘     │
                                  └───────────────────────────────────────────────────────────┘
```

CGFM 只做“上层决策”：它读位姿和 RGB-D，发一个个 `/way_point`（经 UDP relay），避障和走路都交给底层导航栈。详细说明见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 仓库结构

```
cgfm/        核心库：scene_graph / semantic_map / frontier_scorer / vlm_selector / obstacle_map_cgfm / floor_map / palette
scripts/     run_cgfm_pipeline.py（单目标）、run_cgfm_lifelong.py（多目标终身导航）、
             make_cgfm_composite_video.py / make_cgfm_demo_front.py / make_cgfm_demo_lifelong.py（视频）、
             pose/waypoint/cmdvel_udp_relay.py（ROS ↔ UDP 中转，主脚本自动拉起）
tools/       bringup / teardown / health_check（拉起、关闭、检查底层栈）、Qwen 与 YOLO-World 服务启动脚本、env.sh
configs/     paths.example.env（所有路径和地址集中在这里）
docs/        ARCHITECTURE / QUICKSTART / KNOWN_ISSUES / DEMO_VIDEOS
```

## 依赖与外部组件

本仓库**不能独立运行**，还需要下面这些（都不在本仓库里）：

| 组件 | 用途 | 来源 |
|---|---|---|
| `vlfm` 包 + YOLO-World 服务 | `ObstacleMap`、几何工具、开放词汇检测服务（端口 12185） | https://github.com/bdaiinstitute/vlfm （MIT） |
| `frontier_exploration` 包 | `reveal_fog_of_war`、`wrap_heading`（随 vlfm 安装） | vlfm 的依赖 |
| 底层导航栈（ROS 2） | SLAM、terrain、localPlanner、pathFollower、`unitree_webrtc_ros`（`/cmd_vel`、`/liedown`） | 一份快照见 https://github.com/KentLiu-robo/VLFM_GO2_STACK 的 `dev_machine/` |
| Jetson 端服务 | Livox 驱动；RealSense TCP 推流服务（端口 6000） | 推流服务见上面仓库的 `jetson/` |
| Qwen2.5-VL 服务 | 目标选择（vLLM 0.7.2 + transformers 4.49.0，AWQ 量化，约 3.4 GB） | `tools/launch_cgfm_vlm_server.sh`，需要单独的 Python 环境 |
| Python 包 | 见 `requirements.txt`（Python 3.9） | — |

硬件：Unitree Go2、Livox Mid-360、RealSense（相机朝前）、开发机至少一块 24 GB 显卡（CLIP 约 2 GB，YOLO-World 另算；Qwen 单独占一块卡更稳）。

## 快速开始

```bash
git clone https://github.com/KentLiu-robo/CGFM_G1_STACK.git && cd CGFM_G1_STACK
pip install -r requirements.txt            # 然后按 vlfm 的说明安装 vlfm / frontier_exploration
cp configs/paths.example.env configs/paths.env   # 编辑：Jetson 地址、底层栈 workspace、vlfm 路径、Python 解释器
set -a; . configs/paths.env; set +a        # Python 脚本读环境变量

bash tools/bringup_nav_stack.sh            # 拉起底层栈 + YOLO-World，最后应为 VERDICT: READY
bash tools/launch_cgfm_vlm_server.sh &     # Qwen 服务（端口 8101，约 40 s）；不用 --vlm 可以跳过

# 先 dry-run（机器人不会收到任何指令），再真机（会要求输入 ARM；手里必须拿着遥控器）
python scripts/run_cgfm_lifelong.py --targets "trash can,plant,trash can" --vlm --record --dry-run
python scripts/run_cgfm_lifelong.py --targets "trash can,plant,trash can" --vlm --record --max-seconds 900
python scripts/run_cgfm_pipeline.py --target "trash can" --vlm --record      # 单目标：找到就停并趴下
```

运行中可以在终端输入 `target <名>`（立即切换）、`queue <名>`（追加到队尾）、`stop`（安全停止）。完整步骤、开跑前检查清单和关闭流程见 [`docs/QUICKSTART.md`](docs/QUICKSTART.md)。

## 结果目录

每次运行写一个目录到 `$CGFM_RES_DIR`（默认 `results/single_target` 或 `results/life_long`）：`log.txt`、`frames/`、`depth/`、`occupancy_map/`、`scene_graph/`、`semantic_map/`、`sg_images/`、`vlm_events.jsonl`、`vlm_log.jsonl`、`scene_graph.json/.txt`、`lifelong_summary.json`（终身模式）、`first_person.mp4`（复盘合成视频）。这些文件可以很大，已在 `.gitignore` 里。

## 安全

这是在真实机器人上运行的代码。**每次真机运行都必须有人手持遥控器，随时可以物理接管**；软件的停止（杀 `pathFollower`、`/liedown`）是尽力而为，不能替代它。先 `--dry-run`，不要在机器人周围 0.6 m 内留有椅子底座、桌腿等低矮障碍（局部规划器不能倒车，会被夹住，见 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)）。

## 已知问题

WiFi 下 IMU 经 DDS 丢包导致 SLAM 重置、挪动机器人后 SLAM 垂直方向发散、相机画面卡帧、局部规划器被低矮障碍夹住、VLM 裁剪复核会误拒真目标、探索耗尽后没有自动放弃目标等，详见 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)。

## 致谢与许可

MIT 许可，见 [`LICENSE`](LICENSE)。代码的一部分改编自 [MSGNav](https://github.com/Yuxiang-Xiao/MSGNav)（CGFM 分支）、[VLFM](https://github.com/bdaiinstitute/vlfm) 和 [ConceptGraphs](https://github.com/concept-graphs/concept-graphs)，它们同为 MIT 许可，版权声明见 [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)。
