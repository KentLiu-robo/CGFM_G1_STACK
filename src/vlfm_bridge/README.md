# vlfm_bridge (G1)

把 [VLFM](https://github.com/bdaiinstitute/vlfm)（BLIP-2 ITM 语义打分 + YOLO-World 开放词汇检测 + frontier value map）接到 G1 导航栈上：vlfm 选出探索点，发布成 `/way_point`，再由 `localPlanner`/`pathFollower` 避障走过去，`/cmd_vel` 经 `g1_sdk_bridge` 控制机器人。

移植自团队 GO2 上实机跑通的 [VLFM_GO2_STACK](https://github.com/KentLiu-robo/VLFM_GO2_STACK)（`dev_machine/vlfm_bridge` + `jetson/`），整体架构和协议保持一致，这里只说明 G1 上的差异。**不是 colcon 包**（有 `COLCON_IGNORE`），都是独立脚本。

| 文件 | 在哪跑 | 作用 |
|---|---|---|
| `jetson/realsense_stream_server.py` | Jetson (192.168.123.164) | D435i 彩色+深度 → TCP 6000（GO2 版 + 深度对齐到彩色、发送彩色内参） |
| `scripts/{pose,waypoint,cmdvel}_udp_relay.py` | planner 容器内（由主脚本拉起） | vlfm conda 环境里没有 rclpy，三个中继用 127.0.0.1 UDP 代为收发 ROS 话题（照搬） |
| `scripts/run_vlfm_pipeline_g1.py` | 宿主机，`vlfm_g1` conda 环境 | 主脚本：360° 扫描 → 探索 → 锁定目标 → 停车 |
| `scripts/calibrate_camera_floor.py` | 宿主机，`vlfm_g1` 环境 | 相机外参/地面高度标定（不动机器人） |
| `scripts/make_composite_video.py` | 宿主机 | `--record` 结束后合成复盘视频（照搬） |
| `scripts/reconstruct_timeline.py` | 宿主机 | 旧运行目录（2026-10-05 16:58 之前）事后反推 `timeline.csv` / `events.csv` |

相关改动（仓库其它位置）：
- `g1_nav.sh up-vlfm` / `restart-planner-vlfm` / `vlm-up` / `vlm-down`：日常启动入口（等价于 `system_planner_g1_vlfm.sh` 的参数 + 容器 profile）。
- `system_planner_g1_vlfm.sh`：planner 的 VLFM 版本（手动在容器里跑时用；容器需自带 `FASTRTPS_DEFAULT_PROFILES_FILE`，并加 `start_sdk_bridge:=false`）（`use_far_planner:=false`、`autonomyMode:=true`、`start_path_follower:=false`、`ROBOT_CONFIG_PATH=unitree/unitree_g1_vlfm`）。
- `local_planner.launch.py` 新增 `start_path_follower` 参数（默认 `true`，现有 launch 行为不变）；`system_g1_planner.launch` 透传 `autonomyMode`、`start_path_follower`（默认值同原行为）。
- `config/unitree/unitree_g1_vlfm.yaml`：`unitree_g1.yaml` 的副本，`twoWayDrive: false`、速度 0.25 m/s、`maxYawRate` 30、`dirDiffThre` 1.0（与 GO2 VLFM 实跑参数一致）。

## 跟 GO2 版的差异（为什么这么改）

1. **停车绝不趴下/卸力**。GO2 结束时调 `/liedown`；G1 对应的 `damp` 会让人形直接倒地。G1 版：先 kill pathFollower（bridge 0.5 s 收不到指令自动归零），再 `/g1_sdk_bridge/enable false`，机器人保持站立。`damp` 只留给人工急停。
2. **ROS 在容器里**。本机是 20.04 没有原生 Jazzy，中继、pathFollower、`ros2` 命令都通过 `docker exec autonomy_stack_planner` 执行（容器 `--network host`，UDP 照通）。容器进程属 root，宿主机 `pkill` 杀不掉且静默失败，所以 `_pkill()` 走 `docker exec` 并**验证**进程确实没了，失败会红字报警。
3. **`/cmd_vel`、`/way_point` 唯一发布者**。扫描阶段 `cmdvel_udp_relay` 独占 `/cmd_vel`，之后才启动 pathFollower；far_planner 关掉。主脚本启动前会检查（`check_preconditions`），不满足就不启动。
4. **相机位姿用 SLAM 完整 6-DoF**（`G1_VLFM_POSE_MODE=full`，默认）。雷达和 D435i 都刚性装在头部，雷达位姿已经包含步态带来的头部俯仰/横滚晃动；GO2 版只用 (x, y, yaw) + 固定相机高度，在人形上会把地面误判成障碍。`planar` 模式保留 GO2 行为用于对比。
6. **数据过期即视为缺失**（2026-10-05）：相机帧超过 0.5 s、位姿超过 0.3 s（按本机接收时间算，Jetson 时钟不准）就当作没有数据，不会再拿旧画面配新位姿；运动模式下连续 3 s 没有新数据就停车。
7. **扫描没转完就不进入自主行走**：转向过程中漂移、没有位姿、转向无响应、连续超时，都会直接结束本次运行，不启动 pathFollower。
8. **锁定后**：`/way_point` 发在目标前方 0.6 m 处（停靠点，避免 localPlanner 把目标本身当障碍）；到达判断改成 10 Hz，离目标 ≤0.5 m 或离停靠点 ≤0.35 m 就停车，所以实际停在离目标约 0.6–0.95 m 处。
5. **参数**：到达半径 0.25 → 0.5 m；地面余量 0.10 → 0.15 m；障碍高度上限按 G1 身高算到地面以上 1.5 m。

## 相机外参（未标定前主脚本拒绝运动）

全部通过环境变量给出，主脚本启动时会打印当前值：

| 变量 | 含义 | 默认（占位，未测量） |
|---|---|---|
| `G1_CAM_TX/TY/TZ` | 相机在 SLAM sensor（雷达）系下的位置，m | 0.05 / 0 / -0.10 |
| `G1_CAM_PITCH_DOWN_DEG` | 相对雷达系向下俯仰（+ 为朝下） | 30 |
| `G1_CAM_ROLL_DEG` / `G1_CAM_YAW_DEG` | 横滚 / 偏航 | 0 / 0 |
| `G1_FLOOR_Z` | full 模式：地面在 SLAM map 系的 z（≈ −雷达启动时离地高度） | -1.20 |
| `G1_CAM_HEIGHT` | planar 模式：相机离地高度 | 1.10 |
| `G1_CAM_CALIBRATED` | 标定完设为 `1`，否则只允许 `--perception-only` | 0 |

标定：机器人静止站在平地、前方 2 m 以上空地，跑 `calibrate_camera_floor.py`，按提示迭代 export，直到前后/左右坡度 < 0.01。

**已标定（2026-10-02 16:20）**：`g1_camera_calib.env`（pitch_down 50.0°、roll 2.4°、floor_z −1.34；只对 Jetson SLAM 用 `ROBOT_CONFIG_PATH=unitree/unitree_g1` 时有效），运行主脚本前 `source` 它。`G1_FLOOR_Z` 依赖 **SLAM 在机器人站立时启动**；如果 SLAM 是在吊架上或蹲着时启动的，要重新标定 floor_z。

## 运行顺序

前提：G1 的运动控制链路已确认可用（见 `QUICKSTART_G1.md`）。

```bash
# 0. 改动过 launch/config，需要在容器里重编 local_planner 和 vehicle_simulator（config 是拷贝安装的）
docker run --rm -v "$(pwd)":/workspace/autonomy_stack:rw autonomy_stack:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace/autonomy_stack && \
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select local_planner vehicle_simulator"

# 1. Jetson：感知（同 QUICKSTART_G1.md）+ 相机推流（pip3 install --user pyrealsense2==2.55.1.6486，Jetson 无外网需本机下 aarch64 cp38 wheel 再拷过去）
#    Unitree 的 videohub_pc4 占着彩色流（/dev/video4），master_service 每 5 s 会把它拉起来，
#    所以停掉后要立刻启动推流，抢先占住设备（停后再拉起的 videohub 会打不开相机，不影响行走）：
sudo /unitree/sbin/start-stop-daemon --stop --pidfile=/unitree/var/run/videohub_pc4.pid \
  --exec /unitree/module/video_hub_pc4/videohub_pc4; python3 realsense_stream_server.py   # 在 Jetson 上

# 2. 本机：g1_bridge + VLFM 版 planner（带 Fast DDS 白名单 profile、start_sdk_bridge:=false）+ rviz
./g1_nav.sh up-vlfm          # 已在跑的 g1_bridge 不会被重启；只换 planner 用 ./g1_nav.sh restart-planner-vlfm

# 3. 本机：VLM 服务（~/Taowen/vlfm_g1 = 官方 vlfm@584ed56 + GO2 补丁，conda env vlfm_g1，约 6 GB 显存）
./g1_nav.sh vlm-up           # 日志 /tmp/g1_vlm/；./g1_nav.sh status 会显示端口是否在听；用完 ./g1_nav.sh vlm-down

# 4. 不动机器人的联调（推着/抱着走，看 occupancy/value map，RViz 里看 /way_point）
cd ~/Taowen/vlfm_g1 && source ~/Taowen/G1_STACK/src/vlfm_bridge/g1_camera_calib.env
env -u PYTHONPATH PYTHONPATH=$(pwd) ~/anaconda3/envs/vlfm_g1/bin/python \
  ~/Taowen/G1_STACK/src/vlfm_bridge/scripts/run_vlfm_pipeline_g1.py --target chair --perception-only

# 5. 实机（人手持遥控器，分步）：遥控器站立放地上 → start(FSM 501) → enable true → 主脚本 → 输入 ARM
#    （流程和原因见 QUICKSTART_G1.md「运动控制」；stand_up 会先卸力，机器人已站立时不要用）
ros2 service call /g1_sdk_bridge/start std_srvs/srv/Trigger              # 在容器内执行；应返回 SetFsmId(501) -> code 0
ros2 service call /g1_sdk_bridge/enable std_srvs/srv/SetBool "{data: true}"
env -u PYTHONPATH PYTHONPATH=$(pwd) ~/anaconda3/envs/vlfm_g1/bin/python \
  ~/Taowen/G1_STACK/src/vlfm_bridge/scripts/run_vlfm_pipeline_g1.py --target chair --max-seconds 300 --record
```

每次运行还会写 `events.csv`（关键事件 + 真实时间）、`timeline.csv`（每帧一行）、`trajectory.csv`（20 Hz 位姿）、`cmd_vel.csv`；加 `--record-realtime` 再录 30 fps 实时画面（`first_person_realtime.mp4` + `realtime_frames.csv`），用于和第三人称视频剪辑对齐。时间都是开发机时钟。

运行中输入 `target <名字>` 换目标，`stop` 或 Ctrl-C 停止（kill pathFollower + enable false，机器人保持站立）。结果在 `~/Taowen/RES_G1/vlfm_pipeline_<时间>/`（`G1_VLFM_RES_DIR` 可改）。

## 尚未在 G1 实机验证

- ~~Jetson 上 D435i~~：2026-10-02 已验证（USB3、30 fps、对齐深度 → 本机 TCP 30 fps / 23 Mbit/s）。
- ~~相机外参~~：2026-10-02 已标定（`g1_camera_calib.env`），perception-only 实测地面可通行、椅子 2.5 m 处锁定。
- 扫描阶段原地转向：`SCAN_ROT_MIN_WZ`（GO2 死区值 0.25 rad/s）是否适合 G1 行走控制器。
- `enable false` 在高负载下的响应时间（GO2 上 `/liedown` 曾因负载超时）。
