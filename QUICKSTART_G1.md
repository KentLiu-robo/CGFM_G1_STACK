# G1 部署 Quickstart

记录这套仓库在真实 Unitree G1 上的分布式部署方案：感知（雷达+SLAM）跑在 G1 自带的 Jetson 上，规划/推理/运动控制跑在一台独立的开发机上。本文档是给下次接着做的人（或 Claude）快速恢复上下文用的，细节推理过程不重复，只记结论和踩过的坑。

## 架构

```
┌─────────────────────┐         直连网线（经过 G1 内部交换机）        ┌──────────────────────────┐
│  Jetson（G1 自带）    │ ───────────────────────────────────────►  │  开发机（这台机器）         │
│  ROS_DOMAIN_ID=1     │   domain_bridge_g1.yaml（单向转发）         │  ROS_DOMAIN_ID=2          │
│                      │   registered_scan/state_estimation         │  容器 autonomy_stack_planner│
│  - livox_ros_driver2 │                                             │  - terrain_analysis(_ext)  │
│  - arise_slam_mid360 │                                             │  - sensor_scan_generation  │
│  - domain_bridge     │                                             │  - local_planner/pathFollower│
│                      │                                             │  - far_planner              │
│  只做感知，不装/不跑   │                                             │  容器 g1_bridge（单独！）    │
│  任何控制/规划节点     │                                             │  - g1_sdk_bridge（运动控制） │
│                      │                                             │  容器 autonomy_stack_rviz   │
└─────────────────────┘                                             └──────────────────────────┘
                                          g1_sdk_bridge ──unitree_sdk2 (cyclonedds, domain 0)──► G1 本体 .161
```

**日常操作直接用仓库根目录的 `./g1_nav.sh`**（见下文"启动"和"运动控制"），它固化了下面所有的设置和安全规则。

**运动控制单独一个容器**：`g1_loco_driver` 退出时会 `Damp()` 卸力，如果它跟 planner 在同一个容器里，重启 planner 就会让站着的机器人倒下。所以 planner 用 `start_sdk_bridge:=false` 启动，bridge 放在 `g1_bridge` 容器里常驻。

**为什么这么分**：`local_planner`/`terrain_analysis(_ext)`/`sensor_scan_generation`/`far_planner`/`visualization_tools` 全部只依赖 `/state_estimation` + `/registered_scan` 两个 topic（查过源码确认），所以可以完整搬到算力更强、更方便调试的开发机上跑，Jetson 只需要贴近传感器做低延迟的雷达驱动 + SLAM。

**为什么控制走 SDK 不走 WebRTC**：G1 固件 ≥1.5.1 的 WebRTC LAN 握手需要每台设备唯一的 AES-128 key，只能通过 Unitree 云账号获取（`unitree-fetch-aes-key`，有已知 WAF 拦截 bug）。`unitree_sdk2`（DDS）路径不需要这个 key，且原仓库维护者本来就因为这个坑从 WebRTC 切到了 SDK，所以延续这个选择。

## 关键文件

| 文件 | 作用 |
|---|---|
| `src/utilities/domain_bridge/config/domain_bridge_g1.yaml` | domain 1↔2 桥接配置，只转发感知数据（单向），不转发 cmd_vel |
| `src/base_autonomy/vehicle_simulator/launch/system_g1_perception.launch` | Jetson 端 launch：雷达 + SLAM + domain_bridge |
| `src/base_autonomy/vehicle_simulator/launch/system_g1_planner.launch` | 开发机端 launch：规划全家桶 + 可选 `g1_sdk_bridge`。参数：`start_sdk_bridge`（默认 true，**实际用 false**，bridge 单独跑）、`autonomyMode`、`start_path_follower`、`network_interface` |
| `system_perception_g1.sh` / `system_planner_g1.sh` | 两端对应的启动脚本（假设在容器内跑，先 source 好 ROS） |
| **`g1_nav.sh`** | **开发机日常操作脚本**：`up` / `status` / `start` / `enable` / `stop` / `sit` / `damp` / `down` / `down-bridge` / `restart-planner` |
| `docker/fastdds_devmachine_robotnet.xml` | 开发机 Fast DDS profile：ROS 流量只走 `192.168.123.165`（否则 Jetson 数据收不全/收不到，见坑 #6） |
| `src/unitree_g1_sdk_bridge/`（`g1_loco_driver.py`、`g1_sdk_control.py`、launch） | 运动控制 bridge。`network_interface` 可传 IP（按地址绑定）；`start_fsm_id`（默认 501）；打印非 0 的 RPC 返回码 |
| `src/base_autonomy/local_planner/config/unitree/unitree_g1_vlfm.yaml` | G1 慢速配置：0.25 m/s、只前进（`twoWayDrive: false`）。`g1_nav.sh` 默认用它 |
| `system_planner_g1_vlfm.sh`、`src/vlfm_bridge/` | VLFM 语义导航（开发中，见 `src/vlfm_bridge/README.md`） |
| `docker/Dockerfile.sdk` | 已加 `portaudio19-dev`（当时给 WebRTC 方案装的，SDK 方案其实用不上，但留着无害） |
| `src/utilities/livox_ros_driver2/config/MID360_config.json` | **`host_net_info` 四个 IP 已从出厂默认 `.103` 改成 Jetson 实际 `eth0` IP `192.168.123.164`**，不改雷达驱动会 bind failed |

## 网络拓扑（已验证）

- **G1 内部交换机**上挂着：Jetson（`eth0` = `192.168.123.164`）、G1 本体电脑（`eth0` = `192.168.123.161`，hostname `Unitree`，实时内核）、Mid-360 雷达（`192.168.123.120`）
- 开发机通过网线接进这个交换机的空闲口，配了两个 IP：`10.1.1.101/24`（原计划的隔离网段，其实没用上）+ `192.168.123.165/24`（真正在用的，跟 Jetson/机器人同网段）
  - **⚠️ 双 IP 会让 DDS 选错地址**：按网卡名 `enp4s0` 绑定时，DDS 用的是排在前面的 `10.1.1.101`，Jetson/机器人发回来的数据到不了。所以**所有 DDS 都要显式绑定 `192.168.123.165`**：SDK（cyclonedds）用 `network_interface:=192.168.123.165`，ROS（Fast DDS）用 `docker/fastdds_devmachine_robotnet.xml`。`g1_nav.sh` 已经都带上了。
- 开发机给 Jetson 共享了外网（`ip_forward` + `iptables MASQUERADE`，走开发机的 `wlo1`），因为 Jetson 自己没配网关联不上网。**这条 NAT 规则不是持久化的，重启开发机或 Jetson 后需要重新执行**（命令见下方"环境恢复步骤"）。
- Jetson SSH：`unitree@192.168.123.164`，密码问持有人要（没写在这个文件里，这个仓库可能会被 push 到公开/团队 GitHub remote）。只能密码登录；Jetson 上 `docker` 需要 `sudo`。Jetson 时钟比开发机慢约 9.5 分钟（跨机对比时间戳时注意）。
- **Jetson 上没有 RealSense**（2026-09-30 检查：`lsusb` 无 Intel 设备、无 `/dev/video*`），G1 头部 D435i 接在哪块板上待查。

## 环境恢复步骤（重启/重新开始时按顺序做）

### 1. 网线 + IP
两台机器网线接好后：
```bash
# 开发机（这台机器）：
nmcli connection show g1-direct   # 应该已经存在这个 profile，没有的话见下面重建
nmcli connection up g1-direct
ip -brief addr show enp4s0        # 确认有 10.1.1.101/24 和 192.168.123.165/24

# 没有 g1-direct 这个 profile 就重建：
nmcli connection add type ethernet ifname enp4s0 con-name g1-direct \
  ipv4.method manual ipv4.addresses "10.1.1.101/24,192.168.123.165/24" \
  ipv4.never-default yes ipv6.method link-local autoconnect yes
```

### 2. 给 Jetson 供网（可选，只有需要装东西/git pull 时才需要）
```bash
# 开发机：
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -s 192.168.123.0/24 -o wlo1 -j MASQUERADE
sudo iptables -A FORWARD -i enp4s0 -o wlo1 -j ACCEPT
sudo iptables -A FORWARD -i wlo1 -o enp4s0 -m state --state RELATED,ESTABLISHED -j ACCEPT

# Jetson（SSH 进去跑）：
sudo ip route add default via 192.168.123.165 dev eth0 metric 50
```

### 3. 两边 docker 镜像 + 工作区（已经建过，重启机器不需要重建，除非改了代码）
```bash
# 两台机器都一样：
cd G1_STACK && bash docker/build.sh   # 会自动跳过已存在的 :heavy 阶段

# 开发机工作区（跳过感知端专属包）：
docker run --rm -v "$(pwd)":/workspace/autonomy_stack:rw autonomy_stack:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace/autonomy_stack && \
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-skip sam2_detector vlm_nav_bridge livox_ros_driver2 arise_slam_mid360 arise_slam_mid360_msgs unitree_g1_sdk_bridge unitree_webrtc_ros"
# ↑ 注意：unitree_g1_sdk_bridge 其实开发机需要（控制在这边跑），上面这条排除列表是历史遗留，
#   实际按 4 个部分单独补的：serial local_planner（local_planner 依赖它俩）、
#   然后 unitree_g1_sdk_bridge、vehicle_simulator、ros_tcp_endpoint、domain_bridge、far_planner 相关包。
#   最简单的做法：直接 colcon build（不加 --packages-select/skip）全量编一次，省心，只是慢一点。
# 只改了 Python / launch 时（--symlink-install，软链到 src）一般不用重编；
# **新增**了 config/launch 文件（比如 unitree_g1_vlfm.yaml）要重编对应包才会装进 install/：
#   ... colcon build --symlink-install --packages-select local_planner vehicle_simulator unitree_g1_sdk_bridge

# Jetson 工作区（只要感知端）：
docker run --rm -v "$(pwd)":/workspace/autonomy_stack:rw autonomy_stack:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace/autonomy_stack && \
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-select livox_ros_driver2 arise_slam_mid360 arise_slam_mid360_msgs domain_bridge vehicle_simulator ros_tcp_endpoint serial local_planner"
```

### 4. 启动

**Jetson**（先起，让 SLAM 先跑起来；**机器人保持静止**，SLAM 按启动时的姿态初始化）。Jetson 上已经有脚本 `~/jetson_run_perception.sh`，内容就是下面这条（domain 1）：
```bash
ssh unitree@192.168.123.164
sudo bash ~/jetson_run_perception.sh
# 等价于：
sudo docker run -d --name autonomy_stack_perception --network host --ipc host \
  -v "$(pwd)":/workspace/autonomy_stack:rw -e ROBOT_CONFIG_PATH=unitree/unitree_g1 autonomy_stack:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/autonomy_stack/install/setup.bash && \
  export ROS_DOMAIN_ID=1 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  ros2 launch vehicle_simulator system_g1_perception.launch"
```
**`-e ROBOT_CONFIG_PATH=unitree/unitree_g1` 不能少**（坑 #11）：不设的话 SLAM 用默认 `mechanum_drive` 配置，机器人自己的头壳会被当成障碍，localPlanner 找不到任何路径。

**不要**把 Jetson 感知改到 domain 2 直连（试过：SLAM 会劣化，`failureDetected` 暴涨）。

**开发机**（一条命令起 bridge + planner + RViz，bridge 默认 DISABLED，机器人不会动）：
```bash
cd ~/Taowen/G1_STACK
./g1_nav.sh up          # 已经在跑的 g1_bridge 不会被重启
```

<details><summary>等价的手动命令（g1_nav.sh 里做的事）</summary>

```bash
# 运动控制 bridge（单独容器，SDK 绑 IP）
docker run -d --name g1_bridge --network host --ipc host \
  -v "$(pwd)":/workspace/autonomy_stack:rw autonomy_stack:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/autonomy_stack/install/setup.bash && \
  export ROS_DOMAIN_ID=2 RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  ros2 launch unitree_g1_sdk_bridge g1_sdk_control.launch.py network_interface:=192.168.123.165 enable_on_start:=false"

# 规划（Fast DDS 只走 192.168.123.165；G1 慢速配置；不带 bridge）
docker run -d --name autonomy_stack_planner --network host --ipc host \
  -v "$(pwd)":/workspace/autonomy_stack:rw \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/autonomy_stack/docker/fastdds_devmachine_robotnet.xml \
  -e ROBOT_CONFIG_PATH=unitree/unitree_g1_vlfm autonomy_stack:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/autonomy_stack/install/setup.bash && \
  export ROS_DOMAIN_ID=2 RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  ros2 launch vehicle_simulator system_g1_planner.launch use_far_planner:=true start_sdk_bridge:=false network_interface:=192.168.123.165"

# RViz（先 xhost +local:root）
docker run -d --name autonomy_stack_rviz --network host --ipc host \
  -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/autonomy_stack/docker/fastdds_devmachine_robotnet.xml \
  -v "$(pwd)":/workspace/autonomy_stack:rw autonomy_stack:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && source /workspace/autonomy_stack/install/setup.bash && \
  export ROS_DOMAIN_ID=2 RMW_IMPLEMENTATION=rmw_fastrtps_cpp && cd /workspace/autonomy_stack && \
  ros2 run rviz2 rviz2 -d src/base_autonomy/vehicle_simulator/rviz/vehicle_simulator.rviz"
```
</details>

### 5. 验证数据链路（只读，不涉及运动）
```bash
./g1_nav.sh status
```
2026-09-30 实测的正常输出：
```
FSM id: 501 (code 0; ...)            # 机器人用遥控器站起来并 start 之后；刚开机/没站起来时会是别的值
/state_estimation average rate: 50.0
/registered_scan  average rate: 3.3
/terrain_map      average rate: 3.3
/path             average rate: 3.3
/cmd_vel          average rate: 50.0  # pathFollower 在发（未点目标点时是零速）
bridge: cmd_vel forwarding DISABLED
```
`/state_estimation` 没数据或明显低于 50 Hz → planner 容器没带 Fast DDS profile（坑 #6）。`FSM query failed` → SDK 通信不通（坑 #7）。

## 运动控制（涉及机器人真实移动，务必先确认安全）

安全前提：场地空旷（路径上无人/障碍物）、有人手持遥控器随时能物理接管。
**2026-09-30 已实测跑通**：腰部 3 自由度 G1，站在地上，`cmd_vel` 0.5 rad/s 原地转向约 40°。**RViz 点 Waypoint 自主行走还没实测。**

```bash
# 0) 机器人用遥控器站起来：零力矩 -> 阻尼 -> 锁定站立（预备模式），放到地上
#    （进入"走跑模式" = FSM 802 也没关系，下一步会切走）
./g1_nav.sh start      # 切到 FSM 501（3 自由度腰的 SDK 常规模式），应返回 "SetFsmId(501) -> code 0"
./g1_nav.sh enable     # 放行 cmd_vel（此时 pathFollower 发零速，机器人不动）
# 1) 最小测试：原地左转 0.5 rad/s，3 秒（此时没开 planner 更干净，开着也行，见下方说明）
./g1_nav.sh ros2 'topic pub --times 60 -r 20 /cmd_vel geometry_msgs/msg/TwistStamped "{twist: {angular: {z: 0.5}}}"'
# 2) 导航测试：RViz 顶部工具栏选 Waypoint，在机器人正前方 1~1.5 m 空地点一下
#    （WaypointTool 同时发 /way_point 和 /joy，把 localPlanner/pathFollower 切到自主模式）
./g1_nav.sh stop       # 随时停止，机器人保持站立（建议单独开一个终端备好）
./g1_nav.sh damp       # 软件卸力，机器人会软下去，只在挂吊架/已蹲下时用（会要求确认）
```
- 速度：`g1_nav.sh` 用 `unitree_g1_vlfm.yaml`，0.25 m/s、只前进。要换配置：`ROBOT_CONFIG=unitree/unitree_g1 ./g1_nav.sh restart-planner`（重启 planner 是安全的，里面没有 bridge）。bridge 本身还有 0.6 m/s / 0.8 rad/s 的硬上限。
- 手动发 `/cmd_vel` 时如果 planner 开着，pathFollower 也在以 50 Hz 发零速，会跟你的指令打架——做手动测试时先 `./g1_nav.sh down`（只关 planner/RViz，不关 bridge）。

**⚠️ 安全行为**：
- `g1_loco_driver` 退出（停/删 `g1_bridge` 容器、Ctrl-C、崩溃、重启 Docker）时会 **`Damp()` 卸力**——机器人站着时**绝不能**停 bridge，先 `./g1_nav.sh sit` 或用遥控器蹲下/挂吊架。
- `stand_up` 服务是 `Damp -> Squat2StandUp`，**先卸力**；机器人已经用遥控器站起来时不要调用。

### 关闭

```bash
./g1_nav.sh stop         # 1. 停止转发（机器人保持站立）
./g1_nav.sh down         # 2. 关 planner + RViz（不关 bridge）
ssh unitree@192.168.123.164 'sudo docker rm -f autonomy_stack_perception'   # 3. 关 Jetson 感知
# 4. 机器人蹲下（遥控器，或 ./g1_nav.sh sit）或挂上吊架之后，才能：
./g1_nav.sh down-bridge  # 会要求确认
```

### 之前"发 cmd_vel 不动"的真正原因（2026-09-30 已解决）

1. **SDK DDS 根本不通**：开发机 `enp4s0` 有两个 IP（`10.1.1.101` 在前、`192.168.123.165` 在后），按网卡名绑定时 cyclonedds 用了 `10.1.1.101`，收不到机器人任何数据（`rt/lowstate` 0 条），所有 LocoClient RPC 返回 **3102（请求没发出去）**；原驱动又忽略 `Move()` 返回值，所以看起来"不报错但无效"，`Start()` 等应答则一直卡到超时。**修复**：`network_interface` 传 IP `192.168.123.165`（驱动检测到 IP 时按地址绑定），`rt/lowstate` ~1 kHz，RPC 返回 0。
2. **FSM 不对**：`LocoClient.Start()` 固定切 **500**（1 自由度腰机型的常规模式），3 自由度腰机型要 **501**。遥控器进入的"走跑模式"是 **802**：在 802 下 `SetVelocity` 返回 0 但**不执行**，`SetFsmId(500)` 返回 0 但**不切换**。`SetFsmId(501)` 后 cmd_vel 生效。**修复**：bridge 新参数 `start_fsm_id`（默认 501），`start` 服务改为 `SetFsmId(start_fsm_id)` 并返回机器人真实返回码；速度指令改用 `SetVelocity(...,1.0)`（等价 `Move()`），返回码非 0 时打印告警。

排查口诀：先订阅 `rt/lowstate` 看有没有数据、调 `GetFsmId` 看返回码——**3102** = 通信不通（查网卡/IP 绑定），**3104** = 超时（服务不在，可能是调试模式），返回 0 但不动 = 查 FSM（要 501）。
另：G1 手柄 **`L2+R2` / `L2+A` / `L2+B`** 会进入调试模式、关闭 AI sport client（[unitree_sdk2_python#43](https://github.com/unitreerobotics/unitree_sdk2_python/issues/43)），也会让运动指令失效，只能重启 G1 退出——但 09-29/09-30 的"不动"实际是上面两条。

## 踩过的坑（已修复，供参考）

1. `ExecuteProcess` 应该从 `launch.actions` 导入，不是 `launch_ros.actions`（笔误）。
2. `domain_bridge` 可执行文件不能裸名 `ExecuteProcess(cmd=['domain_bridge', ...])` 调用，ROS2 C++ 可执行文件不在 PATH 上，要用 domain_bridge 包自带的 `launch/domain_bridge.launch`（走 `$(exec-in-pkg)` 解析真实路径）。
3. `arise_slam_mid360` 的官方 launch 文件会读 `local_planner` 包的配置文件（共享参数），即使不跑 `local_planner` 节点，Jetson 端也得把它编出来（连带需要 `serial`，`local_planner` 依赖它）。
4. `MID360_config.json` 的 `host_net_info` 四个 IP 出厂默认是 `.103`，必须改成接收主机（Jetson eth0）实际 IP `192.168.123.164`，否则雷达驱动 `bind failed`。
5. rsync 增量同步时 `--exclude=log`（不加前导 `/`）会误删仓库里所有叫 `log` 的子目录（比如 `vehicle_simulator/log/`、`or-tools` 里 abseil 的 `log/`），要写成 `--exclude=/log` 才只排除顶层编译产物目录。
6. **开发机双 IP 导致 ROS 跨机数据收不到/收不全**（2026-09-30）：planner 一加入 domain 2，Jetson 的 domain_bridge 就停发（连 Jetson 本地 domain 2 都收不到），没 planner 时本机也只收到 ~28 Hz。原因同坑 #7：开发机 Fast DDS 对外通告了 Jetson 不可达的 `10.1.1.101`。修复：开发机所有 ROS 容器加 `-e FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/autonomy_stack/docker/fastdds_devmachine_robotnet.xml`（UDPv4 白名单只留 `192.168.123.165`），之后 state 50 Hz、scan/terrain/path 3.3 Hz。Jetson 端不用改。
7. **SDK（cyclonedds）按网卡名绑定选错 IP**：见上文"之前发 cmd_vel 不动的真正原因"第 1 条，`network_interface` 要传 `192.168.123.165`。
8. **3 自由度腰的 G1 要 FSM 501**：`LocoClient.Start()` 固定切 500，对这台机器无效；遥控器"走跑模式"802 下 SDK 速度指令返回 0 但不执行。见上文第 2 条。
9. **bridge 退出会 Damp**：`g1_loco_driver` 的 SIGINT/SIGTERM 处理会 `Damp()`，所以 bridge 必须和 planner 分容器，机器人站立时不能停 bridge。
11. **Jetson SLAM 没设 `ROBOT_CONFIG_PATH` → RViz 点 Waypoint 机器人不动**（2026-10-02）：`arize_slam.launch.py` 默认读 `mechanum_drive.yaml`，盲区盒只有 |y|<0.1 m。G1 雷达正下方的头壳/脖子（雷达系 x −0.06–0.04、y ±0.10–0.17、z −0.05 到 −0.15）漏进 `/registered_scan`，每帧约 290 个点贴在机器人身上被当成障碍 → `/free_paths` 为 0、`/path` 只有 1 个点、`/cmd_vel` 全 0。修复：Jetson 容器加 `-e ROBOT_CONFIG_PATH=unitree/unitree_g1`（盲区 |x|<0.2、|y|<0.3），之后 0.6 m 内自身点为 0，`free_paths` 约 25000。
10. 在 `bash -c "... pkill -f <pattern> ..."` 里用 `pkill -f` 会连同这个 bash 自己一起杀掉（命令行里含 pattern，退出码 143）。要在容器里杀进程，用 `docker exec <容器> pkill -f <pattern>`，不要套 `bash -c`。
