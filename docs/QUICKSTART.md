# 真机快速上手

> 前提：底层导航栈、Jetson 端服务、`vlfm` 包已经装好（见 README 的依赖表），`configs/paths.env` 已填写。所有命令在仓库根目录执行。

## 0. 安全

- **每次真机运行都要有人手持遥控器**，随时可以物理接管。软件的停止是尽力而为。
- 先 `--dry-run`：机器人不会收到任何指令，只记录“将会做什么”。真机会要求输入 `ARM`。
- 机器人周围 0.6 m 内不要有椅子底座、桌腿、纸箱等低矮障碍。相机盲区约 0.7 m，局部规划器不能倒车，被夹住只能原地转。

## 1. 拉起

```bash
set -a; . configs/paths.env; set +a
bash tools/bringup_nav_stack.sh      # Jetson 传感器 → SLAM → WebRTC 控制 → scan/terrain/localPlanner → 检测服务 → 健康检查
bash tools/launch_cgfm_vlm_server.sh > /tmp/cgfm_vlm.log 2>&1 &    # 用 --vlm 才需要；约 40 s
curl -s http://127.0.0.1:8101/v1/models | grep qwen                  # 有输出即就绪
```

`bringup_nav_stack.sh` 最后应为 `VERDICT: READY`。它不会启动 `pathFollower`，所以此时机器人不会走动。**拉起之后到开跑之前不要挪机器人**（SLAM 就在它站的位置初始化）。

## 2. 开跑前检查清单

- `bash tools/health_check.sh` 为 READY。
- SLAM 日志里 `failureDetected` 为 0；静止时位姿稳定（x、y 波动 < 0.02 m，z 接近 0）。
- 相机画面真的在更新。`health_check.sh` 只检查端口是否开着：`bash tools/camera_view_check.sh` 可以抓一帧确认。
- 到 Jetson 的网络 0% 丢包、延迟稳定。
- 没有残留的 `run_cgfm_*`、relay、`pathFollower` 进程。
- 如果开跑前挪动过机器人，**重启一遍导航栈**再开跑（见 KNOWN_ISSUES：挪动后 SLAM 可能垂直发散）。

## 3. 运行

```bash
python scripts/run_cgfm_lifelong.py --targets "trash can,plant,trash can" --vlm --record --dry-run   # 先 dry-run
python scripts/run_cgfm_lifelong.py --targets "trash can,plant,trash can" --vlm --record --max-seconds 900
```

运行中在终端输入：`target <名>` 立即切换目标，`queue <名>` 追加到队尾，`stop`（或 Ctrl-C）安全停止。

日志里的关键字：

| 日志 | 含义 |
|---|---|
| `LOCKED goal=... VLM picked scene-graph #N` | VLM 选中节点并通过复核（或投票），目标已发给底层栈 |
| `RECALL of 'x' found earlier` | 终身模式：已找到过的目标直接取坐标走回去 |
| `FOUND 'x' ... m away` | 走到目标 0.5 m 内，随后切到队列里的下一个 |
| `>>> target next in queue: 'y'. Memory check: ...` | 切换目标，并说明场景图里有没有这类物体 |
| `[STALL] ... blacklisted frontier` | 探索时原地停滞，该 frontier 被拉黑 |
| `NOT STARTING` / `NOT ARMING` / `STOPPING: ...` | 安全检查触发，先查 SLAM 日志，必要时重启导航栈 |

## 4. 关闭

```bash
# teardown 会杀进程组，可能把调用它的 shell 一起带走（退出码 144），所以放后台、单独看结果
(setsid bash tools/teardown_nav_stack.sh --all > /tmp/td.log 2>&1 < /dev/null &)
cat /tmp/td.log     # 最后一行应为: teardown: dev stack procs left=0 ... 
```

`--all` 连检测服务一起关；Qwen 另外关：`ps -eo pid,args | grep "[v]llm.entrypoints.openai.api_server"` 找到 PID 后 `kill -INT <PID>`。teardown 不会让机器人趴下或站起。

## 5. 常见问题

- **相机画面卡在同一帧**：Jetson 上的 RealSense 推流服务卡住了，完整重启导航栈（teardown + bringup 会把 Jetson 上的传感器进程一起重起）。
- **机器人停住而终端还在打印 tick**：运动通道可能卡死（WebRTC 静默断连），输入 `stop`，重启 `unitree_control` 后再试。
- **Jetson 开机后连不上**（WiFi 已连但没有 IPv4、时钟停在 1970 年）：接网线直连 Go2，在 Jetson 上重启 `dhclient` 和 `systemd-timesyncd`，等拿到地址后拔掉网线，再拉起导航栈。拔线之前不要启导航栈，否则 WebRTC 可能走网线。
- 其他已知问题见 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)。
