# 架构

## 1. 进程与端口

| 进程 | 位置 | 端口 / 接口 |
|---|---|---|
| 底层导航栈：SLAM、sensor_scan_generation、terrain_analysis(+ext)、localPlanner、pathFollower、unitree_control | ROS 2（开发机） | `/state_estimation`、`/way_point`、`/cmd_vel` |
| Livox 驱动、RealSense 推流服务 | Jetson | DDS（雷达/IMU）、TCP `:6000`（RGB-D） |
| YOLO-World 检测服务（`vlfm.vlm.yolo_world`） | 开发机 | HTTP `:12185` |
| Qwen2.5-VL 服务（vLLM，OpenAI 兼容接口） | 开发机（建议单独一块卡） | HTTP `:8101`，模型名 `qwen2.5-vl-3b` |
| `pose_udp_relay.py` / `waypoint_udp_relay.py` / `cmdvel_udp_relay.py` | 开发机（系统 python，主脚本自动拉起） | UDP `:8765` 位姿、`:8766` waypoint、`:8767` 偏航角速度 |
| `run_cgfm_pipeline.py` / `run_cgfm_lifelong.py` | 开发机（装有 vlfm / torch / open_clip 的环境） | — |

主脚本本身不依赖 ROS Python 环境：和 ROS 的交互全部通过三个 UDP relay 和 `ros2` 命令行（`/liedown`、启动 `pathFollower`）完成，所以可以跑在 conda 环境里，ROS 用系统 Python。

## 2. 主循环（约 1 Hz）

```
开机 → [可选] 预检 SLAM → 360° 扫描（8 步 × 45°，闭环偏航，pathFollower 尚未启动）
     → 启动 pathFollower（autonomyMode=true）→ 循环：
         读位姿 + RGB-D → YOLO-World 检测 → 更新障碍图 / 地面图 / 场景图
         ├─ 未锁定（探索）：语义场 → frontier 打分 → /way_point
         │                  VLM 查询（后台线程）→ 选中节点 → 复核 → 锁定
         └─ 已锁定（走向目标）：只做感知和记录，用更近的观测精修目标，到 0.5 m 内算找到
```

每个 tick 把相机帧、深度、各张地图、场景图快照写进运行目录，供复盘视频使用。

## 3. 场景图（`cgfm/scene_graph.py`）

- **检测**：目标词 + 10 类常驻词汇（chair、table、desk、monitor、cabinet、door、trash can、couch、whiteboard、plant）。目标在 4.0 m 内、其他物体在 3.0 m 内才入图；距离取框中心深度和“框下沿与地面交点”两者较近的一个。
- **点云**：框内缩 15%，取 0.3–3 m 深度，去掉地面，DBSCAN（eps 0.10 m）取最近的较大簇，5 cm 体素下采样。对网状垃圾桶这类透明物体，只有目标类允许在深度缺失时用“地面接触点”生成合成节点。
- **关联与合并**：点云重叠 + CLIP 余弦之和 > 1.2 且质心距离 < 1.5 m 才合并；周期性合并同类碎片、重复节点、稀疏节点，并按多数投票处理“同一物体两个标签”（目标类受保护，不会被并到别的类）。
- **确认**：看到 2 次以上才进入语义场和 VLM 输入。
- **边**：同一帧里同时出现、质心距离 < 2 m 的节点连边，并记录能看到它们的证据帧（给 VLM 的图）。
- **导出**：`scene_graph.json/.txt`（可直接放进提示词）、`scene_graph_features.npz`。

## 4. 语义场与 frontier（`semantic_map.py`、`frontier_scorer.py`）

对每个确认节点算 CLIP 相似度（减去基线 0.16），在“已观察到的自由空间”（障碍图的已探索区域 ∪ `floor_map.py` 观察到的地面）上做 Dijkstra 扩散得到语义场。frontier 得分 = 语义邻域均值 / (1 + λ·归一化测地距离)；所有得分接近 0 时退化为选最近的；最小距离 0.8 m、切换需要好 30% 以上（迟滞）；15 s 内移动不到 0.5 m 的 frontier 会被拉黑（半径 1 m 内不再选）。

## 5. VLM 守门（`vlm_selector.py`）

1. 提示词给出节点列表（id: 类别）、同帧关系、每个节点能被看到的证据图（框和 `#id` 画在图上），以及当前相机视图，要求只回答 `{"object_id": <id 或 null>}`。
2. 查询在后台线程里做；同一个（目标、节点列表、边集合）只问一次，图变了才再问。
3. **裁剪图复核**：被选中的节点再做一次选择题（“图中间是什么？目标 / chair / couch / desk / table / cabinet / monitor / computer / box / other”）；裁剪图短边 < 48 px 时无法判断，改为只看检测器标签是否等于目标。
4. **投票规则**（终身脚本）：同一节点被 VLM 选中 ≥ 3 次、检测器标签等于目标、且复核答案从没出现明显冲突的家具类别（`box` / `other` 不算冲突），即使复核没通过也接受。原因：小而糊的网状垃圾桶会被复核反复答成 `box` / `table`，而 VLM 的选择是对的。
5. 锁定后目标坐标 = 该节点“朝向机器人一侧”的近边点，由脚本从场景图取出，VLM 从不输出坐标。VLM 不可用或连续失败时退化为检测规则（5 帧内 3 次命中，或场景图里该类节点被看到 ≥ 3 次）。

## 6. 终身导航（`run_cgfm_lifelong.py`）

- **目标队列**：`--targets "a,b,c"`，运行中可 `target <名>` 立即切换、`queue <名>` 追加。目标名按同义词归一（`potted plant → plant`、`sofa → couch`）。队列里所有目标从一开始就被一直检测，这样后面才要找的东西在被设为目标前就已进入场景图。
- **切换**：找到目标（距目标 ≤ 0.5 m）→ 清掉上一个目标的搜索状态（锁定、命中窗口、frontier 迟滞、停滞计时），地图和场景图保留 → 日志里打印记忆检查（图里有多少个该类节点）。
- **回忆规则**：这次运行里**已经找到过**的类别再次被请求时，不经过 VLM 和复核，直接取该节点的当前近边坐标（节点被合并就跟踪别名，节点消失就用当时找到的坐标）发给底层栈走回去。`--exclude-found` 可关闭，让重复的类别表示“找另一个”。见过但没找到过的物体仍走 VLM + 复核。
- **队列走完**：停在最后的目标处只记录、不再发目标，等待 `target` / `queue` / `stop`；不会自动趴下。
- **`--dry-run`**：机器人不动、也不会真的锁定，所以把“将会锁定 / 将会回忆”当作模拟找到来推进队列，只用于验证切换和记忆检查的日志。

## 7. 安全检查

扫描前：SLAM 三个进程都在、启动以来没有崩溃（否则 `NOT STARTING`）；扫描后：位姿还在更新、没有冻结、扫描没有因漂移（原地转却平移 > 1 m）中止（否则 `NOT ARMING`）；运行中：1 s 内位姿跳变超过 `max(0.5 m, 0.5 m/s·dt)`、2 s 收不到位姿、位姿 5 s 完全不变、`slam.log` 出现 `process has died` 都会 `STOPPING`。Ctrl-C / `stop` 在信号处理函数里立即杀掉 `pathFollower`（即使主线程正卡在 VLM 调用里），然后尽力调用 `/liedown`。**还没有“机器人不响应运动指令”的检测。**

## 8. 扩展点

- 换检测器：实现和 `YOLOWorldClient.predict(image, caption)` 相同的接口，改 `detect_scene`。
- 换 VLM：`VLMTargetSelector` 只依赖 OpenAI 兼容的 `/chat/completions`，改 `--vlm-url` 和模型名即可。
- 换机器人：把三个 UDP relay 对应的 ROS 话题换成你的平台，主脚本里与 Go2 相关的只有相机外参（`CAMERA_HEIGHT_M` 等）、`/liedown` 服务和 `pathFollower` 的启动参数。
