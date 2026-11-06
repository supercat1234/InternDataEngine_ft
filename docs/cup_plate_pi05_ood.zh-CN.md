# 杯盘任务：独立 OOD 评测

OOD 相对于当前数据生成任务的 ID 配置定义。使用相同 checkpoint、动作接口和成功判据；专家数据生成与原 ID 评测入口不变。默认开启随机化，光照、相机和未被指定为 OOD 的因素仍沿用 ID 范围。

## 配置与文件

| `OOD_PROFILE` | 改变的因素 | 默认内容 |
| --- | --- | --- |
| `positions` | 杯盘位置 | 杯 X：−0.24～−0.20，Y：−0.20～−0.175；盘 X：−0.08～−0.01，Y：−0.21～−0.16，单位米 |
| `objects` | 杯盘颜色 | 杯：蓝/绿/品红；盘：蓝/橙/绿，各自等概率 |
| `table` | 桌面纹理 | 深胡桃木纹、灰色织纹、水磨石，各占 1/3 |
| `floor` | 地板颜色 | 蓝、绿、赤陶色，各占 1/3；保留地砖尺寸与缝隙 |
| `combined`（默认） | 上述全部 | 同时启用 |

位置为任务坐标系的米制坐标；杯盘均在桌面内，初始轮廓不重叠。新位置的 Y 区间完全落在当前 ID 区间之外，并偏向机器人一侧。它们不经过专家规划器筛选，不能把这里的策略失败自动等同于机械臂不可达。

- `local_configs/cup_plate_pi05_ood/parameters.yaml`：位置范围、杯盘 RGB 色调、纹理选项及权重。RGB 是对原有颜色纹理的乘数，保留原纹理细节。
- 同目录 `positions.yaml`、`objects.yaml`、`table.yaml`、`floor.yaml`、`combined.yaml`：各自的 checkpoint、输出根目录、随机化开关、步数与成功阈值。
- 同目录 `textures/`：独立的新纹理；可用 `scripts/assets/build_cup_plate_ood_textures.py` 重建。添加素材后修改参数文件中的 `file`，相对路径以仓库根目录为基准，也支持绝对路径。
- `scripts/eval/ood_scene.py`：只在启用 OOD 的评测中替换配置并添加颜色覆盖，不写回原始 USD。

此处没有对完整训练集做逐帧颜色/纹理审计，因此“未见”表示与现有 ID 配置不同，不保证真实数据里从未出现相近颜色。

## 命令

在仓库根目录 `/home/agilex/qhx/InternDataEngine` 执行。

先看场景，不调用模型，不产生动作；保存三个相机的初始 PNG：

```bash
OOD_PROFILE=combined bash scripts/run_cup_plate_pi05_ood_eval.sh \
  --headless --observe-only --seed 42
```

去掉 `--headless` 可打开窗口；`--observe-only` 仍会在保存初始画面后结束。

闭环测试需先启动现有模型服务（已经运行时无需再次启动）：

```bash
bash scripts/run_cup_plate_pi05_server.sh
```

另一个终端运行单条或批量测试。无需添加 `--randomize`：

```bash
# 单条位置 OOD
OOD_PROFILE=positions bash scripts/run_cup_plate_pi05_ood_eval.sh --seed 99

# 单实例连续 100 次：固定主 seed，继续场景随机数流
OOD_PROFILE=combined INSTANCES=1 TOTAL=100 BASE_SEED=99 \
bash scripts/run_cup_plate_pi05_ood_batch.sh

# 三路各 100 次，共 300 次，主 seed 分别为 42、43、44
OOD_PROFILE=combined INSTANCES=3 TOTAL=100 BASE_SEED=42 \
bash scripts/run_cup_plate_pi05_ood_batch.sh

# 只检查任务数与种子分配，不启动仿真
OOD_PROFILE=table INSTANCES=3 TOTAL=2 BASE_SEED=42 \
bash scripts/run_cup_plate_pi05_ood_batch.sh --dry-run
```

`OOD_PROFILE` 可选上表五种值。并发测试与 ID 评测共享 GPU/模型服务资源，建议安排在 ID 批次结束后；这不影响配置和结果目录隔离。`TOTAL` 是每路次数。`EPISODE_TIMEOUT` 是每条进程的墙钟超时，默认 600 秒，包含加载与保存时间。

失败默认仿真 45 秒，成功后固定追加 5 秒。`--max-steps N` 显式覆盖步数上限（30 FPS），`--execute-steps N` 调整每次执行动作数（仅单条入口），`--url URL` 指定模型服务，`--save-debug-arrays` 可选保存调试数组（仅单条入口）。

## 输出、复现和回放

默认保存到 `/home/agilex/qhx/evaluation/sim/ood/<profile>/YYYYMMDD_001/`，各 profile 独立编号、`summary.csv` 和 `batches/` 汇总，不计入 ID 成功率。修改对应 profile YAML 的 `output_root` 即可换地址。单条也可用 `--output /新的空目录` 指定本条目录；根汇总仍在 `output_root`。

每条沿用现有结果文件、三路 MP4、轨迹 JSON、运行日志和归档 YAML，额外保存 `ood_state.json`：实际初始杯盘位姿、选中的颜色、桌面纹理与地板颜色。纯预览只保存初始 PNG，没有 MP4。

```bash
# 换成真实存在的 OOD 结果目录；重新推理，使用归档场景参数
bash scripts/run_cup_plate_pi05_eval.sh \
  --reproduce /home/agilex/qhx/evaluation/sim/ood/combined/20261104_001 --headless

# 使用已保存动作回放，不调用模型
bash scripts/run_cup_plate_pi05_eval.sh \
  --reproduce /home/agilex/qhx/evaluation/sim/ood/combined/20261104_001 \
  --replay-saved-actions --headless
```

复现继承归档的 OOD 配置、颜色选项和结果根目录，参数文件后续修改不会改变旧场景。重新推理仍可能因模型采样而改变结果。中断和异常尝试沿用现有排除规则，不计入成功率。
