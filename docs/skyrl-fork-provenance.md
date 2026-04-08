# SkyRL Fork 来源说明

## 仓库关系

```
NovaSky-AI/SkyRL (官方, Apache 2.0)
  ├── yucc-leon/SkyRL (本项目 fork, 从官方 fork)
  │     ├── main 分支: 基于官方代码 + NPU 支持文件
  │     └── codescout-npu 分支: 基于 adityasoni9998 的 commit + NPU 适配
  │
  └── adityasoni9998/SkyRL (codescout 作者 fork)
        └── commit 81e5a97: codescout 项目依赖的 SkyRL 版本
```

## 关键说明

`yucc-leon/SkyRL` 是从官方 `NovaSky-AI/SkyRL` fork 出来的。

但 `codescout-npu` 分支的 base commit `81e5a97` 来自 `adityasoni9998/SkyRL`
（codescout 作者的 fork），不在官方仓库中。这个 commit 包含了 codescout 特有的
修改（如 reasoning parser fix），是 codescout 项目正常运行所必需的。

NPU 适配 patch（commit `6e3ffa5`）打在 `81e5a97` 之上。

## 分支说明

### `main` 分支 (commit 9945745)

- 基于官方 NovaSky-AI/SkyRL 的最新代码
- 包含 SkyRL 新版架构（dataclass config, vllm 0.16 支持等）
- 包含早期的 NPU 支持文件（npu_support/, skyrl_train shim）
- **不用于 codescout NPU 训练**（config 格式与 codescout 不兼容）

### `codescout-npu` 分支

- Base: commit `81e5a97` (来自 adityasoni9998/SkyRL)
  - 旧版 SkyRL，使用 YAML config (hydra)，支持 vllm 0.11
  - 是 codescout 项目 pyproject.toml 中 pin 的版本
- NPU 适配: commit `6e3ffa5`
  - 19 个文件，1312 行新增
  - 包含 npu_support/ 目录 + 源码 device/attention/通信 patch

## 获取方式

### 方案 A: 直接使用 fork

```bash
git clone https://github.com/yucc-leon/SkyRL.git
cd SkyRL
git checkout codescout-npu  # 6e3ffa5, 包含所有 NPU 适配
```

### 方案 B: 从 codescout 作者的 fork + patch

```bash
# 获取 codescout 兼容的 base commit
git clone https://github.com/adityasoni9998/SkyRL.git
cd SkyRL
git checkout 81e5a97c7430503c0c4e6508497cc5aa01a0c624

# 应用 NPU patch
git apply /path/to/codescout/patches/skyrl-ascend-npu.patch
```

### 方案 C: 从官方仓库 + 两步 patch（最透明）

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL

# 添加 codescout 作者的 fork 获取 base commit
git remote add codescout-author https://github.com/adityasoni9998/SkyRL.git
git fetch codescout-author 81e5a97c7430503c0c4e6508497cc5aa01a0c624 --depth=1
git checkout -b codescout-npu 81e5a97c7430503c0c4e6508497cc5aa01a0c624

# 应用 NPU patch
git apply /path/to/codescout/patches/skyrl-ascend-npu.patch
```

## 已验证环境

此 NPU 适配仅在以下环境验证通过：

| 组件 | 版本 |
|------|------|
| CANN | 8.3.RC1 |
| torch | 2.8.0 |
| torch_npu | 2.8.0.post2 |
| vllm | 0.11.0 |
| vllm-ascend | 0.11.0rc1 (源码构建) |
| Python | 3.12 |
| NPU | Ascend 910, 64GB HBM |

其他版本组合未经测试，不保证可用。

## License

所有仓库均为 Apache 2.0 协议。NPU 适配改动遵循上游协议。
