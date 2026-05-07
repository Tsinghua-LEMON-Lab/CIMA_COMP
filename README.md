# CIM_TC_open

## 项目子库简介

本仓库由多个子库协同组成，典型职责如下：

- `Model2IR`：模型解析与前端转换（例如从框架模型/ONNX 转为统一 IR）。
- `irtool`：IR 数据结构、读写、图操作与基础工具链。
- `mapper`：将 IR 映射到 CIMA 硬件拓扑（核/PE/XB 分配、通信关系等）。
- `backend`：后端代码生成，包括 `systemC`、`CIMA/uvm`、`CIMA/chip` 等配置生成。
- `optimizer`：映射与执行相关优化（例如 DMEM 优化、仿真环路优化）。
- `cimruntime`：运行时与仿真组件（用于执行/验证编译结果）。
- `test`：各模型 demo 与脚本入口（如 `test/Yolov5m` 完整编译流程）。

## 环境配置要求

### Python 与系统

- 推荐 `Python 3.9 ~ 3.11`（建议使用 conda 虚拟环境）。
- 操作系统：Windows / Linux 均可（本文命令以 Windows 路径风格为例）。

### Python 依赖盘点（按仓库实际导入）

说明：各子库 `setup.cfg` 当前显式声明几乎只有 `pyyaml`，但代码运行还依赖下列第三方库。  
建议按“基础工具链 + 模型流程”安装。

#### 基础工具链（建议必装）

- `pyyaml`：IR/YAML 读写（`irtool`、`mapper`、`backend` 等）
- `numpy`：数值处理（多个子库广泛使用）

#### Yolov5m 流程必需

- `torch`：权重与量化参数处理（`main_mapper.py`、`main_weight_extract.py`、`gen_act_lut.py`）
- `onnx`：ONNX 解析（`Model2IR/onnx2ir/*`）
- `onnxruntime`：形状推断/运行（`Model2IR/onnx2ir/shape_operation.py`）
- `scipy`：映射搜索距离计算（`mapper/search/Base.py`）
- `graphviz`（Python 包）：IR 图导出（`backend/CIMA/uvm/draw_graph.py`）

#### 可选依赖（部分功能/示例）

- `matplotlib`：部分可视化与分析（`mapper/helper.py`、部分测试脚本）
- `tensorflow`：`irtool/examples/net-1/run_tensorflow.py` 示例需要

推荐安装命令（Yolov5m demo）：

```bash
pip install pyyaml numpy torch onnx onnxruntime scipy graphviz matplotlib
pip install -e .
```

### 系统依赖（画图）

- 需安装 Graphviz 可执行程序，并保证 `dot` 在 `PATH` 中。
- 若仅跳过画图，可在 UVM 阶段使用 `--skip-graph`。

### Yolov5m Demo 运行前置文件

在 `test/Yolov5m/algo/` 下至少准备：

- `hard_params_dict_cpu.pth`
- `weight_int_dict_cpu.pth`

这两个文件通常由训练平台提供。

## 安装（开发模式）

```bash
pip install -e .
```

## Yolov5m Demo 使用文档

以下流程以 `test/Yolov5m` 为例。

### 1) 文件结构准备

在 `test/Yolov5m` 下准备目录与输入文件：

- `algo/`
  - `hard_params_dict_cpu.pth`（训练平台提供）
  - `weight_int_dict_cpu.pth`（训练平台提供）
- `model/`
  - `{model_name}.onnx`（例如 `yolov5m_wo_head.onnx`）
- `ir/`（可为空，编译时自动生成）
- `uvm/`（可为空，编译时自动生成）
- `chip/`（可为空，编译时自动生成）

### 2) 分步编译流程

进入目录：

```bash
cd test/Yolov5m
```

#### Step 1: 运行 mapper

```bash
python main_mapper.py --model-name yolov5m_wo_head
```

输出：`ir/` 下各阶段 IR 文件，最终文件为：

- `{model_name}_dmem_opt_mapped_ir_w_params.yaml`

#### Step 2: 运行 uvm/systemc

```bash
python main_uvm.py --model-name yolov5m_wo_head
```

关键输出（位于 `uvm/`）：

- `{model_name}_systemc.json`
- `{model_name}_graph.pdf`

其余文件为中间结果，可忽略。

#### Step 3: 运行 chip 编译

```bash
python main_chip.py --model-name yolov5m_wo_head
```

关键输出（位于 `chip/`）：

- `{model_name}_chip.json`

#### Step 4: 提取权重与激活 LUT

```bash
python main_weight_extract.py --model-name yolov5m_wo_head
python gen_act_lut.py --model-name yolov5m_wo_head
```

兼容命令（历史脚本名）：

```bash
python main_wight_extract.py --model-name yolov5m_wo_head
```

关键输出（位于 `chip/`）：

- `{model_name}_weight.json`
- `{model_name}_act_lut.json`

---

## 统一命令行接口（推荐）

新增统一入口：`test/Yolov5m/cli.py`

### 查看帮助

```bash
python cli.py --help
```

### 子命令

```bash
python cli.py mapper --model-name yolov5m_wo_head
python cli.py uvm --model-name yolov5m_wo_head
python cli.py chip --model-name yolov5m_wo_head
python cli.py weight --model-name yolov5m_wo_head
python cli.py act --model-name yolov5m_wo_head
```

### 一键全流程

```bash
python cli.py all --model-name yolov5m_wo_head
```

`all` 的执行顺序为：

1. mapper
2. uvm/systemc
3. chip
4. weight extract
5. act lut extract

