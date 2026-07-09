<div align="center">

# ProCAT

### 面向实际在线故障诊断的原型校准测试时自适应

[English](README.md) | **简体中文**

[![Paper](https://img.shields.io/badge/%E8%AE%BA%E6%96%87-%E5%AE%A1%E7%A8%BF%E4%B8%AD-b31b1b.svg)](#-引用)
[![Method](https://img.shields.io/badge/%E6%96%B9%E6%B3%95-%E5%8D%B3%E5%B0%86%E5%85%AC%E5%BC%80-orange.svg)](#-方法代码即将公开)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](#-快速开始)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C.svg?logo=pytorch&logoColor=white)](#-快速开始)
[![License](https://img.shields.io/badge/License-%E5%BE%85%E5%AE%9A-lightgrey.svg)](#-许可证)

面向 **1D 轴承故障诊断** 的可复现 **测试时自适应（TTA）** 基准，
所提出方法 **ProCAT** 的源码将在此仓库发布。

<sub>🔧 故障诊断 · 🌊 域偏移 · 🎯 特征原型 · 🧮 类条件 BN · ⏱️ 在线学习</sub>

</div>

---

> [!TIP]
> **一句话总结** —— 仅依赖分类器的伪标签在流式域偏移与类别不均衡下会漂移。
> ProCAT 用在线 **特征原型** 对其进行校准，使归一化统计量与自训练信号都更加可信。🎯

---

## 🚧 方法代码即将公开

> 所提出的方法 **ProCAT**（原型校准测试时自适应）正在整理与文档化，**即将在本仓库发布**。
>
> 本仓库目前提供论文中使用的 **可复现基准** 与 **10 个基线方法**。
> 欢迎 **⭐ Star / 👀 Watch** 本仓库，以便在 ProCAT 源码与预训练模型上线时第一时间收到通知。

---

## 📝 摘要

在线故障诊断模型常常受到目标域偏移的影响，这些偏移由转速变化、噪声干扰以及局部类别不均衡引起。
测试时自适应（TTA）通过在线利用无标签目标数据更新模型，提供了一种有前景的解决方案。然而，
现有 TTA 方法直接从分类器预测中获取伪监督，在不断演化的目标分布与不均衡的测试流下，这些伪标签
可能是有偏的，从而误导归一化统计量并放大错误的自适应。本文提出面向流式域偏移下在线故障诊断的
**原型校准测试时自适应（Prototype-CAlibrated Test-time adaptation, ProCAT）**，它在教师–学生
自适应框架中引入了特征原型校准机制。具体而言，源域原型被构造为初始语义锚点，同时一个类别均衡的
特征记忆库利用高置信度样本逐步细化目标域原型。随后，原型诱导的后验概率与教师预测融合为一个
校准的伪分布，该伪分布同时驱动类条件批归一化桶，并为学生自训练自适应提供软监督。在部署阶段，
仅对归一化的仿射参数进行梯度更新，而教师则通过指数移动平均更新，为后续目标批次提供时间上稳定
的预测。在机械故障数据集上的大量实验表明，所提方法在具有域偏移、动态噪声与标签不均衡的实际
在线诊断场景中取得了最先进（state-of-the-art）的性能。

---

## ✨ 亮点

- 🧩 面向在线故障诊断的 **原型校准** 测试时自适应方法。
- 🧠 由源域初始化的 **类别原型** 通过类别均衡的 **特征记忆库** 在线持续更新。
- 🔗 教师概率与原型概率 **融合为一个校准的伪分布**。
- ⚙️ 该融合信号同时驱动 **类条件批归一化（class-conditional BN）** 与 **软标签自训练**。
- 🚀 在两种主干网络（TFN、ResNet18）上，跨域偏移与类别不均衡场景下均取得 **显著提升**。

---

## 🧠 方法

ProCAT 在教师–学生自适应框架的基础上引入了 **特征原型推理通道**。
在轻量的 *准备阶段*，类别原型由源域特征初始化。
测试阶段，对每个到来的批次，教师概率 `p^T` 与原型诱导概率 `p^I` 被融合为一个
**校准的伪分布** `p*`，其作用为：(1) 更新类别均衡的特征记忆库，(2) 提供校准后的伪标签，
(3) 驱动类条件 BN 统计量——在几乎不增加开销的前提下，保持归一化与自训练的一致性。

<div align="center">
<img src="assets/framework.png" width="95%" alt="ProCAT 框架"/>
<br/>
<em>ProCAT 概览：源域原型准备阶段（左）与原型校准测试时自适应（右）。</em>
</div>

<br/>

<div align="center">
<img src="assets/motivation.png" width="90%" alt="伪标签校准动机"/>
<br/>
<em>动机：有偏的分类器预测 (a) 经原型诱导分布 (b) 校正为校准伪分布 (c)，
在特征空间中将伪标签拉向真实类别 (d)。</em>
</div>

---

## 📊 实验结果

**可靠性 / 置信度校准。** ProCAT 给出校准性最好的预测（最低的 ECE / MCE），
而部分 TTA 基线甚至比冻结的源模型更加过度自信。

<div align="center">
<img src="assets/reliability.png" width="95%" alt="可靠性图"/>
</div>

**特征对齐。** 目标数据流的 t-SNE 可视化显示，ProCAT 逐步将跨转速的特征分布
对齐为区分度良好的类别簇。

<div align="center">
<img src="assets/feature_tsne.png" width="95%" alt="t-SNE 特征可视化"/>
</div>

**Bogie 在线故障诊断。** 六个 Bogie 任务（$T_1^{\mathrm{BG}}$–$T_6^{\mathrm{BG}}$，TFN）
上沿在线数据流的累计准确率曲线。每个任务在两种轴转速上预训练，并在未见转速与不同标签
不均衡条件下测试。ProCAT（粗黑线）在整个数据流上持续优于强 TTA 基线，在未见转速段进入后
优势尤为明显。

<div align="center">
<img src="assets/bogie_online_curves.png" width="95%" alt="Bogie 在线累计准确率曲线"/>
</div>

<!-- 论文发表前暂时隐藏；需要展示时取消注释即可。
**伪标签质量。** 相比仅教师或仅原型的伪标签，校准后的伪标签更忠实地跟踪
真实的逐批次标签分布，有效缓解了数据流后段的类别坍塌。

<div align="center">
<img src="assets/pseudo_label_quality.png" width="95%" alt="伪标签分布对比"/>
</div>
-->

---

## 🧪 基准

在三个 1D 轴承故障诊断数据集上的可复现 TTA 基准：

| 数据集 | 任务 | 偏移场景 |
|---|---|---|
| **SQ** | T1–T8 | 跨转速域偏移、动态噪声、Dirichlet 标签偏移 |
| **Bogie** | T1–T6 | 跨 RPM 域偏移，Dirichlet γ 取值不同 |
| **OUB** | T1–T8 | 跨域渥太华大学轴承数据（2048 点，`light_trial1` 子集） |

实验采用 **在线批次（online-batch）** 协议（批大小 64，随机种子 0），对比 **10 个基线方法**：

`source_only`、`bn_adapt`、`rotta`、`cotta`、`petta`、`tea`、`tact`、`eata`、`tribe_official`、`tribe`

预训练的 **TFN** 与 **ResNet18** 源模型权重已包含在 `checkpoints/` 下。
原始数据文件 **不随仓库分发**。

---

## ⚡ 快速开始

```bash
pip install -r requirements.txt
cp configs/paths.example.yaml configs/paths.yaml   # 编辑为你的数据缓存路径
```

需要 Python 3.10+ 与 PyTorch（推荐 CUDA）。

将 `TTA_DATA_ROOT` 设为包含 `SQdata/`、`OUBdata/` 等的父目录，或编辑 `configs/paths.yaml`。

## ▶️ 运行实验

```bash
# SQ —— 全部 8 个任务，TFN
python -m tta.sq.run_sq_full eval --tasks all --model tfn --device cuda

# Bogie —— 全部 6 个任务
python -m tta.bogie.run_bogie_full eval --tasks all --model tfn --device cuda

# OUB light_trial1 —— 全部 8 个任务
python -m tta.oub.run_oub_2048_full eval --subset light_trial1 --tasks all --model tfn --device cuda
```

## 📖 引用

> [!NOTE]
> 📌 论文 **尚未正式发表**，目前处于审稿阶段。
> 论文正式发表后，将在此处补充引用信息（BibTeX）。敬请关注！🔔

## 📄 许可证

见 LICENSE（待添加）。

---

<div align="center">
<sub>为故障诊断与测试时自适应社区用 ❤️ 打造 · ⭐ Star 关注 ProCAT 发布进展</sub>
</div>
