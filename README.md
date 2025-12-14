# ML-Class-Assignment25
2025机器学习基础课程大作业

## 1 使用说明 (快速开始)

### 1.1 安装 uv

电脑安装 [uv](https://docs.astral.sh/uv/) （一个类似于 conda 的 python 综合管理工具）

> uv 非常方便，使用 uv run 运行代码会自动帮你配置好项目环境、安装相关依赖并调用 python 执行程序（如果你没有python也可以自动安装python）。

示例：Windows 下快速安装 uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.2 运行代码

在**项目根目录**运行以下命令：

1. 实验一：手写数字识别

    ```powershell
    uv run 1_digit/train_pytorch.py
    ```

2. 实验二：待续

运行代码后会输出模型参数文件（例如`pytorch_model.pth`）到对应实验文件夹的 `models` 目录下，并输出性能指标等数据到控制台。

## 2 报告

简单为模型和训练效果撰写了一个报告，在对应实验文件夹内的 `reports` 目录下。