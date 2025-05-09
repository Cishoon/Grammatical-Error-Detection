# 基于 Deepseek 的语病检测

输入为 latex 格式的文件。

该项目能自动将 latex 格式的文件按 "。" 句号分割为多个句子，并使用 deepseek 的 api 检测每个句子的语病。

自动忽略 latex 的图表、注释、代码块等。

输出以一个**清晰的方式**展示每个句子的是否存在语病，如果存在，则给出原因和修改结果。

## 功能特点

- 自动解析 LaTeX 文件，提取文本内容
- 忽略 LaTeX 的图表、公式、注释、代码块等不需要检测的部分
- 按句号智能分割句子
- 使用 Deepseek API 检测每个句子是否存在语病
- 支持多句并发处理，大幅提高处理速度
- 支持文件夹批量处理，自动为每个文件生成独立报告
- 优化的线程分配机制，每个线程独立处理句子
- 实时进度条显示，直观展示处理进度和剩余时间
- 生成清晰的 Markdown 格式检测报告，包含语病分析和修改建议
- 全面检测13种常见语病类型，包括主谓搭配不当、主宾搭配不当、成分残缺或赘余等

## 安装

1. 克隆本仓库：

```bash
git clone https://github.com/yourusername/Grammatical-Error-Detection.git
cd Grammatical-Error-Detection
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 准备工作

1. 获取 Deepseek API 密钥：
   - 在 [Deepseek 官网](https://deepseek.com) 注册账号并申请 API 密钥

2. 设置 API 密钥：
   - 方法一：通过环境变量设置 API 密钥
     ```bash
     export DEEPSEEK_API_KEY="your_api_key_here"
     ```
   - 方法二：在命令行参数中提供 API 密钥

### 运行程序

处理单个文件：

```bash
python main.py path/to/your/latex_file.tex --api-key your_api_key_here
```

处理整个文件夹中的所有 LaTeX 文件：

```bash
python main.py path/to/your/folder --api-key your_api_key_here
```

参数说明：
- `path/to/your/latex_file.tex`：要检测的 LaTeX 文件路径，或包含多个 LaTeX 文件的目录路径
- `--api-key`：Deepseek API 密钥（如果未通过环境变量设置）
- `--no-concurrent`：禁用并发处理模式（默认启用）
- `--concurrent-limit, -cl`：设置最大并发数量（默认为16）
- `--file-pattern, -fp`：处理目录时的文件匹配模式（默认为 `*.tex`）

### 输出报告

程序会为每个处理的文件生成一个独立的 Markdown 格式报告，文件名格式为 `文件名_时间戳.md`。
报告包含以下内容：

- 检测句子总数、语病句子数和语病比例统计
- 每个句子的详细检测结果
- 每个语病句子的问题描述和修改建议
- 使用 diff 格式显示原句与修改建议的对比

## 示例

单个文件处理：

```bash
python main.py example.tex
```

处理特定目录中的所有 LaTeX 文件：

```bash
python main.py ./papers --concurrent-limit 32
```

处理特定目录中的所有文本文件：

```bash
python main.py ./documents --file-pattern "*.txt"
```

## 检测的语病类型

本工具能够检测以下常见语病类型：

1. **句法结构类**
   - 成分残缺或赘余：句子缺少必要成分或多余成分
   - 语序不当：词语排列顺序不合理
   - 搭配不当：词语之间搭配不合理，语义不协调
   - 句式杂糅：不同句式混合使用，结构混乱

2. **语义逻辑类**
   - 表意不明：句子意思模糊，读者难以理解
   - 前后矛盾：句子内部逻辑矛盾
   - 不合逻辑：句子内容与常识或逻辑不符
   - 句子歧义：句子有多种理解方式，导致歧义

3. **用词错误类**
   - 词义误用：词语使用不当，表达错误
   - 词性误用：词语词性使用错误
   - 重复赘余：句子中出现多余的重复成分
   - 数量词误用：数量词使用不当

4. **标点符号类**
   - 标点误用：标点符号使用错误，影响句意

## 注意事项

- Deepseek API 可能有请求限制和收费标准，请合理使用
- 大型 LaTeX 文件处理可能需要较长时间，请耐心等待
- 语病检测结果仅供参考，最终修改请根据实际情况决定
- 使用并发模式可以大幅提高处理速度，但可能导致API请求过快被限流

## 许可证

MIT

