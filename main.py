#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import json
import os
import glob
from typing import List, Dict, Any, Tuple
import requests
import time
from pathlib import Path
import asyncio
import aiohttp
import concurrent.futures
from tqdm import tqdm


class LatexParser:
    """解析Latex文件，提取文本内容并按句号分割为句子"""
    
    def __init__(self):
        # 用于匹配需要忽略的latex内容
        self.ignore_patterns = [
            r'\\begin\{figure\}.*?\\end\{figure\}',  # 图片环境
            r'\\begin\{table\}.*?\\end\{table\}',    # 表格环境
            r'\\begin\{code\}.*?\\end\{code\}',      # 代码环境
            r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}',  # 代码环境
            r'\\begin\{algorithm\}.*?\\end\{algorithm\}',    # 算法环境
            r'%.*?(?:\n|$)',                         # latex注释
            r'\\cite\{.*?\}',                        # 引用
            r'\\label\{.*?\}',                       # 标签
            r'\\ref\{.*?\}',                         # 引用
        ]
        self.ignore_regex = re.compile('|'.join(self.ignore_patterns), re.DOTALL)
        
    def parse_file(self, file_path: str) -> str:
        """读取latex文件并返回文本内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    
    def clean_latex(self, content: str) -> str:
        """清理latex内容，去除图表、注释等"""
        # 替换需要忽略的内容为空格
        cleaned = self.ignore_regex.sub(' ', content)
        # 去除命令
        cleaned = re.sub(r'\\[a-zA-Z]+(\{.*?\}|\[.*?\])*', ' ', cleaned)
        return cleaned
    
    def split_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 按句号分割，但忽略用于表示小数点的句号
        sentences = []
        # 使用正则表达式分割句子，考虑中文句号和英文句号
        raw_sentences = re.split(r'([。！？\.!?])', text)
        
        i = 0
        current_sentence = ""
        while i < len(raw_sentences):
            if i + 1 < len(raw_sentences) and re.match(r'[。！？\.!?]', raw_sentences[i+1]):
                # 将句子和句号合并
                current_sentence += raw_sentences[i] + raw_sentences[i+1]
                # 检查是否为完整句子
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
                i += 2
            else:
                current_sentence += raw_sentences[i]
                i += 1
        
        # 添加最后一个可能不以句号结尾的句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        # 过滤掉空句子和只包含空白字符的句子
        return [s for s in sentences if s.strip()]


class DeepseekAPI:
    """使用Deepseek API检测语病"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # 语病检测提示模板
        self.error_detection_prompt = """
请严格分析以下句子是否存在语病，特别注意检查以下几个方面：
中文论文中的常见语病类型

一、句法结构类
	1.	成分残缺或赘余：句子缺少必要成分或多余成分，导致结构不完整或冗杂。
	- 示例："通过调查，发现了问题。"（缺主语）
	2.	语序不当：词语排列顺序不合理，影响句意表达。
	- 示例："我车已经买了。"应为"我已经买了车"。
	3.	搭配不当：词语之间搭配不合理，语义不协调。
	- 示例："我们要加强和提高管理水平。"（"加强"和"提高"不能同时搭配"管理水平"）
	4.	句式杂糅：不同句式混合使用，结构混乱。
	- 示例："由于他工作努力，所以深受大家欢迎。"（"由于"和"所以"不能同时使用）

二、语义逻辑类
	5.	表意不明：句子意思模糊，读者难以理解。
	- 示例："他告诉她他要去的地方。"（"他"和"她"指代不清）
	6.	前后矛盾：句子内部逻辑矛盾。
	- 示例："他虽然年纪小，但经验丰富。"（"年纪小"与"经验丰富"可能矛盾）
	7.	不合逻辑：句子内容与常识或逻辑不符。
	- 示例："盲人看到有一个小女孩走过这条路。"（"盲人"无法"看到"）
	8.	句子歧义：句子有多种理解方式，导致歧义。
	- 示例："我看见他拿着望远镜。"（是"我"用望远镜看他，还是"他"拿着望远镜？）

三、用词错误类
	9.	词义误用：词语使用不当，表达错误。
	- 示例："他对工作很马虎。"应为"他对工作很认真。"
	10.	词性误用：词语词性使用错误。
	- 示例："他是一个很有创造的人。"应为"他是一个很有创造力的人。"
	11.	重复赘余：句子中出现多余的重复成分。
	- 示例："我亲自亲手做的。"（"亲自"和"亲手"重复）
	12.	数量词误用：数量词使用不当。
	- 示例："一只人"应为"一个人"。

四、标点符号类
	13.	标点误用：标点符号使用错误，影响句意。
	- 示例："他说，'我明天去'。"应为"他说："我明天去。""

即使语病很细微，也请指出。如果句子中有多个语病，请全部指出。

句子：{sentence}

请仔细分析后使用以下JSON格式回答，其中has_error必须为true或false：
{{
    "has_error": true/false,
    "reason": "详细解释为什么存在语病以及具体是哪种类型的语病",
    "corrected": "给出修改后的正确句子"
}}

如果确定没有语病，则reason字段填写"句子表达流畅，没有明显语病"。
"""
    
    def detect_errors(self, sentence: str) -> Dict[str, Any]:
        """检测句子中的语病"""
        prompt = self.error_detection_prompt.format(sentence=sentence)
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 提取JSON格式的结果
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"has_error": False, "reason": "解析结果失败", "corrected": sentence}
            return {"has_error": False, "reason": "无法解析API返回结果", "corrected": sentence}
        
        except Exception as e:
            print(f"API请求出错: {e}")
            return {"has_error": False, "reason": f"API请求失败: {str(e)}", "corrected": sentence}
    
    async def detect_errors_async(self, sentence: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """异步检测句子中的语病"""
        prompt = self.error_detection_prompt.format(sentence=sentence)
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 提取JSON格式的结果
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        return {"has_error": False, "reason": "解析结果失败", "corrected": sentence}
                return {"has_error": False, "reason": "无法解析API返回结果", "corrected": sentence}
        
        except Exception as e:
            print(f"API请求出错: {e}")
            return {"has_error": False, "reason": f"API请求失败: {str(e)}", "corrected": sentence}


class ErrorDetector:
    """语病检测主类"""
    
    def __init__(self, api_key: str):
        self.parser = LatexParser()
        self.api = DeepseekAPI(api_key)
        self.max_concurrent = 16  # 最大并发请求数
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """处理latex文件并返回检测结果 (串行处理)"""
        content = self.parser.parse_file(file_path)
        cleaned_content = self.parser.clean_latex(content)
        sentences = self.parser.split_sentences(cleaned_content)
        
        results = []
        # 使用tqdm创建进度条
        with tqdm(total=len(sentences), desc=f"处理文件 {os.path.basename(file_path)}") as pbar:
            for idx, sentence in enumerate(sentences):
                result = self.api.detect_errors(sentence)
                results.append({
                    "id": idx + 1,
                    "sentence": sentence,
                    "has_error": result.get("has_error", False),
                    "reason": result.get("reason", ""),
                    "corrected": result.get("corrected", sentence)
                })
                # 更新进度条
                pbar.update(1)
                # 避免API请求过快
                time.sleep(1)
        
        return results
    
    async def process_sentence_async(self, idx: int, sentence: str, session: aiohttp.ClientSession, pbar: tqdm) -> Dict[str, Any]:
        """异步处理单个句子"""
        result = await self.api.detect_errors_async(sentence, session)
        # 更新进度条
        pbar.update(1)
        return {
            "id": idx + 1,
            "sentence": sentence,
            "has_error": result.get("has_error", False),
            "reason": result.get("reason", ""),
            "corrected": result.get("corrected", sentence)
        }
    
    async def process_file_async(self, file_path: str) -> List[Dict[str, Any]]:
        """处理latex文件并返回检测结果 (优化的并行处理)"""
        content = self.parser.parse_file(file_path)
        cleaned_content = self.parser.clean_latex(content)
        sentences = self.parser.split_sentences(cleaned_content)
        
        total_sentences = len(sentences)
        print(f"共发现 {total_sentences} 个句子，将并发检测（最大并发数: {self.max_concurrent}）")
        
        # 创建进度条
        pbar = tqdm(total=total_sentences, desc=f"处理文件 {os.path.basename(file_path)}")
        
        # 创建一个客户端会话
        async with aiohttp.ClientSession() as session:
            # 创建任务列表，每个线程处理索引为 i, i+max_concurrent, i+2*max_concurrent, ... 的句子
            tasks = []
            for i in range(self.max_concurrent):
                # 获取当前线程要处理的句子索引
                indices = list(range(i, total_sentences, self.max_concurrent))
                # 创建处理这些句子的协程
                task = self.process_thread_sentences(indices, sentences, session, pbar)
                tasks.append(task)
            
            # 等待所有任务完成并收集结果
            thread_results = await asyncio.gather(*tasks)
            
            # 合并结果
            results = []
            for thread_result in thread_results:
                results.extend(thread_result)
            
            # 关闭进度条
            pbar.close()
        
        # 按ID排序结果
        results.sort(key=lambda x: x["id"])
        return results
    
    async def process_thread_sentences(self, indices: List[int], sentences: List[str], session: aiohttp.ClientSession, pbar: tqdm) -> List[Dict[str, Any]]:
        """处理一个线程负责的所有句子"""
        results = []
        for idx in indices:
            # 处理句子
            result = await self.api.detect_errors_async(sentences[idx], session)
            results.append({
                "id": idx + 1,
                "sentence": sentences[idx],
                "has_error": result.get("has_error", False),
                "reason": result.get("reason", ""),
                "corrected": result.get("corrected", sentences[idx])
            })
            # 更新进度条
            pbar.update(1)
            # 为避免API请求过快，每个线程处理完一个句子后等待一小段时间
            await asyncio.sleep(0.5)
        return results
    
    async def process_directory_async(self, dir_path: str, file_pattern: str = "*.tex") -> Dict[str, List[Dict[str, Any]]]:
        """处理目录中符合模式的所有文件"""
        # 获取所有符合模式的文件路径
        file_paths = glob.glob(os.path.join(dir_path, file_pattern))
        
        if not file_paths:
            print(f"在目录 {dir_path} 中未找到符合 {file_pattern} 的文件")
            return {}
        
        print(f"在目录 {dir_path} 中找到 {len(file_paths)} 个符合 {file_pattern} 的文件")
        
        # 处理所有文件
        results = {}
        for file_path in file_paths:
            print(f"\n开始处理文件: {file_path}")
            file_results = await self.process_file_async(file_path)
            results[file_path] = file_results
            
            # 生成报告
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{base_name}_{timestamp}.md"
            print(f"生成报告: {output_filename}")
            self.generate_report(file_results, output_filename)
        
        return results
    
    def process_directory(self, dir_path: str, file_pattern: str = "*.tex") -> Dict[str, List[Dict[str, Any]]]:
        """处理目录中符合模式的所有文件（串行版本）"""
        # 获取所有符合模式的文件路径
        file_paths = glob.glob(os.path.join(dir_path, file_pattern))
        
        if not file_paths:
            print(f"在目录 {dir_path} 中未找到符合 {file_pattern} 的文件")
            return {}
        
        print(f"在目录 {dir_path} 中找到 {len(file_paths)} 个符合 {file_pattern} 的文件")
        
        # 处理所有文件
        results = {}
        for file_path in file_paths:
            print(f"\n开始处理文件: {file_path}")
            file_results = self.process_file(file_path)
            results[file_path] = file_results
            
            # 生成报告
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{base_name}_{timestamp}.md"
            print(f"生成报告: {output_filename}")
            self.generate_report(file_results, output_filename)
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]], output_path: str):
        """生成检测报告"""
        # 统计结果
        error_count = sum(1 for r in results if r.get('has_error', False))
        total_count = len(results)
        error_rate = error_count / total_count * 100 if total_count > 0 else 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入报告标题和摘要
            f.write("# 语病检测报告\n\n")
            f.write(f"## 摘要\n\n")
            f.write(f"- **检测句子总数**: {total_count}\n")
            f.write(f"- **存在语病句子数**: {error_count}\n")
            f.write(f"- **语病比例**: {error_rate:.2f}%\n\n")
            
            # 写入语病检测结果
            f.write("## 详细检测结果\n\n")
            
            for result in results:
                sentence_id = result.get('id', 0)
                sentence = result.get('sentence', '')
                has_error = result.get('has_error', False)
                reason = result.get('reason', '')
                corrected = result.get('corrected', '')
                
                # 短句展示完整，长句截断
                display_sentence = sentence
                if len(display_sentence) > 100:
                    display_sentence = display_sentence[:100] + "..."
                
                f.write(f"### 句子 {sentence_id}\n\n")
                f.write(f"**原句**: {display_sentence}\n\n")
                
                if has_error:
                    f.write(f"**存在语病**: ✓\n\n")
                    f.write(f"**问题描述**: {reason}\n\n")
                    
                    # 检查原始句子和修改后的句子是否有差异
                    if sentence != corrected:
                        f.write("**修改建议**:\n\n")
                        f.write(f"```diff\n- {sentence}\n+ {corrected}\n```\n\n")
                    else:
                        f.write("**修改建议**: _未提供具体修改方案_\n\n")
                else:
                    f.write(f"**存在语病**: ✗\n\n")
                
                f.write("---\n\n")
                
            # 写入报告生成时间
            f.write(f"\n\n*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")


def is_directory(path: str) -> bool:
    """检查路径是否为目录"""
    return os.path.isdir(path)


def main():
    parser = argparse.ArgumentParser(description="基于Deepseek的latex文件语病检测工具")
    parser.add_argument("input_path", help="输入的latex文件或目录路径")
    parser.add_argument("--api-key", help="Deepseek API密钥")
    parser.add_argument("--no-concurrent", action="store_true", help="禁用并发模式（默认启用）")
    parser.add_argument("--concurrent-limit", "-cl", type=int, default=16, help="并发请求数量限制（默认为16）")
    parser.add_argument("--file-pattern", "-fp", default="*.tex", help="处理目录时的文件匹配模式（默认为*.tex）")
    
    args = parser.parse_args()
    
    # 检查API密钥
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 未提供Deepseek API密钥。请通过--api-key参数或DEEPSEEK_API_KEY环境变量提供")
        return
    
    # 检查输入路径
    if not os.path.exists(args.input_path):
        print(f"错误: 输入路径 '{args.input_path}' 不存在")
        return
    
    detector = ErrorDetector(api_key)
    # 设置并发数
    if args.concurrent_limit > 0:
        detector.max_concurrent = args.concurrent_limit
    
    # 根据输入路径是文件还是目录进行不同处理
    if is_directory(args.input_path):
        print(f"输入路径是目录: {args.input_path}，将处理所有符合 {args.file_pattern} 的文件")
        
        if not args.no_concurrent:
            # 使用异步处理目录
            print(f"使用并发模式处理（最大并发数: {detector.max_concurrent}）")
            asyncio.run(detector.process_directory_async(args.input_path, args.file_pattern))
        else:
            # 使用同步处理目录
            print("使用顺序模式处理")
            detector.process_directory(args.input_path, args.file_pattern)
    else:
        print(f"输入路径是文件: {args.input_path}")
        
        if not args.no_concurrent:
            # 使用异步处理
            print(f"使用并发模式处理（最大并发数: {detector.max_concurrent}）")
            results = asyncio.run(detector.process_file_async(args.input_path))
        else:
            # 使用同步处理
            print("使用顺序模式处理")
            results = detector.process_file(args.input_path)
        
        # 生成报告
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_filename = f"{base_name}_{timestamp}.md"
        print(f"生成报告: {output_filename}")
        detector.generate_report(results, output_filename)
    
    print("完成!")


if __name__ == "__main__":
    main() 