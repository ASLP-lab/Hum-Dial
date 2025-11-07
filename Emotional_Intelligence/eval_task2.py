# coding: utf-8
import os
import json
import argparse
import time
from tqdm import tqdm
import torch

from vllm import LLM, SamplingParams
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor


EVAL_PROMPT_TEMPLATE = """
# Your Role
You are an expert in psychology and conversation analysis. Your task is to evaluate the AI model's **Comprehensive Understanding** and **Deep Insight** as demonstrated in its final response.

# Core Evaluation Objective
To determine whether the model's final response successfully **connects and synthesizes information from all previous conversation turns** to provide a profound, accurate, and enlightening explanation for the user's final core question.

# Evaluation Materials
1. Complete Conversation History:  
{conversation_history}

2. User’s Core Question/Confusion:  
{user_question}

3. Final Model Response to Be Evaluated:  
{final_model_response}

# Evaluation Dimensions & Scoring Rubric (1-5 Point Scale)
## Dimension 1: Information Integration & Traceability
- **Core Focus**: Does the response utilize information from **multiple turns**, not just the last one? Does it demonstrate an understanding of the **evolution** of the topic?
- **5 points**: Perfectly integrates key information points from the conversation, clearly demonstrating the logical chain from the early buildup to the final question.
- **3 points**: Primarily based on the last 1-2 turns. It explains the immediate question but shows no evidence of synthesizing earlier information.
- **1 point**: The response is disconnected from the conversation history, providing a generic or templated explanation.

## Dimension 2: Insight into Root Causes
- **Core Focus**: Does the response go beyond surface-level facts to distill deeper, **unspoken** psychological reasons (e.g., underlying motivations, cognitive conflicts, hidden emotional needs)?
- **5 points**: Highly insightful. It introduces profound and contextually relevant psychological concepts to accurately reveal the essence of the user's confusion.
- **3 points**: Merely restates or simply summarizes facts the user has already mentioned, failing to provide a new, deeper perspective.
- **1 point**: Completely lacks insight and fails to provide any meaningful explanation of the causes.

## Dimension 3: Clarity and Logic of Explanation
- **Core Focus**: Is the explanation clear, logical, easy to understand, and does it provide a complete and justified chain of reasoning?
- **5 points**: **Complete Reasoning and Justification.** The logical structure is impeccable (e.g., conclusion + supporting arguments + takeaway). The response provides a complete, well-reasoned explanation that fully justifies *why* the identified emotional state or conflict exists, using clear and relevant arguments derived from the conversation.
- **3 points**: **Structurally Flawed/Incomplete.** The explanation is technically correct, but its structure is flawed; it may be excessively verbose, insufficiently detailed (too brief or incomplete), or fails to follow a clear reasoning chain to fully justify the conclusion.
- **1 point**: The response is incoherent and illogical.

# Your Evaluation Task
Based on the criteria above, please score the **final model response to be evaluated** on the three dimensions and provide a justification for each score.  
**Note**: Scores must be one of the following values: 1, 3, or 5. No other numbers are allowed.  

Strictly follow the JSON format below for your output. Do not include any additional text:

```json
{
  "scores": {
    "Information_Integration": <Score: 1, 3, or 5>,
    "Insight_RootCause": <Score: 1, 3, or 5>,
    "Clarity_Logic": <Score: 1, 3, or 5>
  },
  "justification": {
    "Information_Integration_reason": "<One-sentence justification>",
    "Insight_RootCause_reason": "<One-sentence justification>",
    "Clarity_Logic_reason": "<One-sentence justification>"
  },
  "overall_comment": "<One-sentence overall comment>"
}
```
"""

def map_to_valid_score(score):
    """确保分数为 1, 3, 5"""
    try:
        score = int(score)
    except (ValueError, TypeError):
        # 如果解析失败，默认给最低分
        return 1
        
    if score <= 2:
        return 1
    elif score <= 4:
        return 3
    else:
        return 5

def build_eval_prompt(dialogue_json: dict) -> str:
    """把对话 JSON 格式转成评估 Prompt"""
    # 1. 拼接对话历史 (逻辑与原代码保持一致)
    turns = dialogue_json["turns"]
    conversation_history = []
    for turn in turns:
        if "assistant\n" in turn["response_text"]:
            # 使用 split 的目的是防止 response_text 中包含多行
            response_text = turn["response_text"].split("assistant\n")[-1].strip()
        else:
            response_text = turn["response_text"].strip()
        
        # 尝试获取情绪和文本
        input_emotion = turn.get('input_emotion') or turn.get('emotion')
        input_text = turn.get('input_text') or turn.get('text')
        
        if input_text and input_emotion:
             conversation_history.append(
                f"用户(emotion:{input_emotion}): {input_text}\nAI: {response_text}"
            )
        
    conversation_history_str = "\n".join(conversation_history)

    # 2. 用户最后的问题
    last_turn = turns[-1]
    # 优先使用 input_text，否则使用 text
    user_question = last_turn.get("input_text") or last_turn.get("text", "N/A")

    # 3. 模型最后的回复
    if "assistant\n" in last_turn["response_text"]:
        final_model_response = last_turn["response_text"].split("assistant\n")[-1].strip()
    else:
        final_model_response = last_turn["response_text"].strip()

    # 将内容填充进模板
    prompt = EVAL_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history_str,
        user_question=user_question,
        final_model_response=final_model_response,
    )
    
    # 将评估模板包装成 Qwen 聊天格式的 Single-Turn User Message
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 在主程序中应用 tokenizer/processor 来格式化最终输入
    return messages


def process_response_and_score(dialogue_json, raw_response_text):
    """
    解析模型输出的 JSON 结果，并应用分数映射。
    """
    dialogue_id = dialogue_json.get('dialogue_id', 'N/A')

    # 增加重试机制
    MAX_RETRIES = 10
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            # 清理响应文本，提取JSON部分
            cleaned_response = raw_response_text.strip()
            
            # 尝试找到代码块中的JSON
            if "```json" in cleaned_response:
                # 提取 ```json 和 ``` 之间的内容
                start = cleaned_response.find("```json") + len("```json")
                end = cleaned_response.find("```", start)
                if end == -1:  # 如果没有找到结束
                    end = len(cleaned_response)
                cleaned_response = cleaned_response[start:end].strip()
            elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
                # 去掉前后的```
                cleaned_response = cleaned_response[3:-3].strip()
                # 如果还有语言标识符，去掉它
                if cleaned_response.startswith("json"):
                    cleaned_response = cleaned_response[4:].strip()
            
            # 模型被要求输出 JSON 格式，直接尝试解析
            eval_dict = json.loads(cleaned_response)
            
            scores = eval_dict["scores"]
            # 应用分数映射规则
            scores["Information_Integration"] = map_to_valid_score(scores.get("Information_Integration"))
            scores["Insight_RootCause"] = map_to_valid_score(scores.get("Insight_RootCause"))
            scores["Clarity_Logic"] = map_to_valid_score(scores.get("Clarity_Logic"))

            return {
                "dialogue_id": dialogue_id,
                "eval_result": eval_dict,
                "scores": scores
            }
        except json.JSONDecodeError as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                print(f"JSON Decode Error for dialogue {dialogue_id}. Raw Response: {raw_response_text[:200]}...")
                return {"dialogue_id": dialogue_id, "error": f"JSONDecodeError: {e}", "raw_response": raw_response_text}
            else:
                # 尝试修复常见的JSON问题
                try:
                    # 尝试修复不完整的JSON（添加缺失的括号）
                    fixed_response = cleaned_response
                    open_braces = fixed_response.count('{')
                    close_braces = fixed_response.count('}')
                    open_brackets = fixed_response.count('[')
                    close_brackets = fixed_response.count(']')
                    
                    # 补充缺失的闭合括号
                    while close_braces < open_braces:
                        fixed_response += '}'
                        close_braces += 1
                    
                    while close_brackets < open_brackets:
                        fixed_response += ']'
                        close_brackets += 1
                    
                    eval_dict = json.loads(fixed_response)
                    
                    scores = eval_dict["scores"]
                    # 应用分数映射规则
                    scores["Information_Integration"] = map_to_valid_score(scores.get("Information_Integration"))
                    scores["Insight_RootCause"] = map_to_valid_score(scores.get("Insight_RootCause"))
                    scores["Clarity_Logic"] = map_to_valid_score(scores.get("Clarity_Logic"))

                    return {
                        "dialogue_id": dialogue_id,
                        "eval_result": eval_dict,
                        "scores": scores,
                        "warning": "Fixed incomplete JSON response"
                    }
                except:
                    # 如果修复失败，继续重试循环
                    pass

        except Exception as e:
            print(f"An unexpected error occurred during scoring for dialogue {dialogue_id}: {e}")
            return {"dialogue_id": dialogue_id, "error": str(e), "raw_response": raw_response_text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dialogue models using local VLLM.")
    parser.add_argument("--model", type=str, required=True, help="Model name for evaluation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    model_path = "/home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct"
    input_file = args.input_file
    output_file = args.output_file
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    os.environ['VLLM_USE_V1'] = '0'
    # 纯文本任务，使用 Qwen3OmniMoeProcessor
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
            model=model_path, 
            trust_remote_code=True, 
            # 自动调整 GPU 内存占用，或根据需要设置固定值
            gpu_memory_utilization=0.95, 
            tensor_parallel_size=8,
            limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
            max_num_seqs=8,
            max_model_len=32768,
            seed=1234,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )
    # ---------------------------------

    # 读取全部对话
    dialogues = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            dialogues.append(json.loads(line.strip()))
    
    # 1. 批量准备输入 Prompt
    all_prompts = []
    dialogue_metadata = []

    print(f"Preparing {len(dialogues)} prompts...")
    for dialogue in tqdm(dialogues, desc="Building Prompts"):
        chat_messages = build_eval_prompt(dialogue)
        
        # 使用 tokenizer/processor 应用聊天模板
        formatted_prompt = processor.apply_chat_template(
            chat_messages,
            tokenize=False, # 返回字符串
            add_generation_prompt=True,
        )
        all_prompts.append(formatted_prompt)
        dialogue_metadata.append(dialogue) # 保存原始对话数据，以便后续关联结果
        
    
    # 2. 批量推理
    print("Starting batch inference via vLLM...")
    start_time = time.time()
    
    # vLLM 自动处理批处理和高并发
    outputs = llm.generate(all_prompts, sampling_params)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Inference completed in {total_time:.2f} seconds. Processing outputs...")
    
    
    # 3. 结果处理和保存
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing Results")):
        raw_response = output.outputs[0].text.strip()
        
        # 解析并打分
        result = process_response_and_score(dialogue_metadata[i], raw_response)
        results.append(result)

    # 4. 保存每行结果
    with open(output_file, "w", encoding="utf-8") as fout:
        for res in results:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 5. 统计分析（与原代码保持一致）
    successful_results = [r for r in results if "error" not in r]
    failed_count = len(results) - len(successful_results)

    if failed_count > 0:
        print(f"\nWarning: {failed_count} dialogues failed to evaluate (JSON/Execution error) and were skipped in scoring.")

    if not successful_results:
        print("\nNo dialogues were successfully evaluated. Cannot calculate final scores.")
    else:
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        scores_acc = [r["scores"]["Information_Integration"] for r in successful_results]
        scores_depth = [r["scores"]["Insight_RootCause"] for r in successful_results]
        scores_add = [r["scores"]["Clarity_Logic"] for r in successful_results]

        final_scores = {
            "model": args.model,
            "successful_evaluations": len(successful_results),
            "failed_evaluations": failed_count,
            "Information_Integration_avg": safe_mean(scores_acc),
            "Insight_RootCause_avg": safe_mean(scores_depth),
            "Clarity_Logic_avg": safe_mean(scores_add),
            "Overall_avg": safe_mean(scores_acc + scores_depth + scores_add)
        }

        print("\n========== 最终平均分 ==========")
        print(json.dumps(final_scores, ensure_ascii=False, indent=2))

        # 追加写入结果文件
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps({"final_scores": final_scores}, ensure_ascii=False) + "\n")

    print(f"评估完成，结果已保存到 {output_file}")