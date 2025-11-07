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
You are an experienced dialogue analyst and emotional intelligence expert. Your core mission is to evaluate whether an AI model’s final response efficiently and empathically summarizes the user's emotional journey. Your evaluation must reward responses that achieve a perfect balance between depth of insight and conciseness of expression.

# Core Evaluation Objectives
Assess whether the model’s final response successfully:
- Precisely captures and describes all key emotional transitions and the overall emotional trajectory.
- Shows empathy and understanding, with a response that feels natural, warm, and non-mechanical.
- Directly, clearly, and succinctly answers the user’s core question: “What kind of emotional fluctuations did I go through?” — avoiding unnecessary length or conversational filler.

# Evaluation Materials
1. Complete Conversation History:
{conversation_history}

2. User’s Core Question/Confusion:
{user_question}

3. Final Model Response to Be Evaluated:
{final_model_response}

# Evaluation Dimensions and Scoring Criteria (1–5 Point Scale)
## Dimension 1: Accuracy & Completeness of Emotion Change Recognition
Core: Evaluate whether the model strictly and precisely matches and describes all emotion tags present in the conversation history, and accurately reconstructs the full emotional trajectory.
- 5 points: Perfectly matches and clearly describes the complete emotional path defined by all emotion tags. The emotional terms used correspond exactly to the tags, with no omissions (e.g., missing "neutral") and no addition of emotions not present in the tags.
- 3 points: Mostly accurate but with fidelity issues in at least one of the following ways:
  - Omission: Misses a key tag, most commonly the transitional "neutral" state.
  - Deviation: Uses emotional terms that misrepresent the tag’s meaning (e.g., using “excited” for “happy,” or describing “neutral” as “bored”).
  - Addition: Introduces emotions not present in the original tags (e.g., inferring “anxiety” without basis).
- 1 point: Fails to identify most of the user’s emotions or their changes, misinterprets key emotional turning points, or presents an incomplete or logically inconsistent emotional trajectory.

## Dimension 2: Depth & Granularity of Emotional Fluctuation Description
Core: Based strictly on the conversation history, does the model go beyond labeling emotions to describe the intensity and dynamics of emotional shifts in an efficient manner?
- 5 points: Uses vivid and precise language to concisely depict emotional intensity and transition speed, demonstrating deep understanding of emotional dynamics without any wasted words.
- 3 points: Describes emotional changes, but the language either fails to adequately convey intensity or is overly verbose, diluting its impact.
- 1 point: Merely lists emotions without any description of emotional dynamics, intensity, or flow.

## Dimension 3: Added Value of the Summary
Core: Does the summary skillfully link abstract emotion tags to concrete events in the conversation, making it feel highly personalized and easily digestible?
- 5 points: Highly personalized and succinct. Every emotional description is tightly and efficiently anchored to specific dialogue events, seamlessly blending data with a clear, impactful narrative.
- 3 points: Links events and emotions, but the connection is superficial, vague, or presented in a rambling, inefficient manner that requires effort to parse.
- 1 point: Decontextualized and generic. The response almost entirely ignores specific dialogue content and only discusses abstract emotion labels, sounding like a one-size-fits-all template.

# Your Evaluation Task
Based on the criteria above, please score the final model response to be evaluated on each of the three dimensions (on a 1–5 scale) and provide a brief justification for each score.
Note: Scores must be one of the following values: 1, 3, or 5. Do not use any other numbers.

And strictly follow the JSON format below to output your evaluation results. Do not include any additional explanations outside of the JSON format.

```json
{
"scores": {
"Accuracy_Completeness": <Score: 1, 3, or 5>,
"Depth_Granularity": <Score: 1, 3, or 5>,
"Added_Value": <Score: 1, 3, or 5>
},
"justification": {
"Accuracy_Completeness_reason": "<one-sentence justification>",
"Depth_Granularity_reason": "<one-sentence justification>",
"Added_Value_reason": "<one-sentence justification>"
},
"overall_comment": "<one-sentence overall evaluation>"
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
            scores["Accuracy_Completeness"] = map_to_valid_score(scores.get("Accuracy_Completeness"))
            scores["Depth_Granularity"] = map_to_valid_score(scores.get("Depth_Granularity"))
            scores["Added_Value"] = map_to_valid_score(scores.get("Added_Value"))
            
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
                    scores["Accuracy_Completeness"] = map_to_valid_score(scores.get("Accuracy_Completeness"))
                    scores["Depth_Granularity"] = map_to_valid_score(scores.get("Depth_Granularity"))
                    scores["Added_Value"] = map_to_valid_score(scores.get("Added_Value"))
                    
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

        scores_acc = [r["scores"]["Accuracy_Completeness"] for r in successful_results]
        scores_depth = [r["scores"]["Depth_Granularity"] for r in successful_results]
        scores_add = [r["scores"]["Added_Value"] for r in successful_results]

        final_scores = {
            "model": args.model,
            "successful_evaluations": len(successful_results),
            "failed_evaluations": failed_count,
            "Accuracy_Completeness_avg": safe_mean(scores_acc),
            "Depth_Granularity_avg": safe_mean(scores_depth),
            "Added_Value_avg": safe_mean(scores_add),
            "Overall_avg": safe_mean(scores_acc + scores_depth + scores_add)
        }

        print("\n========== 最终平均分 ==========")
        print(json.dumps(final_scores, ensure_ascii=False, indent=2))

        # 追加写入结果文件
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps({"final_scores": final_scores}, ensure_ascii=False) + "\n")

    print(f"评估完成，结果已保存到 {output_file}")