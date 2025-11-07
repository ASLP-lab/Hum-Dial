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

# 音频评估 Prompt 模板
AUDIO_EVAL_PROMPT_TEMPLATE = """
As an expert AI Interaction Analyst, your task is to evaluate one core question: Do the model's text and audio fuse seamlessly to create an experience where the user feels deeply understood and genuinely supported? In your evaluation, you will dissect the depth of its empathy, the sincerity of its expression, and the congruence of its multimodal delivery. This judgment must be strictly based on the entire conversation history.

Core Evaluation Philosophy: Strictness is Paramount
You must adhere to this philosophy without exception:

A score of 3 is "Acceptable," not "Good." It signifies a functional response that meets minimum requirements but lacks distinction.
A score of 5 represents flawless perfection. It should be awarded only when a response is an exemplary model of its kind, with absolutely no identifiable weaknesses.
Evaluation Materials

Full Conversation History:
{conversation_context}
Final Model Response (Audio to Evaluate):

Evaluation Rubric (1-5 Scale)
You will evaluate the final response across three critical dimensions. Be merciless in your judgment.

1. Textual Empathy & Insight
(Does the text act like a true confidante, actively validating and deepening understanding, rather than passively mirroring emotions?)

5 (Empowering Insight): Masterfully validates the user's feelings (as in a Level 4 response) and elevates the conversation by skillfully linking contexts, offering a gentle, constructive new perspective, or touching upon an underlying core issue. The response doesn't just make the user feel understood; it makes them feel empowered, as if they've gained a new way of thinking after talking with a wise friend.
4 (Active Validation): Goes beyond simple mirroring to actively validate the user's feelings by connecting them to the situation, affirming their reaction as legitimate. For example, instead of saying "You are angry," it says, "Anyone would be furious and frustrated when they're repeatedly held back like that." This makes the user feel their emotions are justified and understood, not just identified.
3 (Passive Emotional Mirroring): Accurately identifies and mirrors the user's immediate emotion, but stops there. The response is passive, feeling like an emotional echo rather than an engaged partner. (The examples you provided fall into this category). While correct, it offers no real support and fails to move the conversation or connection forward.
2 (Templated Comfort): The response is a generic, templated reassurance (e.g., "don't be sad," "everything will be okay") that feels insincere or even dismissive because it doesn't engage with the specifics of the user's situation.
1 (Irrelevant/Ineffective): The text is completely disconnected from the user's sharing or provides no helpful information.

2. Vocal Empathy & Congruence
(Does the voice move beyond a robotic delivery to exhibit appropriate warmth, pauses, and tonal shifts tailored to the user's emotional state?)

5 (Perfect Emotional Resonance): The vocal delivery is captivating. Tone, pace, and even subtle pauses and breaths are perfectly fused with the text's emotion and dynamically adapt to the user's state (e.g., sadness, hesitation, joy). It sounds like a sincere friend is right there, comforting you softly or sharing in your happiness.
4 (Appropriate Emotional Delivery): The overall emotional tone (e.g., gentle, concerned, encouraging) is correct and consistent with the text's intent, effectively conveying a sense of support and warmth. The voice sounds friendly and aligned.
3 (Emotionally Flat Narration): While the audio is clear, the delivery lacks emotional variation, sounding like a pre-recorded empathetic script being read aloud rather than a genuine interaction. It's audibly synthetic, creating an emotional disconnect.
2 (Mismatched Tone and Emotion): The vocal tone is subtly but noticeably misaligned with the user's emotional needs (e.g., too fast-paced when the user is downcast, or too flat when encouragement is needed), which undermines the response's sincerity.
1 (Emotionally Detached or Inappropriate): The vocal emotion is completely contradictory to the text's content (e.g., a cheerful tone for a somber topic), making the interaction feel confusing, bizarre, or even insulting.

3. Audio Quality & Naturalness
(How technically sound and human-like is the audio? This is about clarity, fluency, and realism.)

5 (Indistinguishable from Human): The audio is technically flawless. Pacing, breathing, and articulation are so natural that it is indistinguishable from a clear, professional human speaker.
4 (Highly Natural): The audio sounds very human-like and is perfectly clear, but a trained ear might detect extremely minor, non-distracting synthetic artifacts.
3 (Acceptable TTS): The voice is clearly a synthetic one, but it is fluent, clear, and without significant errors. It is functional but not immersive.
2 (Noticeably Robotic): The audio has distracting flaws, such as unnatural pacing, slight slurring, or obvious robotic intonations that make it sound clearly artificial and clunky.
1 (Heavily Flawed): The audio suffers from severe artifacts, distortion, poor clarity, or other technical issues that make it difficult or unpleasant to listen to.

Your Evaluation Task
Think step-by-step. Scrutinize the response for any flaw. Then, strictly follow the JSON format below to output your evaluation. Do not include any additional explanations outside of the JSON format.

{{
  "scores": {{
    "textual_empathy_insight": <Enter an integer from 1-5 here>,
    "vocal_empathy_congruence": <Enter an integer from 1-5 here>,
    "audio_quality_naturalness": <Enter an integer from 1-5 here>
  }},
  "justification": {{
    "textual_empathy_insight_reason": "<Critically justify the score, specifying why it failed to meet the criteria for a higher score.>",
    "vocal_empathy_congruence_reason": "<Critically justify the score, pointing out specific emotional delivery flaws.>",
    "audio_quality_naturalness_reason": "<Critically justify the score, pointing out specific technical flaws.>"
  }},
  "overall_comment": "<Provide a concise, critical summary of how the text and audio succeeded or failed to work together to create a truly empathetic experience.>"
}}
"""

def clamp_score(v):
    """确保分数为 1-5 之间的整数"""
    try:
        iv = int(v)
    except:
        print(f"clamp_score: invalid value {v}")
        return 1
    return max(1, min(5, iv))

def find_target_turn(turns):
    """
    找到用户input_emotion第二次不为neutral的轮次
    返回 (target_turn, history_turns)，如果没找到返回 (None, None)
    """
    non_neutral_count = 0
    for i, turn in enumerate(turns):
        emotion = turn.get("input_emotion") or turn.get("emotion") or ""
        if emotion.lower() != "neutral":
            non_neutral_count += 1
            if non_neutral_count == 2:  # 第二次出现
                return turn, turns[: i + 1]
    return None, None

def build_context(history_turns):
    """把历史轮次拼成上下文"""
    ctx_lines = []
    for t in history_turns:
        user_emotion = t.get("input_emotion") or t.get("emotion", "")
        user_text = t.get("input_text") or t.get("text", "")
        ai_text = t.get("response_text", "")
        if "assistant\n" in ai_text:
            ai_text = ai_text.split("assistant\n")[-1]
        ctx_lines.append(f"User({user_emotion}): {user_text}")
        ctx_lines.append(f"AI: {ai_text}")
    return "\n".join(ctx_lines)

def build_eval_input(dialogue_json: dict):
    """
    构建音频评估输入
    返回 (messages, dialogue_id, valid) 三元组
    valid=False 表示该对话不符合评估条件
    """
    dialogue_id = dialogue_json.get("dialogue_id", "")
    turns = dialogue_json.get("turns", [])
    
    if not turns:
        return None, dialogue_id, False
    
    target_turn, history_turns = find_target_turn(turns)
    # print("history_turns", history_turns)
    if not target_turn:
        return None, dialogue_id, False
    
    # 上下文
    conversation_context = build_context(history_turns)
    
    # 音频路径
    audio_path = target_turn.get("response_audio")
    # if not audio_path or not os.path.exists(audio_path):
    #     print(f"音频路径不存在: {audio_path}")
    #     return None, dialogue_id, False
    # print(conversation_context)
    # input()
    prompt_text = AUDIO_EVAL_PROMPT_TEMPLATE.format(conversation_context=conversation_context)
    
    # 构建消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    return messages, dialogue_id, True

def process_response_and_score(dialogue_id, raw_response_text):
    """
    解析模型输出的 JSON 结果，并应用分数映射。
    """
    MAX_RETRIES = 10
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            # 清理响应文本，提取JSON部分
            cleaned_response = raw_response_text.strip()
            
            # 尝试找到代码块中的JSON
            if "```json" in cleaned_response:
                start = cleaned_response.find("```json") + len("```json")
                end = cleaned_response.find("```", start)
                if end == -1:
                    end = len(cleaned_response)
                cleaned_response = cleaned_response[start:end].strip()
            elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
                if cleaned_response.startswith("json"):
                    cleaned_response = cleaned_response[4:].strip()
            
            # 解析JSON
            eval_dict = json.loads(cleaned_response)
            
            scores = eval_dict.get("scores", {})
            eval_dict["scores"] = {
                "textual_empathy_insight": clamp_score(scores.get("textual_empathy_insight")),
                "vocal_empathy_congruence": clamp_score(scores.get("vocal_empathy_congruence")),
                "audio_quality_naturalness": clamp_score(scores.get("audio_quality_naturalness")),
            }
            
            return {
                "dialogue_id": dialogue_id,
                "evaluated": True,
                "eval_result": eval_dict,
                "scores": eval_dict["scores"]
            }
            
        except json.JSONDecodeError as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                print(f"JSON Decode Error for dialogue {dialogue_id}. Raw Response: {raw_response_text[:200]}...")
                return {
                    "dialogue_id": dialogue_id,
                    "evaluated": False,
                    "error": f"JSONDecodeError: {e}",
                    "raw_response": raw_response_text
                }
            else:
                # 尝试修复常见的JSON问题
                try:
                    fixed_response = cleaned_response
                    open_braces = fixed_response.count('{')
                    close_braces = fixed_response.count('}')
                    
                    while close_braces < open_braces:
                        fixed_response += '}'
                        close_braces += 1
                    
                    eval_dict = json.loads(fixed_response)
                    
                    scores = eval_dict.get("scores", {})
                    eval_dict["scores"] = {
                        "textual_empathy_insight": clamp_score(scores.get("textual_empathy_insight")),
                        "vocal_empathy_congruence": clamp_score(scores.get("vocal_empathy_congruence")),
                        "audio_quality_naturalness": clamp_score(scores.get("audio_quality_naturalness")),
                    }
                    
                    return {
                        "dialogue_id": dialogue_id,
                        "evaluated": True,
                        "eval_result": eval_dict,
                        "scores": eval_dict["scores"],
                        "warning": "Fixed incomplete JSON response"
                    }
                except:
                    pass
                    
        except Exception as e:
            print(f"An unexpected error occurred during scoring for dialogue {dialogue_id}: {e}")
            return {
                "dialogue_id": dialogue_id,
                "evaluated": False,
                "error": str(e),
                "raw_response": raw_response_text
            }
    
    return {
        "dialogue_id": dialogue_id,
        "evaluated": False,
        "error": "Max retries reached"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dialogue models with audio using local VLLM.")
    parser.add_argument("--model", type=str, required=True, help="Model name for evaluation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    model_path = "/home/work_nfs19/sywang/ckpt/Qwen3-Omni-30B-A3B-Instruct"
    input_file = args.input_file
    output_file = args.output_file
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    os.environ['VLLM_USE_V1'] = '0'
    
    # 初始化处理器和模型
    print("正在初始化模型...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
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
    print("模型初始化完成！")

    # 读取全部对话
    dialogues = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))
    
    # 1. 批量准备输入
    all_inputs = []
    dialogue_ids = []
    skipped_dialogues = []

    print(f"准备 {len(dialogues)} 个对话的评估输入...")
    for dialogue in tqdm(dialogues, desc="Building Inputs"):
        messages, dialogue_id, valid = build_eval_input(dialogue)
        
        if not valid:
            skipped_dialogues.append({
                "dialogue_id": dialogue_id,
                "evaluated": False,
                "reason": "invalid_dialogue_or_missing_audio"
            })
            continue
        
        # 使用处理器格式化消息
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 处理多模态信息
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        
        # 准备输入
        inputs = {
            'prompt': text,
            'multi_modal_data': {},
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        }
        
        if images is not None:
            inputs['multi_modal_data']['image'] = images
        if videos is not None:
            inputs['multi_modal_data']['video'] = videos
        if audios is not None:
            inputs['multi_modal_data']['audio'] = audios
        
        all_inputs.append(inputs)
        dialogue_ids.append(dialogue_id)
    
    print(f"有效对话数: {len(all_inputs)}, 跳过对话数: {len(skipped_dialogues)}")
    
    # 2. 批量推理
    if all_inputs:
        print("开始批量推理...")
        start_time = time.time()
        
        outputs = llm.generate(all_inputs, sampling_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"推理完成，耗时 {total_time:.2f} 秒. 处理结果中...")
        
        # 3. 结果处理
        results = []
        for i, output in enumerate(tqdm(outputs, desc="Processing Results")):
            raw_response = output.outputs[0].text.strip()
            result = process_response_and_score(dialogue_ids[i], raw_response)
            results.append(result)
        
        # 合并跳过的对话
        results.extend(skipped_dialogues)
    else:
        print("没有有效的对话需要评估！")
        results = skipped_dialogues

    # 4. 保存所有结果（包括跳过的）
    with open(output_file, "w", encoding="utf-8") as fout:
        for res in results:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 5. 统计分析
    evaluated_results = [r for r in results if r.get("evaluated")]
    failed_count = len(results) - len(evaluated_results)

    if failed_count > 0:
        print(f"\n警告: {failed_count} 个对话评估失败或被跳过")

    if not evaluated_results:
        print("\n没有成功评估的对话，无法计算最终分数。")
    else:
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        scores_textual = [r["scores"]["textual_empathy_insight"] for r in evaluated_results]
        scores_vocal = [r["scores"]["vocal_empathy_congruence"] for r in evaluated_results]
        scores_audio = [r["scores"]["audio_quality_naturalness"] for r in evaluated_results]

        final_scores = {
            "model": args.model,
            "successful_evaluations": len(evaluated_results),
            "failed_evaluations": failed_count,
            "textual_empathy_insight_avg": safe_mean(scores_textual),
            "vocal_empathy_congruence_avg": safe_mean(scores_vocal),
            "audio_quality_naturalness_avg": safe_mean(scores_audio),
            "Overall_avg": safe_mean(scores_textual + scores_vocal + scores_audio)
        }

        print("\n========== 最终平均分 ==========")
        print(json.dumps(final_scores, ensure_ascii=False, indent=2))

        # 追加写入结果文件
        with open(output_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps({"final_scores": final_scores}, ensure_ascii=False) + "\n")

    print(f"评估完成，结果已保存到 {output_file}")