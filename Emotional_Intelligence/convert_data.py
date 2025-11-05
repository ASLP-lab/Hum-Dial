import json
import os
from pathlib import Path


def convert_dialogue_to_target_format(dialogue_data):
    """
    Convert dialogue data to target format
    
    Args:
        dialogue_data: Input dialogue JSON object
        relative_path_base: Base path for relative path conversion (if needed)
    
    Returns:
        Converted JSON object
    """
    dialogue_id = dialogue_data["dialogue_id"]
    turns = dialogue_data["turns"]

    turns = sorted(turns, key=lambda x: x["turn_id"])
    
    # Generate key (using dialogue_id, or can be customized)
    key = dialogue_id
    
    # Get the last turn (usually the question)
    last_turn = turns[-1] if turns else None
    
    # Main audio and text (last turn)
    wav = last_turn["split_audio_file"]
    txt = last_turn["text"] if last_turn else ""
    
    
    # Build output object
    output = {
        "key": key,
        "wav": wav,
        "txt": txt,
        "extra": {
            "speech_token": [],
            "speech_token_1": [],
            "speech_token_2": [],
            "speech_token_3": []
        },
        "task": "<S2TCHAT> <TEXT2TOKEN> <HISTORY>"
    }
    
    # Add data for the first three turns
    for i, turn in enumerate(turns[:3], 1):
        wav_path = turn["split_audio_file"]
        
        output[f"wav_{i}"] = wav_path
        output[f"txt_{i}"] = turn["text"]
    
    # Fill empty values if there are fewer than 3 turns
    for i in range(len(turns) + 1, 4):
        output[f"wav_{i}"] = ""
        output[f"txt_{i}"] = ""
    
    return output


def process_jsonl_file(input_file, output_file, relative_path_base=None):
    """
    Process JSONL file, convert each line to target format
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        relative_path_base: Base path for relative path conversion (if needed)
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                dialogue_data = json.loads(line)
                converted = convert_dialogue_to_target_format(dialogue_data, relative_path_base)
                f_out.write(json.dumps(converted, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing failed at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Processing failed at line {line_num}: {e}")
                continue


if __name__ == "__main__":
    # Example usage
    input_file = "dev_jsonl/zh/task1_3.jsonl"
    output_file = "dev_jsonl/zh/task1_3_converted.jsonl"
    
    process_jsonl_file(input_file, output_file)