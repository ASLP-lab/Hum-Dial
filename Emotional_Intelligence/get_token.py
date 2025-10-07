"""
this script is used to extract speech tokens from multi-turn dialogue data
"""
import json
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
import torchaudio
import torch
from tqdm import tqdm

# Speech tokenizer
tokenizer_path = "glm4/glm-4-voice-tokenizer"
whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to("cuda")
feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path) 

def process_multiturn_dialogue(data):
    """
    Process multi-turn dialogue data and extract speech tokens
    """
    messages = []
    
    for turn in data['turns']:
        user_content = []
        
        # Add audio input (no text prompt needed)
        user_content.append({
            "text": "", 
            "audio": turn['split_audio_file'], 
            "speech_units": "",
            "spk_emb": ""
        })
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        response_audio_path = turn['response_audio']
        audio_tokens = extract_speech_token(
            whisper_model, feature_extractor, [response_audio_path]
        )[0]

        # Convert tokens to the format <|audio_0|><|audio_1|>...
        speech_units_str = "".join([f"<|audio_{token}|>" for token in audio_tokens])
        
        assistant_content = [{
            "text": turn['response_txt'],
            "audio": "", 
            "speech_units": speech_units_str,
            "spk_emb": ""
        }]
        
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
    
    return {"messages": messages}

# Example usage
if __name__ == "__main__":
    input_path = "/home/work_nfs19/sywang/code/OpenS2S/train.jsonl"
    output_path = "/home/work_nfs19/sywang/code/OpenS2S/train_token.jsonl"
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile):
            data = json.loads(line.strip())
            processed_data = process_multiturn_dialogue(data)
            outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
    
    print(f"Processed data saved to {output_path}")
