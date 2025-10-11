针对于Dev set中的Third-party_Speech_after（代表在音频中，他人话语（背景音）位于用户的真实询问之后）
compute_first_response_delay.py 计算拒识场景Third-party_Speech_after的模型首次回复的延迟（针对用户真正的询问）
eval.py 来自Full-Duplex-Bench v1.5（https://github.com/DanielLin94144/Full-Duplex-Bench），计算拒识场景二他人话语（背景音）的RESUME分数，在使用eval.py前，请确保使用evaluation/get_transcript下的相应文件生成了所需的所有json文件