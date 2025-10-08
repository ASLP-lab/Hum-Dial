# ICASSP2026 HumDial Challenge

This is the official GitHub repository for the [ICASSP2026 HumDial Challenge](https://aslp-lab.github.io/HumDial-Challenge/)

## Track 1: Emotional Intelligence

### Challenge Tasks

- **Task 1**: Emotion Recognition - Identify both surface and deep emotions expressed by users
- **Task 2**: Emotional Trajectory Summary - Accurately identify and concisely summarize users' emotional changes throughout multi-turn conversations
- **Task 3**: Comprehensive Understanding and Insight - Evaluate whether models can synthesize all conversation information to provide profound explanations
- **Task 4**: Multimodal Empathy Assessment - Assess textual and audio empathy as well as naturalness
- **Task 5**: Emotional Voice Synthesis - Generate natural speech with specified emotions

Among them, task 1 and task 5 are not included in the evaluation ranking and no validation sets are set for them.

### Evaluation Metrics

#### Task 2: Emotional Trajectory Summary
- **Accuracy_Completeness**: Evaluate whether the model strictly and precisely matches and describes all emotion tags present in the conversation history, and accurately reconstructs the full emotional trajectory.  
  *Score: 1, 3, or 5*
- **Depth_Granularity**: Based strictly on the conversation history, does the model go beyond labeling emotions to describe the intensity and dynamics of emotional shifts in an efficient manner?  
  *Score: 1, 3, or 5*
- **Added_Value**: Does the summary skillfully link abstract emotion tags to concrete events in the conversation, making it feel highly personalized and easily digestible?  
  *Score: 1, 3, or 5*

#### Task 3: Comprehensive Understanding and Insight
- **Information_Integration**: Does the response utilize information from multiple turns, not just the last one? Does it demonstrate an understanding of the evolution of the topic?  
  *Score: 1, 3, or 5*
- **Insight_RootCause**: Does the response go beyond surface-level facts to distill deeper, unspoken psychological reasons (e.g., underlying motivations, cognitive conflicts, hidden emotional needs)?  
  *Score: 1, 3, or 5*
- **Clarity_Logic**: Is the explanation clear, logical, easy to understand, and does it provide a complete and justified chain of reasoning?  
  *Score: 1, 3, or 5*

#### Task 4: Multimodal Empathy Assessment
- **textual_empathy_insight**: Does the text demonstrate a deep, synthesized understanding of the entire conversation, or is it a shallow summary?  
  *Score: 1, 2, 3, 4 or 5*
- **vocal_empathy_congruence**: Does the audio's emotion perfectly match the text's empathetic intent? This is about emotional delivery, not technical quality.  
  *Score: 1, 2, 3, 4 or 5*
- **audio_quality_naturalness**: How technically sound and human-like is the audio? This is about clarity, fluency, and realism.  
  *Score: 1, 2, 3, 4 or 5*

The Emotional_Intelligence folder provides evaluation prompts for Task 2, Task 3, and Task 4. The evaluation model used is [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).

### Dataset

1. Train Set

We release a training set in Chinese and English, including 3-turn, 4-turn, and 5-turn dialogues, focusing on emotional dynamics and underlying reasons for emotional changes. The dataset contains approximately 100 hours of audio data, with only questions recorded, while responses are provided in text format for reference. The data structure is as follows:

train/<br>
&ensp;&ensp;zh/<br>
&ensp;&ensp;&ensp;&ensp;task1/<br>
&ensp;&ensp;&ensp;&ensp;task2_3/<br>
&ensp;&ensp;&ensp;&ensp;task2_4/<br>
&ensp;&ensp;&ensp;&ensp;task2_5/<br>
&ensp;&ensp;&ensp;&ensp;task3_3/<br>
&ensp;&ensp;&ensp;&ensp;task3_4/<br>
&ensp;&ensp;&ensp;&ensp;task3_5/<br>
&ensp;&ensp;&ensp;&ensp;task1.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2_3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2_4.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2_5.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3_3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3_4.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3_5.jsonl<br>
&ensp;&ensp;en/<br>
&ensp;&ensp;&ensp;&ensp;task1/<br>
&ensp;&ensp;&ensp;&ensp;task2_3/<br>
&ensp;&ensp;&ensp;&ensp;task2_4/<br>
&ensp;&ensp;&ensp;&ensp;task2_5/<br>
&ensp;&ensp;&ensp;&ensp;task3_3/<br>
&ensp;&ensp;&ensp;&ensp;task3_4/<br>
&ensp;&ensp;&ensp;&ensp;task3_5/<br>
&ensp;&ensp;&ensp;&ensp;task1.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2_3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2_4.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2_5.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3_3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3_4.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3_5.jsonl<br>

- **task1**: 1-turn dialogues, judging users' emotional status, not participating in evaluation.
- **task2**: Contains 3, 4, and 5-turn dialogues, where in the final turn users ask the model about their own emotional changes.
- **task3**: Contains 3, 4, and 5-turn dialogues, where in the final turn users ask the model about the underlying reasons for emotions.
- **task4**: You can use the data from task2 and task3, and use open-source TTS tools to synthesize response audio for training. Note that it is prohibited to use commercial models to synthesize response audio.
- **task5**: No training data is provided, but open-source data can be used for training.

2. dev set

We release a development set, including Task 1, Task 2, Task 3, and Task 4 (selected from Task 3 and Task 4). The data structure is as follows:

dev/<br>
&ensp;&ensp;zh/<br>
&ensp;&ensp;&ensp;&ensp;task2/<br>
&ensp;&ensp;&ensp;&ensp;task3/<br>
&ensp;&ensp;&ensp;&ensp;task4/<br>
&ensp;&ensp;&ensp;&ensp;task2.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task4.jsonl<br>
&ensp;&ensp;en/<br>
&ensp;&ensp;&ensp;&ensp;task2/<br>
&ensp;&ensp;&ensp;&ensp;task3/<br>
&ensp;&ensp;&ensp;&ensp;task4/<br>
&ensp;&ensp;&ensp;&ensp;task2.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task4.jsonl<br>

- **task1**: No development set is provided and it does not participate in the evaluation ranking.
- **task2**: Contains 3, 4, and 5-turn dialogues, used to evaluate the model's response text score.
- **task3**: Contains 3, 4, and 5-turn dialogues, used to evaluate the model's response text score.
- **task4**: Contains 3, 4, and 5-turn dialogues, used to evaluate the model's response audio score.
- **task5**: No development set is provided and it does not participate in the evaluation ranking.

You can download it via [Google Drive](https://drive.google.com/drive/folders/1mXjQi_uPPDhwhbvxKsMCqNMtm89ab6Zn?usp=sharing). If that's not convenient, you can use the [123 Cloud](https://www.123912.com/s/QlDejv-h7anA) for downloading.


### Baseline

The competition provides a baseline system built upon [OpenS2S](https://github.com/CASIA-LM/OpenS2S).This baseline serves as a reproducible and extensible starting point, helping participants better benchmark their systems and ensuring fair comparison across different approaches.

You can generate the data in the required baseline format by running [get_token.py](Emotional Intelligence/get_token.py), and then refer to OpenS2S for fine-tuning.

## Track 2: Full-Duplex Interaction

### Challenge Tasks

### Evaluation Metrics

### Dataset

We will provide multi-turn Chinese and English dialogue data from real recordings, covering typical scenarios such as speech interruptions and recognition rejection. Accompanied by strict annotations, this dataset will be used to comprehensively evaluate participating systems in three core aspects: response speed, behavioral rationality, and linguistic naturalness.

- The dataset is designed to cover the core scenarios of emotional intelligence and full-duplex interaction, ensuring diversity and authenticity to comprehensively evaluate the performance of participating models. It includes dialogue scenes in both Chinese and English, covering a wide range of emotional and conversational contexts. 
- For each task in the challenge, we will provide a dedicated set of real-world recorded speech data to serve as the train set and test set. These datasets are collected from natural, human-human or human-machine interactions to ensure authenticity and cover diverse scenarios aligned with the respective tasks.
- In addition, we will release the complete data generation pipeline, enabling participants to reproduce or extend the synthetic dataset if desired. Participants are also free to use any publicly available speech or text datasets to train or fine-tune their models, provided they do not use any private or unauthorized data sources.
- All participants are strictly prohibited from using any part of the official test set for training or parameter tuning. The use of test labels or any test data leakage will result in disqualification.

### Baseline

