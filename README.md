# ICASSP2026 HumDial Challenge

This is the official GitHub repository for the [ICASSP2026 HumDial Challenge](https://aslp-lab.github.io/HumDial-Challenge/)

## Track 1: Emotional Intelligence

### Challenge Tasks

- **Task 1**: Emotion Recognition - Identify both surface and deep emotions expressed by users.
- **Task 2**: Emotional Trajectory Summary - Accurately identify and concisely summarize users' emotional changes throughout multi-turn conversations.
- **Task 3**: Comprehensive Understanding and Insight - Evaluate whether models can synthesize all conversation information to provide profound explanations.
- **Task 4**: Multimodal Empathy Assessment - Assess textual and audio empathy as well as naturalness.
- **Task 5**: Emotional Voice Synthesis - Generate natural speech with specified emotions.

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
The full-duplex benchmark primarily encompasses two major categories: **interruption** and **rejection**.

The **interruption** category consists of five sub-scenarios:

1. **Ask** – The user poses a follow-up question based on the model’s previous response, interrupting the ongoing output. The model should promptly address the new query.
2. **Deny** – The user expresses dissatisfaction or disagreement with the model’s response using negative statements, interrupting the model mid-sentence.
3. **Repeat** – The user interrupts to request a repetition of the model’s previous response due to inaudibility or misunderstanding.
4. **Shift** – The user initiates a new topic, interrupting the model’s current answer.
5. **Wait** – The user asks the model to stop speaking. The model should immediately cease its response and express readiness to resume the dialogue when prompted.

The **rejection** category includes four sub-scenarios:

1. **Backchannel** – During the model’s response, the user may produce short interjections such as “uh-huh” or “yeah.” The model should correctly ignore these backchannels.
2. **Pause** – The user may pause mid-sentence due to thinking or hesitation, leading to incomplete semantics. The model should wait until the user’s intent is complete before responding.
3. **Others Talk to User(Background Speech)** – The model must recognize and reject speech from other speakers or background noise, ensuring interaction only with the true user.
4. **Talk to others** – The user may suddenly turn to converse with another person during interaction. The model should detect and reject such utterances appropriately.

### Evaluation Metrics

For evaluation, we largely follow [Full-Duplex-Bench v1.5](https://github.com/DanielLin94144/Full-Duplex-Bench), while introducing additional metrics to further assess full-duplex capability. 

For **interruption** scenarios, we evaluate the response rate (corresponding to the **RESPOND** score in [Full-Duplex-Bench v1.5](https://github.com/DanielLin94144/Full-Duplex-Bench)), as well as two latency metrics in [Full-Duplex-Bench v1.5](https://github.com/DanielLin94144/Full-Duplex-Bench) — the **stop latency** (how quickly the model halts its current response upon interruption) and the **response latency** (how quickly it begins responding to the new query). 

For **rejection** scenarios, we measure the rejection rate (corresponding to the **RESUME** score in [Full-Duplex-Bench v1.5](https://github.com/DanielLin94144/Full-Duplex-Bench)) and the **early interrupt rate**, assessing the model’s ability to correctly ignore backchannels, incomplete utterances caused by pauses, background or external speech, and conversations directed at others. Additionally, we introduce **first response delay** to evaluate the overall responsiveness of the model.

### Dataset
1. Train Set

Our training set covers both interruption and rejection scenarios, comprising over 107 hours of real human recordings in both Chinese and English, featuring more than 100 speakers. The data structure is as follows:

train/<br>
&ensp;&ensp;zh/<br>
&ensp;&ensp;&ensp;&ensp;ask/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.TextGrid/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.wav/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;.../<br>
&ensp;&ensp;&ensp;&ensp;backchannel/<br>
&ensp;&ensp;&ensp;&ensp;deny/<br>
&ensp;&ensp;&ensp;&ensp;others_talk_to_user(background speech)/<br>
&ensp;&ensp;&ensp;&ensp;pause/<br>
&ensp;&ensp;&ensp;&ensp;repeat/<br>
&ensp;&ensp;&ensp;&ensp;shift/<br>
&ensp;&ensp;&ensp;&ensp;wait/<br>
&ensp;&ensp;en/<br>
&ensp;&ensp;&ensp;&ensp;ask/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.TextGrid/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.wav/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;.../<br>
&ensp;&ensp;&ensp;&ensp;backchannel/<br>
&ensp;&ensp;&ensp;&ensp;deny/<br>
&ensp;&ensp;&ensp;&ensp;others_talk_to_user(background speech)/<br>
&ensp;&ensp;&ensp;&ensp;pause/<br>
&ensp;&ensp;&ensp;&ensp;repeat/<br>
&ensp;&ensp;&ensp;&ensp;shift/<br>
&ensp;&ensp;&ensp;&ensp;wait/<br>

2. Dev Set

We release a development set covering two major scenarios—**interruption** and **rejection**, each consisting of nine sub-tasks. Each sub-task contains 200 test samples (100 in Chinese and 100 in English). The data structure is as follows:

dev/<br>
&ensp;&ensp;zh/<br>
&ensp;&ensp;&ensp;&ensp;ask/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx_sentence.json/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.TextGrid/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.wav/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;.../<br>
&ensp;&ensp;&ensp;&ensp;backchannel/<br>
&ensp;&ensp;&ensp;&ensp;deny/<br>
&ensp;&ensp;&ensp;&ensp;others_talk_to_user(background speech)/<br>
&ensp;&ensp;&ensp;&ensp;pause/<br>
&ensp;&ensp;&ensp;&ensp;repeat/<br>
&ensp;&ensp;&ensp;&ensp;shift/<br>
&ensp;&ensp;&ensp;&ensp;talk_to_others/<br>
&ensp;&ensp;&ensp;&ensp;wait/<br>
&ensp;&ensp;en/<br>
&ensp;&ensp;&ensp;&ensp;ask/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx_sentence.json/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.TextGrid/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;xxxx.wav/<br>
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;.../<br>
&ensp;&ensp;&ensp;&ensp;backchannel/<br>
&ensp;&ensp;&ensp;&ensp;deny/<br>
&ensp;&ensp;&ensp;&ensp;others_talk_to_user(background speech)/<br>
&ensp;&ensp;&ensp;&ensp;pause/<br>
&ensp;&ensp;&ensp;&ensp;repeat/<br>
&ensp;&ensp;&ensp;&ensp;shift/<br>
&ensp;&ensp;&ensp;&ensp;talk_to_others/<br>
&ensp;&ensp;&ensp;&ensp;wait/<br>

You can download it via [Google Drive](https://drive.google.com/drive/folders/1mXjQi_uPPDhwhbvxKsMCqNMtm89ab6Zn?usp=sharing). If that's not convenient, you can use the [123 Cloud](https://www.123912.com/s/QlDejv-h7anA) for downloading.

### Baseline
The competition provides a baseline system built upon [Easy Turn](https://github.com/ASLP-lab/Easy-Turn) and [OSUM-EChat](https://github.com/ASLP-lab/OSUM).This baseline serves as a reproducible and extensible starting point, helping participants better benchmark their systems and ensuring fair comparison across different approaches.

We enable [OSUM-EChat](https://github.com/ASLP-lab/OSUM) with full-duplex capability by integrating it with [Easy Turn](https://github.com/ASLP-lab/Easy-Turn). For our baseline, we fine-tune the Easy Turn model using only the training set. You can refer to Easy Turn to generate data in the required format for the baseline and then perform fine-tuning.

