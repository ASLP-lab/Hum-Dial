# ICASSP2026 HumDial Challenge

This is the official GitHub repository for the ICASSP2026 HumDial Challenge

## Challenge Call 

Have you been following the recent buzz around the impressive performance of next-generation voice dialogue models like GPT-4o, Doubao, and the newly released GPT-Realtime? They are not only lightning-fast and expressive but also enable seamless multimodal interactions, making conversations feel remarkably human.

From the traditional “clunky AI” to today’s “AI assistant,” the evolution of voice dialogue systems has been nothing short of astonishing. **But just how far are we from achieving truly “natural human-machine dialogue”?** While current voice models excel in technical metrics, they still lack a certain “human touch.” They may recognize single emotions like “happiness” or “sadness,” but struggle to truly understand the complexity of our emotional changes or empathize with our situations. They may engage in fluent one-on-one exchanges, yet become flustered in real-world interaction scenarios such as interruptions, overlapping speech, or group chats. This is the “uncanny valley” that current voice dialogue systems struggle to cross.

To break through this bottleneck and advance technology toward truly “human-like” interaction, a coalition of institutions—including **Northwestern Polytechnical University, Nanjing University, The Chinese University of Hong Kong, Huawei Technologies Co., Ltd., and AISHELL**—has jointly launched the HumDial (Human-like Spoken Dialogue Systems) Challenge! We believe a truly intelligent dialogue system must not only “understand clearly, reason logically, and express coherently” but also possess the ability to interact seamlessly with humans in real, emotionally complex environments.

The inaugural HumDial2026 Challenge will be held at ICASSP 2026, a premier conference for speech research, and will focus on two core challenges:
- **Emotional Intelligence:** Moving beyond simplistic emotion labeling, this track will test a model's ability to accurately understand context-dependent emotions, provide empathetic responses, conduct in-depth reasoning, and dynamically track emotional shifts—empowering AI to truly understand and connect with users.
- **Full-Duplex Interaction:** Breaking free from rigid turn-based exchanges, this track will evaluate a system's ability to handle interruptions, overlapping speech, real-time feedback, and natural conversational rhythms, helping AI learn to communicate more naturally.

We will not only introduce brand-new evaluation dimensions but also release exclusive, finely annotated datasets of real-world scenarios for each track. If you’re passionate about “human-like” dialogue systems and eager to shape the future of next-generation voice interaction, we welcome you to follow and register for the challenge! Let’s work together to turn AI into a warm, emotionally aware communication partner.

## Dataset

### **Track 1: Emotional Intelligence**  

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

task1: 1-turn dialogues, judging users' emotional status, not participating in evaluation

task2: Contains 3, 4, and 5-turn dialogues, where in the final turn users ask the model about their own emotional changes

task3: Contains 3, 4, and 5-turn dialogues, where in the final turn users ask the model about the underlying reasons for emotions


2. dev set

We release a development set, including Task 1, Task 2, Task 3, and Task 4 (selected from Task 3 and Task 4). The data structure is as follows:

dev/<br>
&ensp;&ensp;zh/<br>
&ensp;&ensp;&ensp;&ensp;task1/<br>
&ensp;&ensp;&ensp;&ensp;task2/<br>
&ensp;&ensp;&ensp;&ensp;task3/<br>
&ensp;&ensp;&ensp;&ensp;task4/<br>
&ensp;&ensp;&ensp;&ensp;task1.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task4.jsonl<br>
&ensp;&ensp;en/<br>
&ensp;&ensp;&ensp;&ensp;task1/<br>
&ensp;&ensp;&ensp;&ensp;task2/<br>
&ensp;&ensp;&ensp;&ensp;task3/<br>
&ensp;&ensp;&ensp;&ensp;task4/<br>
&ensp;&ensp;&ensp;&ensp;task1.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task2.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task3.jsonl<br>
&ensp;&ensp;&ensp;&ensp;task4.jsonl<br>

Task 1: 1-turn dialogues, judging users' emotional status, not participating in ranking calculation

Task 2: Contains 3, 4, and 5-turn dialogues, used to evaluate the model's response text score

Task 3: Contains 3, 4, and 5-turn dialogues, used to evaluate the model's response text score

Task 4: Contains 3, 4, and 5-turn dialogues, used to evaluate the model's response audio score

You can download it via [Google Drive](https://drive.google.com/file/d/1f9muDtrvEoVZDrel3HN4p1lyRjKt_EP9/view?usp=sharing).

### **Track 2: Full-Duplex Interaction**  
We will provide multi-turn Chinese and English dialogue data from real recordings, covering typical scenarios such as speech interruptions and recognition rejection. Accompanied by strict annotations, this dataset will be used to comprehensively evaluate participating systems in three core aspects: response speed, behavioral rationality, and linguistic naturalness.

- The dataset is designed to cover the core scenarios of emotional intelligence and full-duplex interaction, ensuring diversity and authenticity to comprehensively evaluate the performance of participating models. It includes dialogue scenes in both Chinese and English, covering a wide range of emotional and conversational contexts. 
- For each task in the challenge, we will provide a dedicated set of real-world recorded speech data to serve as the train set and test set. These datasets are collected from natural, human-human or human-machine interactions to ensure authenticity and cover diverse scenarios aligned with the respective tasks.
- In addition, we will release the complete data generation pipeline, enabling participants to reproduce or extend the synthetic dataset if desired. Participants are also free to use any publicly available speech or text datasets to train or fine-tune their models, provided they do not use any private or unauthorized data sources.
- All participants are strictly prohibited from using any part of the official test set for training or parameter tuning. The use of test labels or any test data leakage will result in disqualification.

## Baseline

### **Emotional Intelligence:**

The competition provides a baseline system built upon [OpenS2S](https://github.com/CASIA-LM/OpenS2S).This baseline serves as a reproducible and extensible starting point, helping participants better benchmark their systems and ensuring fair comparison across different approaches.

You can generate the data in the required baseline format by running [get_token.py](Emotional Intelligence/get_token.py), and then refer to OpenS2S for fine-tuning.

### **Full-Duplex Interaction**

......

## Evaluation

The Emotional_Intelligence folder provides evaluation prompts for Task 2, Task 3, and Task 4. The evaluation model used is [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).

## Guidelines for participants
1. **External Resource Usage**
   - **Permitted Scope**: For both Track I and Track II, participants are allowed to use external datasets and pre-trained models, including but not limited to speech foundation models and large language models (LLMs).
   - **Openness Requirement**: Any external resources used must be freely available to all research communities.
   - **Declaration Obligation**: Participants must clearly and completely list all external resources used and their sources in the final system report.

2. **Dataset Usage Guidelines**
   - **Training Data**: Participants may use the officially provided training subset, or any publicly available open-source datasets. When using any non-official datasets, such as synthetic datasets, the source must be clearly indicated and explicitly stated in the final system report.
   - **Development Data**: The development set may only be used for model performance evaluation and debugging.
   - **Evaluation Data**: Any form of unauthorized use of evaluation set data is strictly prohibited. This includes, but is not limited to, using evaluation data for any form of model training, fine-tuning, or parameter tuning. Violation will result in disqualification.
   - For more detailed information about datasets, please refer to the dataset description page.

3. **Model and Inference Requirements**
   Participants are free to choose any pre-trained model or modeling technique, but the final submitted model must meet the following two mandatory conditions:
   - **Offline Inference**: The inference process must run independently without an internet connection.
   - **Hardware Limitation**: The entire inference process must be able to run in a server environment with a single GPU with no more than 48 GB of memory.

4. **Submission Requirements**
   - **Deliverables**: Participants must submit a complete set of system deliverables to ensure reproducibility of results.
   - **Submission Content**: Expected to include final results, model files, and a Docker image that supports one-click inference execution.
   - **Detailed Guidelines**: Specific submission specifications and operational procedures will be provided after the baseline implementation is released.

5. **Final Interpretation**
   The right of final interpretation of the competition rules belongs to the organizers. In special circumstances, the organizers will coordinate and decide according to the specific situation.