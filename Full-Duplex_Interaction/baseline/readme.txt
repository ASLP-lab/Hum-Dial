（1）环境配置：
conda create -n easy_turn_osum_echat python=3.12
conda activate easy_turn_osum_echat
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
（2）Baseline模型部署前，请先到Hugging Face（）下载所需模型，保存到ckpt文件夹下（所需模型参考run_demo_flask.sh）
（3）Baseline模型部署：（需要至少两张4090及以上显卡）
cd ./baseline
./run_demo_flask.sh
（4）Baseline模型批量推理：
cd ./baseline
python ./web_flask/inference.py --server_ip "127.0.0.1:5000" --input_list ./data/Follow-up_Questions.list
（ip：127.0.0.1; 端口号默认5000; input_list指的是输入一个list文件，每一行为一条需要推理的音频的路径）
（5）./web_flask文件夹下的代码是主要的部署代码（chat_stream.py, modules.py等），实现部署的大部分逻辑（主要在Freeze-omni部署框架的基础上进行了改动）。
（6）./finetune_for_easy_turn文件夹下记录了finetune Easy Turn的相关内容。
（7）./wenet文件夹下是OSUM-EChat的相关代码，./wenet_interupt文件夹下是Easy Turn的相关代码。