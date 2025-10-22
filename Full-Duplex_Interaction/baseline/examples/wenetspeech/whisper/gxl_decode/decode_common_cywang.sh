# 通用推理脚本
# cd ..
. ./path.sh || exit 1;

data_type= # raw  shard_full_data
test_sets=
test_data_dir=
gpu_id=
dir=
ckpt_name=  # 需要带pt
task=no_set_task # transcribe  sot_task  emotion_task
lang=  # en zh

english_sets='ami librispeech_clean librispeech_other'
train_config=conf/config_llm_huawei_instruct-version_cosyvoice1-token.yaml
average_checkpoint=false
average_num=5
average_mode=step
decode_modes="llmasr_decode"
decoding_chunk_size=-1
ctc_weight=0
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
batch_size=12
prompt_file=conf/prompt_config.yaml

. tools/parse_options.sh || exit 1;
# 将所有传进来的元素打印出来
echo "传入的参数为:"
echo 'data_type is '$data_type
echo 'test_sets is '$test_sets
echo 'test_data_dir is '$test_data_dir
echo 'gpu_id is '$gpu_id
echo 'dir is '$dir
echo 'ckpt_name is '$ckpt_name
echo 'task is '$task
echo 'lang is '$lang
echo 'print over'
echo 'prompt_file'$prompt_file



test_sets="${test_sets//---/ }"
echo "待推理的数据集为:"$test_sets
decode_checkpoint=$dir/$ckpt_name
export CUDA_VISIBLE_DEVICES="$gpu_id"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

if [ ${average_checkpoint} == true ]; then
  decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}.pt
  echo "do model average and final checkpoint is $decode_checkpoint"
  python wenet/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path $dir  \
    --num ${average_num} \
    --mode ${average_mode} \
    --max_step ${max_step} \
    --val_best
fi


# 定义函数，根据传入的testset值返回对应的task
function get_task_description {
    local input="$1"
    case "$input" in
        "age")
            echo "<TRANSCRIBE><AGE>"
            ;;
        "align")
            echo "<TRANSCRIBE><ALIGN>"
            ;;
        "gender")
            echo "<TRANSCRIBE><GENDER>"
            ;;
        "caption")
            echo "<TRANSCRIBE><CAPTION>"
            ;;
        "chat")
            echo "<TRANSCRIBE><S2TCHAT>"
            ;;
        "emotion")
            echo "<TRANSCRIBE><EMOTION>"
            ;;
        "style")
            echo "<TRANSCRIBE><STYLE>"
            ;;
        "speech_token")
            echo "<TRANSCRIBE><SPEECH_TOKEN>"
            ;;
        "public_test/aishell1_gender")
            echo "<TRANSCRIBE><GENDER>"
            ;;
        "public_test/kaggle_age")
            echo "<TRANSCRIBE><AGE>"
            ;;
        "public_test/kaggle_gender")
            echo "<TRANSCRIBE><GENDER>"
            ;;
        "public_test/MELD_test")
            echo "<TRANSCRIBE><EMOTION>"
            ;;
        "public_test/MER23_test")
            echo "<TRANSCRIBE><EMOTION>"
            ;;
        "public_test/roobo_100")
            echo "<TRANSCRIBE><ALIGN>"
            ;;
        "public_test/AirBench_speech")
            echo "<TRANSCRIBE><S2TCHAT>"
            ;;
        "caption_0107_esc50")
            echo "<TRANSCRIBE><CAPTION>"
            ;;
        "caption_0107_vocalsound")
            echo "<TRANSCRIBE><CAPTION>"
            ;;
        "caption_aslp_record")
            echo "<TRANSCRIBE><CAPTION>"
            ;;
        "3500_chat")
            echo "<SPEECH2TEXT_SPEECH_TOKEN>"
            ;;
        "3500_asr") 
            echo "<TEXT2SPEECH_TOKEN>"
            ;;
        "aslp_chat_test")
            echo "<TRANSCRIBE><S2TCHAT>"
            ;;
        "aslp_chat_test_for_asr")
            echo "<TRANSCRIBE>"
            ;;
        "aligin_english_clean")
            echo "<TRANSCRIBE><ALIGN>"
            ;;
        "align_cn_noize")
            echo "<TRANSCRIBE><ALIGN>"
            ;;
        "align_en_noize")
            echo "<TRANSCRIBE><ALIGN>"
            ;;
        "dialect_test_new_add_2025_3_5")
            echo "<TRANSCRIBE><DIALECT>"
            ;;
        "punctuation_test_add_2025_3_5")
            echo "<TRANSCRIBE><PUNCTUATION>"
            ;;
        "Ali_near_test")
            echo "<TRANSCRIBE><SPEAKER>"
            ;;
        "b9")
            echo "<TRANSCRIBE><AGE>"
            ;;
        "b9")
            echo "<TRANSCRIBE><AGE>"
            ;;
        "b9")
            echo "<TRANSCRIBE><AGE>"
            ;;
        *)
            echo "no_set_task"
            ;;
    esac
}
#
#test_sets="age align gender caption chat emotion style speech_token aishell"
#for testset in ${test_sets}; do
#    task=$(get_task_description "$testset")
#    if [ "$task" == "no_set_task" ]; then
#        task="<TRANSCRIBE>"
#    fi
#
#    echo "task is $task; test set is $testset"
#done
#exit -1


# 假设test_sets是一个包含测试集名称的变量或者数组，以下是示例的赋值（实际中替换为真实内容）
#test_sets="age align sex gender caption chat emotion style speech_token"
for testset in ${test_sets}; do
    task=$(get_task_description "$testset")
    if [ "$task" == "no_set_task" ]; then
        task="<TRANSCRIBE>"
    else
        echo "Task is: $task, 匹配成功"
    fi

    echo "task is $task; test set is $testset"


  # shellcheck disable=SC2193
  if [[ "${data_type}" == "raw" ]]; then
    echo "data_type 为 raw"
    test_data="$test_data_dir/$testset/data.list"
  else
    echo "data_type 为 shard_full_data"
    test_data="$test_data_dir/$testset/shards_list.txt"
  fi
  echo "test_data is $test_data"

  if [[ " ${english_sets} " =~ " ${testset} " ]]; then
    lang="en"
  else
    lang="zh"
  fi
  echo "lang is $lang"
  base=$(basename $decode_checkpoint)
  result_dir=$dir/test_${base}/$testset
  echo "result_dir is $result_dir"
  mkdir -p $result_dir
  echo "哈哈哈哈我是耿雪龙"
  python ./runtime/infer_with_torchair_cyw.py --gpu ${gpu_id} \
    --modes $decode_modes \
    --config $train_config \
    --data_type $data_type \
    --test_data $test_data \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size $batch_size \
    --blank_penalty ${blank_penalty} \
    --length_penalty ${length_penalty} \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $result_dir \
    --task $task \
    --lang $lang \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

  echo "哈哈啊啊啊啊"  

  mkdir -p ${result_dir}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      $test_data_dir/$testset/text $result_dir/$mode/text > $result_dir/$mode/wer
  done
done

