import os
import sys
sys.path.insert(0, '../../../../../')
import re

from gxl_ai_utils.utils import utils_file

import jieba
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def get_tag_from_str(input_str):
    # 使用正则表达式提取所有 <> 中的内容
    matches = re.findall(r'<.*?>', input_str)
    # 输出结果
    if len(matches)==0:
        return '<--no_tag-->'
    return matches[0].upper()
def convert_to_nearest_multiple(numbers, step=0.3):
    # 使用numpy的四舍五入功能将每个数字转换为最近的step的倍数
    return [round(num / step) * step for num in numbers]

def get_full_tag_from_str(input_str, smooth_interval=0.1):
    matches = re.findall(r'<.*?>', input_str)
    # 输出结果
    # if len(matches) == 0:
    #     return '<--no_tag-->'
    num_list = []
    for match in matches:
        try:
            # 尝试将匹配到的字符串转换为浮动数
            num_list.append(float(match[1:-1]))  # 去掉字符串的首尾字符后转换为 float
        except ValueError:
            # 如果转换失败，则将 -1 添加到 num_list
            num_list.append(-1)
    num_list = convert_to_nearest_multiple(num_list, step=smooth_interval)
    res_str = " ".join([f"{num:.6f}" for num in num_list])
    return res_str
def do_convert_to_pure_tag(input_file, output_file):
    text_dict = utils_file.load_dict_from_scp(input_file)
    new_text_dict = {}
    num_blank_tag = 0
    for key , value in text_dict.items():
        res_value = get_full_tag_from_str(value, smooth_interval=0.2)
        new_text_dict[key] =res_value
        if len(res_value) == 0:
            num_blank_tag += 1
    utils_file.logging_warning('空白时间戳的数量为：', num_blank_tag)
    output_file_blank_num = output_file + '_blank_num'
    utils_file.write_list_to_file([f'空白帧数量为{num_blank_tag}'],output_file_blank_num)
    utils_file.write_dict_to_scp(new_text_dict, output_file)
    return num_blank_tag


def do_showing_confusion_matrix(labels, matrix,title='', output_fig_path: str=None):
    """
    可视化混淆矩阵的函数
    :param labels: 标签序列，类型为列表等可迭代对象
    :param matrix: 代表混淆矩阵的二维列表，元素为整数，形状应为(len(labels), len(labels))
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # 设置中文字体为黑体，以支持中文显示（确保系统中已安装黑体字体）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号显示问题（有时候会出现负号显示异常，这一步是为了保证显示正常）
    plt.rcParams['axes.unicode_minus'] = False
    num_classes = len(labels)
    fig, ax = plt.subplots()
    # 使用imshow来绘制热力图展示混淆矩阵
    im = ax.imshow(np.array(matrix), cmap=plt.cm.Blues)

    # 设置x轴和y轴的刻度以及对应的标签
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 旋转x轴刻度标签，让其更美观显示
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个方格中的数值标签
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, matrix[i][j],
                           ha="center", va="center", color="black")

    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()
    if output_fig_path is not None:
        plt.savefig(output_fig_path)
    else:
        plt.show()

def calculate_bleu(candidate_text: str, reference_texts: list) -> float:
    """
    计算中文对话任务的 BLEU 分数。

    参数:
        candidate_text (str): 生成的文本。
        reference_texts (list): 参考文本列表（可以有多个参考句子）。

    返回:
        float: BLEU 分数。
    """
    # 对生成文本和参考文本进行分词
    candidate_tokens = list(jieba.cut(candidate_text))
    reference_tokens = [list(jieba.cut(ref)) for ref in reference_texts]

    # 使用 nltk 的 sentence_bleu 计算 BLEU 分数
    smoothing = SmoothingFunction().method1  # 避免分数为零
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens,
                               weights=(0.25, 0.25, 0.25, 0.25),  # 1-gram 到 4-gram 的权重
                               smoothing_function=smoothing)

    return bleu_score

def do_compute_acc(ref_file, hyp_file, output_dir, output_dir2):
    """
    计算正确率和错误率，并列出混淆矩阵
    :return:
    """
    utils_file.makedir_sil(output_dir)
    utils_file.makedir_sil(output_dir2)
    utils_file.do_compute_wer(ref_file, hyp_file, output_dir)
    utils_file.logging_info('字符错误率计算完毕')
    wer_all = utils_file.do_get_wer_from_wer_file4all(os.path.join(output_dir, 'wer'))
    utils_file.copy_file(ref_file, os.path.join(output_dir2, 'ref_text'), use_shell=True)
    utils_file.copy_file(hyp_file, os.path.join(output_dir2, 'hyp_text'), use_shell=True)
    utils_file.copy_file(os.path.join(output_dir,'wer'), os.path.join(output_dir2, 'wer'))


    utils_file.logging_info('开始计算正确率，acc')
    acc_path = os.path.join(output_dir, 'acc')
    hyp_dict = utils_file.load_dict_from_scp(hyp_file)
    ref_dict = utils_file.load_dict_from_scp(ref_file)
    tag_hyp_dict = {}
    for key, value in hyp_dict.items():
        tag_hyp_dict[key] = get_tag_from_str(value)
    tag_ref_dict = {}
    for key, value in ref_dict.items():
        tag_ref_dict[key] = get_tag_from_str(value)
    output_acc_f = open(acc_path, 'w', encoding='utf-8')
    same_num = 0
    all_tags = set()
    # utils_file.print_dict(utils_file.do_get_random_subdict(tag_ref_dict, 10))
    for tag in tag_ref_dict.values():
        all_tags.add(tag)
    # for tag in tag_hyp_dict.values():
    #     all_tags.add(tag)
    labels = sorted(list(all_tags))
    # style_convert_dict = {
    #     "<新闻科普>": "<xinwei_kepu>",
    #     "<恐怖故事>": "<kongbu_gushi>",
    #     "<童话故事>": "<tonghua_gushi>",
    #     "<客服>": "<kefu>",
    #     "<诗歌散文>": "<shige_sanwen>",
    #     "<有声书>": "<youshengshu>",
    #     "<日常口语>": "<richang_kouyu>",
    #     "<其他>": "<qita>",
    # }

    utils_file.logging_info(f'标签种类为：{labels}')
    # labels = [style_convert_dict.get(label, label) for label in labels]
    utils_file.logging_info(f'标签种类为：{labels}')
    num_classes = len(labels)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for key, hyp_tags in tag_hyp_dict.items():
        if key not in tag_ref_dict:
            continue
        ref_tags = tag_ref_dict[key]
        # 判断标签是否相同
        if ref_tags == hyp_tags:
            if_same = True
            same_num += 1
            index = labels.index(ref_tags)
            matrix[index][index] += 1
        else:
            if_same = False
            # 标签不同的情况，找到真实标签和预测标签对应的索引，在相应的位置加1
            ref_index = labels.index(ref_tags)
            if hyp_tags in labels:
                hyp_index = labels.index(hyp_tags)
                matrix[ref_index][hyp_index] += 1
        # 向文件写入数据
        output_acc_f.write(f"key: {key}\n")
        output_acc_f.write(f"ref_tag: {ref_tags}\n")
        output_acc_f.write(f"hyp_tag: {hyp_tags}\n")
        output_acc_f.write(f"if_same: {if_same}\n")
        output_acc_f.write("\n")  # 添加空行分隔不同的条目
    acc_num = same_num / len(tag_hyp_dict)
    output_acc_f.write(f'正确率为：{acc_num}')
    output_acc_f.flush()
    output_acc_f.close()
    utils_file.copy_file(acc_path, os.path.join(output_dir2, 'acc'), use_shell=True)
    figure_path = os.path.join(output_dir, 'confusion_matrix.png')
    do_showing_confusion_matrix(labels, matrix, output_fig_path=figure_path)
    utils_file.copy_file(figure_path, os.path.join(output_dir2, 'confusion_matrix.png'))
    return {
        'acc': acc_num,
        'wer': wer_all,
    }

def do_compute_align(ref_file, hyp_file, output_dir, output_dir2):
    utils_file.makedir_sil(output_dir)
    utils_file.makedir_sil(output_dir2)
    utils_file.logging_info(f'开始计算的wer')
    utils_file.do_compute_wer(ref_file, hyp_file, output_dir)
    utils_file.logging_info('字符错误率计算完毕')
    wer_all = utils_file.do_get_wer_from_wer_file4all(os.path.join(output_dir, 'wer'))
    utils_file.copy_file(ref_file, os.path.join(output_dir2, 'ref_text'), use_shell=True)
    utils_file.copy_file(hyp_file, os.path.join(output_dir2, 'hyp_text'), use_shell=True)
    utils_file.copy_file(os.path.join(output_dir,'wer'), os.path.join(output_dir2, 'wer'))


    utils_file.logging_info(f'开始计算纯标签的wer_smooth')
    wer_pure_tag_smooth_path = os.path.join(output_dir, 'wer_pure_tag_smooth')
    hyp_file_tmp = utils_file.do_get_fake_file()
    ref_file_tmp = utils_file.do_get_fake_file()
    blank_num_hyp =  do_convert_to_pure_tag(hyp_file, hyp_file_tmp)
    blank_num_ref = do_convert_to_pure_tag(ref_file, ref_file_tmp)
    output_dir_tmp = '/home/xlgeng/.cache/'
    utils_file.do_compute_wer(ref_file_tmp, hyp_file_tmp, output_dir_tmp)
    utils_file.copy_file(os.path.join(output_dir_tmp, 'wer'), wer_pure_tag_smooth_path)
    utils_file.copy_file(wer_pure_tag_smooth_path, os.path.join(output_dir2, 'wer_pure_tag_smooth'))
    wer_smooth = utils_file.do_get_wer_from_wer_file4all(os.path.join(output_dir, 'wer_pure_tag_smooth'))
    utils_file.logging_info('纯标签的wer_smooth计算完毕')
    return {
        'wer': wer_all,
        'wer_smooth': wer_smooth,
        'blank_num_hyp': blank_num_hyp,
    }

from collections import defaultdict
import re
import edit_distance


class AverageShiftCalculator():
    def __init__(self):
        print("Calculating average shift.")

    def __call__(self, refs, hyps):
        ts_list1 = self.read_timestamps(refs)
        ts_list2 = self.read_timestamps(hyps)
        # print(ts_list1)
        # print(ts_list2)
        res = self.as_cal(ts_list1, ts_list2)
        print("Average shift : {}.".format(str(res)[:8]))
        print("Following timestamp pair differs most: {}, detail:{}".format(self.max_shift, self.max_shift_uttid))
        return res

    def _intersection(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        if set1 == set2:
            print("Uttid same checked.")
            return set1
        itsc = list(set1 & set2)
        print("Uttid differs: file1 {}, file2 {}, lines same {}.".format(len(list1), len(list2), len(itsc)))
        return itsc

    def read_timestamps(self, body_list):
        ts_list = []
        pattern_error = 0
        for body in body_list:
            body = body.replace("<|startoftranscript|>", "").replace("<|transcribe|>", "")
            ts_pattern = r"<\d{1,2}\.\d+>"
            if "<|en|>" in body:
                body = body.replace("<|en|>", "")
                lan = "en"
            elif "<|zh|>" in body:
                body = body.replace("<|zh|>", "")
                lan = "zh"

            all_time_stamps = re.findall(ts_pattern, body)
            all_time_stamps = [float(t.replace("<", "").replace(">", "")) for t in all_time_stamps]
            all_word_list = [x for x in re.split(ts_pattern, body)][1:-1]
            all_word_list = [x.strip() for x in all_word_list if x.strip() != '']

            if len(all_time_stamps) != 2 * len(all_word_list):
                pattern_error += 1
                continue
            text = "\t".join(all_word_list)
            ts = [all_time_stamps[i:i + 2] for i in range(0, len(all_time_stamps) - 1, 2)]
            ts_list.append((text, ts))
            assert len(ts) == len(all_word_list), f"{body}"
        print(f"pattern_error_num: {pattern_error}")
        return ts_list

    def _shift(self, filtered_timestamp_list1, filtered_timestamp_list2):
        shift_time = 0
        for fts1, fts2 in zip(filtered_timestamp_list1, filtered_timestamp_list2):
            shift_time += abs(fts1[0] - fts2[0]) + abs(fts1[1] - fts2[1])
        num_tokens = len(filtered_timestamp_list1)
        return shift_time, num_tokens

    def as_cal(self, ts_list1, ts_list2):
        # calculate average shift between timestamp1 and timestamp2
        # when characters differ, use edit distance alignment
        # and calculate the error between the same characters
        assert len(ts_list1) == len(ts_list2), f"{len(ts_list1)}, {len(ts_list2)}"
        self._accumlated_shift = 0
        self._accumlated_tokens = 0
        self.max_shift = 0
        self.max_shift_uttid = None
        for uttid in range(len(ts_list1)):
            (t1, ts1) = ts_list1[uttid]
            (t2, ts2) = ts_list2[uttid]
            # print(t1)
            # print(ts1)
            # print(t2)
            # print(ts2)
            # input()
            _align, _align2, _align3 = [], [], []
            fts1, fts2 = [], []
            _t1, _t2 = [], []
            sm = edit_distance.SequenceMatcher(t1.split('\t'), t2.split('\t'))
            s = sm.get_opcodes()
            # print("sm1",s)
            for j in range(len(s)):
                if s[j][0] == "replace" or s[j][0] == "insert":
                    _align.append(0)
                if s[j][0] == "replace" or s[j][0] == "delete":
                    _align3.append(0)
                elif s[j][0] == "equal":
                    _align.append(1)
                    _align3.append(1)
                else:
                    continue
            # use s to index t2
            for a, ts, t in zip(_align, ts2, t2.split('\t')):
                if a:
                    fts2.append(ts)
                    _t2.append(t)
            sm2 = edit_distance.SequenceMatcher(t2.split('\t'), t1.split('\t'))
            s = sm2.get_opcodes()
            for j in range(len(s)):
                if s[j][0] == "replace" or s[j][0] == "insert":
                    _align2.append(0)
                elif s[j][0] == "equal":
                    _align2.append(1)
                else:
                    continue
            # use s2 tp index t1
            for a, ts, t in zip(_align3, ts1, t1.split('\t')):
                if a:
                    fts1.append(ts)
                    _t1.append(t)
            if len(fts1) == len(fts2):
                shift_time, num_tokens = self._shift(fts1, fts2)
                if num_tokens == 0:
                    # print(ts_list1[uttid], ts_list2[uttid])
                    continue
                self._accumlated_shift += shift_time
                self._accumlated_tokens += num_tokens
                if shift_time / num_tokens > self.max_shift:
                    self.max_shift = shift_time / num_tokens
                    self.max_shift_uttid = uttid
            else:
                print("length mismatch")
        return self._accumlated_shift / self._accumlated_tokens

def _do_format_for_acc(input_file, output_file):
    pattern = re.compile(r'<\d+\.\d+>|[^\s<>]+')
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split(maxsplit=1)  # 分离ID和文本部分
            # print(parts)
            if len(parts) > 1:
                id_part = parts[0]
                text_part = parts[1]
                tokens = pattern.findall(text_part)

                # 将文本部分按照每3个字符插入空格
                try:
                    spaced_text = ' '.join(
                        [f"{tokens[i]}{tokens[i + 1]}{tokens[i + 2]}" for i in range(0, len(tokens), 3)])
                    # spaced_text = ' '.join([text_part[i:i+3] for i in range(0, len(text_part), 3)])
                    # 写入新文件
                    outfile.write(f'{id_part} {spaced_text}\n')
                except Exception as e:
                    print(f"跳过处理失败的行: {line.strip()}\n异常信息: {e}")
    print("格式化完成，输出文件为:", output_file)

def do_compute_align2(ref_file, hyp_file, output_dir, output_dir2):
    utils_file.makedir_sil(output_dir)
    utils_file.makedir_sil(output_dir2)
    utils_file.logging_info(f'开始计算的wer')
    utils_file.do_compute_wer(ref_file, hyp_file, output_dir)
    utils_file.logging_info('字符错误率计算完毕')
    wer_all = utils_file.do_get_wer_from_wer_file4all(os.path.join(output_dir, 'wer'))
    utils_file.copy_file(ref_file, os.path.join(output_dir2, 'ref_text'), use_shell=True)
    utils_file.copy_file(hyp_file, os.path.join(output_dir2, 'hyp_text'), use_shell=True)
    utils_file.copy_file(os.path.join(output_dir,'wer'), os.path.join(output_dir2, 'wer'))


    utils_file.logging_info(f'开始计算纯标签的ass指标')
    hyp_file_tmp = utils_file.do_get_fake_file()
    blank_num_hyp =  do_convert_to_pure_tag(hyp_file, hyp_file_tmp)

    output_famat_file_hyp = utils_file.do_get_fake_file()
    print('hyp_file',hyp_file)
    _do_format_for_acc(hyp_file, output_famat_file_hyp)
    print('ref_file',ref_file)
    output_famat_file_ref = utils_file.do_get_fake_file()
    _do_format_for_acc(ref_file, output_famat_file_ref)

    datas = defaultdict(list)
    refs = []
    hyps = []
    print(f'output_famat_file_ref,{output_famat_file_ref}')
    ref_keys = set()
    with open(output_famat_file_ref, 'r') as ref_timestamp:
        for line in ref_timestamp:
            tmp = line.strip().split(maxsplit=1)
            if tmp[0] not in ref_keys:
                ref_keys.add(tmp[0])
                datas[tmp[0]].append(tmp[1])
            else:
                continue
    print(f'output_famat_file_hyp,{output_famat_file_hyp}')
    hyp_keys = set()
    with open(output_famat_file_hyp, 'r') as hyp_timestamp:
        for line in hyp_timestamp:
            tmp = line.strip().split(maxsplit=1)
            if tmp[0] not in hyp_keys:
                hyp_keys.add(tmp[0])
                datas[tmp[0]].append(tmp[1])
            else:
                continue

    print('len datas ',len(datas))
    for utt, tmp in datas.items():
        if len(tmp) == 2:
            refs.append(tmp[0])
            hyps.append(tmp[1])


    # Create an instance of AverageShiftCalculator
    calculator = AverageShiftCalculator()

    # Call the calculator with the sample input data
    print('refs lens:', len(refs))
    aas_score = calculator(refs, hyps)
    print("aas_score", aas_score, '\n')


    utils_file.logging_info('ass计算完毕')
    return {
        'wer': wer_all,
        'ass': aas_score,
        'blank_num_hyp': blank_num_hyp,
    }



def do_compute_bleu_for_chat(ref_file, hyp_file, output_dir, output_dir2):
    # 计算BLEU得分
    utils_file.makedir_sil(output_dir)
    utils_file.makedir_sil(output_dir2)
    utils_file.copy_file(ref_file, os.path.join(output_dir2, 'ref_text'), use_shell=True)
    utils_file.copy_file(hyp_file, os.path.join(output_dir2, 'hyp_text'), use_shell=True)
    ref_dict = utils_file.load_dict_from_scp(ref_file)
    hyp_dict = utils_file.load_dict_from_scp(hyp_file)
    res_dict = {}
    total_bleu_score = 0
    for key, value in tqdm.tqdm(hyp_dict.items(), total=len(hyp_dict)):
        if key in ref_dict:
            ref_str = ref_dict[key]
            hyp_str = value
            bleu_score = calculate_bleu(hyp_str, [ref_str])
            res_dict[key] = bleu_score
            total_bleu_score += bleu_score
        else:
            print(f"key {key} not in ref_dict")
            continue

    avg_bleu_score = total_bleu_score / len(res_dict)
    res_dict['avg_bleu_score'] = avg_bleu_score
    utils_file.write_dict_to_scp(res_dict, os.path.join(output_dir, 'bleu'))
    utils_file.write_dict_to_scp(res_dict, os.path.join(output_dir2, 'bleu'))
    return {
        'avg_bleu_score': avg_bleu_score,
    }

if __name__ == '__main__':
    # 示例用法
    labels = ['class1', 'class2', 'class3', 'class4', 'class5']
    matrix = [[10, 2, 1, 0, 1],
              [3, 12, 1, 1, 0],
              [0, 0, 8, 0, 0],
              [1, 1, 0, 9, 0],
              [0, 0, 0, 1, 10]]
    do_showing_confusion_matrix(labels, matrix,title='age', output_fig_path='./output.png')