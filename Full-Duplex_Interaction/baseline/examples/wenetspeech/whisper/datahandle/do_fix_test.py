# style
import re

global_style_dict = {
    "朗读":"新闻科普",
    "科普百科":"新闻科普",
    "悬疑恐怖":"恐怖故事",
    "童话故事":"童话故事",
    "客服":"客服",
    "诗歌":"诗歌散文",
    "散文":"诗歌散文",
    "武侠评书":"有声书",
    "小说":"有声书",
    "历史":"有声书",
    "科幻":"有声书",
    "对话":"日常口语",
    "口语":"日常口语",
    "幽默":"其他",
    "其他":"其他",
}
global_age_dict = {
    "MIDDLE_AGE":"ADULT",
    "YOUTH":"ADULT",
    "CHILD":"CHILD",
    "OLD":"OLD",
}
def replace_keys_in_brackets(input_str, key_value_dict):
    for key, value in key_value_dict.items():
        # 构造匹配 <key> 形式的正则表达式模式
        pattern = re.compile(r'<{}>'.format(key))
        input_str = pattern.sub(f"<{value}>", input_str)
    return input_str
from gxl_ai_utils.utils import utils_file
if __name__ == '__main__':
    input_str = "哈哈<幽默>fsdfs发噶大哥dss<口语>vfad<嘻嘻嘻>"
    print(replace_keys_in_brackets(input_str,global_style_dict))
    input_text_path = "/mnt/sfs/asr/test_data/test_sets_format_3000/age/text"
    utils_file.copy_file(input_text_path, input_text_path+"_old")
    text_dict =utils_file.load_dict_from_scp(input_text_path)
    new_dict = {}
    for key, value in text_dict.items():
        new_dict[key] = replace_keys_in_brackets(value,global_age_dict)
    utils_file.write_dict_to_scp(new_dict, input_text_path)
    input_text_path = "/mnt/sfs/asr/test_data/test_sets_format_3000/style/text"
    utils_file.copy_file(input_text_path, input_text_path + "_old")
    text_dict = utils_file.load_dict_from_scp(input_text_path)
    new_dict = {}
    for key, value in text_dict.items():
        new_dict[key] = replace_keys_in_brackets(value, global_style_dict)
    utils_file.write_dict_to_scp(new_dict, input_text_path)
