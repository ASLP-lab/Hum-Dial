Third-party Speech有两个子场景：after 和 before (即他人话语位于用户的真实询问之后或者之前)
Third-party Speech的after场景：使用eval.py会得到一个RESUME分数（保留三位小数，如果结果为0.52，则可以写成0.520）。在使用eval.py前，请确保使用evaluation/get_transcript下的相应infer文件生成了所需的所有json文件
Third-party Speech的before场景: 使用compute_rejection_rate_before.py会得到一个rejection_rate（保留三位小数，比如0.425，即42.5%）。在使用compute_rejection_rate_before.py前，请确保使用evaluation/get_transcript下的相应infer文件生成了所需的所有json文件。
最后，将上面的得到的RESUME分数和rejection_rate两者加权（权重0.5和0.5，因为after和before条数比例为1：1，保留三位小数），得到最终的拒识场景Third-party Speech的分数。
