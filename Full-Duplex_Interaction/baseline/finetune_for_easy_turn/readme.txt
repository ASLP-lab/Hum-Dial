（1）对于baseline设置，我们使用训练数据finetune Easy Turn这个单独的、即插即用的对话轮次检测模块，然后结合OSUM-EChat，使其具备全双工能力。
（2）相关Easy Turn的训练代码可参考https://github.com/ASLP-lab/Easy-Turn
（3）参赛者也可以对OSUM-EChat进行finetue，可参考https://github.com/ASLP-lab/OSUM/tree/main/OSUM-EChat
（4）我们在finetune Easy Turn时，使用的prompt与原始Easy Turn的prompt有所不同（详见prompt.yaml）。
（5）我们使用竞赛的训练集用于finetune Easy Turn构建baseline的流程：对于用户的首次问题和用户的打断截取出来，给定<complete>标签；在拒识场景User Real-time Backchannels中，对于用户的简短附和，给定<backchannel>标签；在拒识场景Pause Handling中，我们截取[break]标签之前的内容，给定<incomplete>标签。
（5）原本Easy Turn的wait状态对应于我们竞赛的打断场景的Silence/Termination子场景。然后对于我们竞赛而言，在打断场景中，面对用户的打断，模型应该及时回应。因此我们在finetune时专门去掉了wait状态，将训练集中类似于wait状态的语句给定<complete>标签。
（6）我们曾尝试加入其他状态，比如背景音标签<background>，但是效果不好而且还会影响Easy Turn原本几种状态的效果。因此，我们使用训练集进行finetune，进一步优化了Easy Turn原本几种状态的效果。