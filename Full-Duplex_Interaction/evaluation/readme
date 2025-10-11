（1）将打断（Interruption）场景的五个子场景的RESPOND分数取平均得到总分（满分为100，所以rate要记得先乘以100）。
（2）将拒识场景（Rejection）的四个子场景的RESUME分数（满分为100，所以rate要记得先乘以100）（注意场景Pause_Handling和Third-party_Speech_before使用了拒识率（rejection_rate），但表达的含义是一致的，取百分号前面的数字）取平均得到总分（满分为100）。
（3）总延迟：打断场景的两个延迟（stop latency和 response latency），打断和拒识场景的模型的首次回复延迟（first_response_delay），三者取平均值。
（4）对于总延迟的计算，打断的两个延迟（stop latency和 response latency），打断共五个小场景，每个场景都要计算，最后取平均值；模型的首次回复延迟（first_response_delay）是打断和拒识共九个场景都要计算，最后取平均值。
（5）最终的RESPOND和RESUME分数保留小数点后两位，总延迟保留小数点后三位。