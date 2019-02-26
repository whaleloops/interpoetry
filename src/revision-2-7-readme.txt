修改内容

在evaluator类中 
增加了函数 
1.init_bpe
2.eval_mono
3.get_blank_acc
在 evaluator的init函数中调用init_bpe初始化了bpe

调用eval_mono后输出结果
在dump路径下生成 hyp（epoch.{lang}.{datatype}.txt
记录每个evaluation instance的结果
每行分成两部分 前一部分为被blank的诗句 后一部分为预测结果 中间以'\t###\t'分割
e.g. 我[blank]你大爷    ### 我爱你大爷

scores['acc_%s_%s' % (lang, data_type)]中记录了accuracy
