### 基于词嵌入法（word embedding）的分类方法代码运行方法

作者是王骁，学号2020K8009922005

##文件功能:
基于词嵌入法训练一个模型用于进行文本分类预测，在这里并没有使用gensim word2vec进行预训练。


##文件内容
包括对训练集与测试集进行文本预处理，基于训练集训练模型，并对测试集进行预测，预测结果生成csv文件。
在这里利用keras构建了一个序贯模型，其结构按顺序如下
  输入层
  词嵌入层（Embedding），将离散的词汇转换为连续的向量
  一维卷积层（Conv1D）和最大池化层（MaxPooling1D）
  2个全连接层（Dense）


##注意事项：
使用者需要对训练集，测试集和生成的csv文件的位置进行相应的修改

该文件是在anaconda中编写运行的

部分库需要专门下载部分库，正常来说按提示下载缺少的库即可

代码源文件为`WordEmbedding_wangxiao.py`，可以在anaconda上下载相应库后即可使用对应‘run’按钮直接运行

为了训练模型所需的硬件资源，本代码直接使用CPU进行训练，可根据CPU线程数进行相应的设置

训练完毕并用训练得到的模型，会储存在‘model.h5’，最后调用该模型将预测的结果储存在‘try0.csv’中


##参考文献：
https://blog.csdn.net/kobeyu652453/article/details/107228172
https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794