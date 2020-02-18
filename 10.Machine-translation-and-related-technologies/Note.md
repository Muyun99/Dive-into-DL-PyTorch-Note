### 机器翻译和数据集

机器翻译（MT）：将一段文本从一种语言自动翻译为另一种语言，用神经网络解决这个问题通常称为神经机器翻译（NMT）。
主要特征：输出是单词序列而不是单个单词。 输出序列的长度可能与源序列的长度不同。

#### 1. 数据预处理

##### 1.1 数据清洗及分词

建立字符串---单词组成的列表

```python
source = [['go', '.'], ['hi', '.'], ['hi', '.']],
target = [['va', '!'], ['salut', '!'], ['salut', '.']]
```

##### 1.2 建立词典

建立单词组成的列表---单词id组成的列表

#### 2. Encoder-Decoder 简介

- encoder：输入到隐藏状态  
- decoder：隐藏状态到输出

![Image Name](https://cdn.kesci.com/upload/image/q5jcat3c8m.png?imageView2/0/w/640/h/640)

#### 3. Sequence to Sequence (seq2seq)模型简介

##### 3.1 模型：
- 训练  
  ![Image Name](https://cdn.kesci.com/upload/image/q5jc7a53pt.png?imageView2/0/w/640/h/640)
- 预测

![Image Name](https://cdn.kesci.com/upload/image/q5jcecxcba.png?imageView2/0/w/640/h/640)



##### 3.2 具体结构：
![Image Name](https://cdn.kesci.com/upload/image/q5jccjhkii.png?imageView2/0/w/500/h/500)

#### 4. Beam Search

##### 4.1 简单greedy search：

![Image Name](https://cdn.kesci.com/upload/image/q5jchqoppn.png?imageView2/0/w/440/h/440)

维特比算法：选择整体分数最高的句子（搜索空间太大）

##### 4.2 集束搜索：

![Image Name](https://cdn.kesci.com/upload/image/q5jcia86z1.png?imageView2/0/w/640/h/640)