### 多层感知机

#### 1. 多层感知机的基本知识

##### 1.1 隐藏层

隐藏层依然是一个线性层，其表达式依然是一个线性模型

![Image Name](https://cdn.kesci.com/upload/image/q5ho684jmh.png)

##### 1.2 表达公式



我们先来看一种含单隐藏层的多层感知机的设计。其输出$\boldsymbol{O} \in \mathbb{R}^{n \times q}$的计算为

$$
\begin{aligned} \boldsymbol{H} &= \boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h,\\ \boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o, \end{aligned}
$$


也就是将隐藏层的输出直接作为输出层的输入。如果将以上两个式子联立起来，可以得到


$$
 \boldsymbol{O} = (\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h)\boldsymbol{W}_o + \boldsymbol{b}_o = \boldsymbol{X} \boldsymbol{W}_h\boldsymbol{W}_o + \boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o. 
$$


从联立后的式子可以看出，虽然神经网络引入了隐藏层，却依然等价于一个单层神经网络：其中输出层权重参数为$\boldsymbol{W}_h\boldsymbol{W}_o$，偏差参数为$\boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o$。不难发现，即便再添加更多的隐藏层，以上设计依然只能与仅含输出层的单层神经网络等价。

##### 1.3 激活函数

由1.2的结论得出，这样的模型仍然是一个单层神经网络，全连接层只是对数据进行仿射变换，而多个仿射变换的叠加依然是一个仿射变换，所以为了引入非线性变换，本身即为非线性函数的激活函数出现了

###### 1.3.1 ReLU函数

ReLU 函数(rectified linear unit)函数提供了一个很简单的非线性变换。给定元素$x$，该函数定义如下。可以看出，ReLU 函数只保留正数元素，并将负数元素清零。


$$
\text{ReLU}(x) = \max(x, 0).
$$

<img src=".\PIcs\ReLU1.png" alt="ReLU1" style="zoom:80%;" /><img src=".\PIcs\ReLU2.png" alt="ReLU2" style="zoom:80%;" />

###### 1.3.2 Sigmoid函数

sigmoid函数可以将元素的值变换到0和1之间：
$$
\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.
$$
依据链式法则，sigmoid函数的导数


$$
\text{sigmoid}'(x) = \text{sigmoid}(x)\left(1-\text{sigmoid}(x)\right).
$$


下面绘制了sigmoid函数的导数。当输入为0时，sigmoid函数的导数达到最大值0.25；当输入越偏离0时，sigmoid函数的导数越接近0。

<img src=".\PIcs\Sigmoid1.png" alt="Sigmoid1" style="zoom:80%;" /><img src=".\PIcs\Sigmoid2.png" alt="Sigmoid2" style="zoom:80%;" />



###### 1.3.3 Tanh函数

tanh（双曲正切）函数可以将元素的值变换到-1和1之间。我们接着绘制tanh函数。当输入接近0时，tanh函数接近线性变换。虽然该函数的形状和sigmoid函数的形状很像，但tanh函数在坐标系的原点上对称。
$$
\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.
$$

<img src=".\PIcs\tanh1.png" alt="Tanh1" style="zoom:80%;" /><img src=".\PIcs\tanh2.png" alt="Tanh2" style="zoom:80%;" />