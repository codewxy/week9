训练模型地址：https://www.tinymind.com/code-wxy/week9

数据集地址：https://www.tinymind.com/code-wxy/datasets/week9



日志截图:



# 心得体会

本次作业做完之后感觉难度并不是很大，因为在整个过程中，最开始的数据准备阶段是需要自己根据参考代码进行修改，修改过程以及运行数据生成的过程中，按照下面的步骤进行，基本是没什么错误。其中自己需要制作trainval.txt文件等。在虚拟机中生成了训练和验证数据之后，把文件下载到本地用于上传到tinymind的数据集。

后面的上传代码和运行过程，遇到了些路径没写对的问题，还有部分tinymind的问题，数据集无法上传，数据集无法识别等。

总之由于不需要自己根据论文复现写代码。按照步骤会轻松很多。



以下是整理的一些资料的步骤。

## 一、数据准备

### 1.安装相应的库

```shell
安装Protobuf、python3-pil、python3-lxml、python3-tk等
sudo apt-get install protobuf-compiler python3-pil python3-lxml python3-tk

安装相应的库
pip install Cython
pip install contextlib2
pip install jupyter
pip install matplotlib

依赖的库
pip install Cython
pip install contextlib2
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
```

### 2.编译Protobuf

Tensorflow对象检测API使用Protobufs配置模型和训练参数。 在使用框架之前，Protobuf库必须编译。 这应该通过运行以下命令来完成.(在object_detection的当前目录运行)

```shell
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

### 3.修改数据生成代码

./object_detection/dataset_tools/create_data.py.

修改此文件，由于主要使用的是research/object_detection目录下的物体检测框架的代码。这个框架同时引用slim框架的部分内容，需要对运行路径做一下设置，不然会出现找不到module的错误 。

修改的方式我采用的是直接在代码中插入路径，使用**sys.path.insert** 或者**sys.path.append**。

具体的用法：

```
python程序中使用 import XXX 时，python解析器会在当前目录、已安装和第三方模块中搜索 xxx，如果都搜索不到就会报错。使用sys.path.append()方法可以临时添加搜索路径，方便更简洁的import其他包和模块。这种方法导入的路径会在python程序退出后失效。

1. 加入上层目录和绝对路径
import sys
sys.path.append('..') #表示导入当前文件的上层目录到搜索路径中
sys.path.append('/home/model') # 绝对路径
from folderA.folderB.fileA import functionA

2. 加入当前目录
import os,sys
sys.path.append(os.getcwd()) #os.getcwd()用于获取当前工作目录，而非py代码文件的当前目录

3. 定义搜索优先顺序
sys.path.insert(1, "./crnn")定义搜索路径的优先顺序，序号从0开始，表示最大优先级，sys.path.insert()加入的也是临时搜索路径，程序退出后失效。

import sys
sys.path.insert(1, "./model")
```

###  4.编码数据准备代码

```
python3 object_detection/dataset_tools/create_data.py --label_map_path=./data/labels_items.txt --data_dir=./data/ --output_dir=./data/out 

(shell脚本中等号后不要留空格)
```

需要建立out目录。执行完后，会在/data/out下面生成两个.record文件。

- pet_train.record
- pet_val.record



### 5.问题及解决

#### 5.1 运行时报如下警告：

```
/home/wxy/ai/week9/object_detection/utils/dataset_util.py:75: FutureWarning: The behavior of this method will change in future versions. Use specific 'len(elem)' or 'elem is not None' test instead.
```

![1531927614218](C:\Users\WXY\AppData\Local\Temp\1531927614218.png)

解决方法：

忽略警告。已经得到结果了。

#### 5.2 未找到文件trainval.txt

![1531929205430](C:\Users\WXY\AppData\Local\Temp\1531929205430.png)

解决办法：

根据代码中使用方式，推断出文件的格式。自己根据文件名，制作包含155个文件名的文件。

## 二、编辑pipline.config文件

以models/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config为基础进行修改。本文档所在仓库也提供了这个文件。 这个文件里面放了训练和验证过程的所有参数的配置，包括各种路径，各种训练参数（学习率，decay，batch_size等）。有这个文件，命令行上面可以少写很多参数，避免命令行内容太多。

注意要点：

- num_classes， 原文件里面为37,这里的数据集为5
- num_examples， 这个是验证集中有多少数量的图片，请根据图片数量和数据准备脚本中的生成规则自行计算。
- PATH_TO_BE_CONFIGURED，这个是原文件中预留的字段，一共5个，分别包含预训练模型的位置，训练集数据和label_map文件位置，验证集数据和label_map文件位置。这个字段需要将数据以及配置文件等上传到tinymind之后才能确定路径的具体位置（把PATH_TO_BE_CONFIGURED替换成了/data/code-wxy/week9这个路径）。
- num_steps，这个是训练多少step，后面的训练启动脚本会用到这个字段，直接将原始的200000改成0.注意不要添加或者删除空格等，后面的训练启动脚本使用sed对这个字段进行检测替换，如果改的有问题会影像训练启动脚本的执行。不通过run.sh本地运行需要将这个数字改成一个合适的step数，改成0的话会有问题。
- max_evals，这个是验证每次跑几轮，这里直接改成1即可，即每个训练验证循环只跑一次验证。
- eval_input_reader 里面的shuffle， 这个是跟eval步骤的数据reader有关，如果不使用GPU进行训练的话，这里需要从false改成true，不然会导致错误，详细内容参阅 <https://github.com/tensorflow/models/issues/1936>

> 训练和验证过程次数相关的参数，后面在训练启动脚本中会自动进行处理，这里不需要过多关注，但是实际使用的时候，需要对这些参数进行合适的设置，比如**num_steps**参数，后面的训练启动脚本中，每轮运行100个step，同时根据数据集图片总数all_images_count和batch_size的大小，可以计算出epoch的数量，最后输出模型的质量与epoch的数量密切相关。epoch=num_step*batch_size/all_images_count。具体的计算留给学员自己进行。

> config文件需要跟代码一起上传，运行的时候会先被复制到output文件夹里面。

## 三、下载预训练模型

由于本模型采用的是mobilenet模型ssd检测框架，在如下的文档中<https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md> 

下载了[ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) 。

在本地解压后，将如下三个文件放入数据集文件中备用

model.ckpt.data-00000-of-00001

model.ckpt.index

model.ckpt.meta



## 四、上传文件以及数据开始训练

