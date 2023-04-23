# 项目说明
## 一、环境说明

1、nvcc版本：

`nvcc: NVIDIA (R) Cuda compiler driver`
`Copyright (c) 2005-2019 NVIDIA Corporation`
`Built on Sun_Jul_28_19:07:16_PDT_2019`
`Cuda compilation tools, release 10.1, V10.1.243`

2、python版本：Python 3.8.10

3、cuda版本：cuda 10.1

4、pytorch版本： 1.6.0

5、具体环境需求，请看"`能跑通的环境.txt`"和`requirements.txt`两个文件说明

### 其他依赖 
- pytorch=1.6 
- transformers=3.1.0
- pytorch-lightning=1.0.6
- spacy=3.0 # conflicts with transformers
- pytorch-struct=0.4 


## 二、训练说明
1、训练前，请在链接“`https://huggingface.co/fnlp/bart-base-chinese/tree/main`”里下载`pytorch_model.bin`文件并放到`gen-arg-Chinese/fnlp/bart-base-chinese`路径下，这是中文bart模型，需要自己下载

2、训练时请调用`gen-arg-Chinese/scripts/train_military.sh`脚本进行训练，可以在`gen-arg-Chinese`目录下用命令：`bash scripts/train_military.sh`进行训练。

3、调用时可能会报错，xxx文件夹已经存在等问题，如果出现，请根据提示找到对应的文件夹处理即可。或者训练前用命令`rm -rf preprocessed_data/`以及`rm -rf checkpoints/gen-MilitaryExercise/`把两个文件夹删除（他们都是每次训练会生成的文件，所以需要删除上一次生成的文件才能进行新的一轮训练）

4、综合上面两点，可以在`gen-arg-Chinese`目录下用一条命令完成上面的1和2两步：`rm -rf preprocessed_data/ && rm -rf checkpoints/gen-MilitaryExercise/ && bash scripts/train_military.sh`

5、如果需要修改训练epoch数、学习率、batchsize等参数，可以在`scripts/train_military.sh`中进行修改，更具体的可以自行修改代码

## 三、测试说明
1、训练完成后，会在`gen-arg-Chinese/checkpoints/gen-MilitaryExercise/`文件夹里生成几个`.ckpt`模型检查点文件

2、注意可以在`gen-arg-Chinese/scripts/test_military.sh`中修改调用的检查点文件名称（`.ckpt`文件）。还有其他参数也可以根据需求自行更改

3、测试命令是：`rm -rf checkpoints/gen-MilitaryExercise-pred/ && bash scripts/test_military.sh`

4、上面其实是两条命令，可以看到是删除了`checkpoints/gen-MilitaryExercise-pred/`文件夹，因为这是每次测试生成的结果文件存储的地方

## 四、调用训练好的模型应用
1、我也提供了一个预训练好的模型检查点，存储在`checkpoints/`文件夹下的文件：`epoch=2.ckpt`

2、可以通过`bash main.sh`或者`bash load_model_gen.sh`调用预训练好的模型对单条文本进行抽取，他们分别会调用`main.py`和`load_model_gen.py`文件。这两个`py`文件是类似的功能，`main.py`是模型和无监督方法结合后的调用，`load_model_gen.py`则是只有模型的结果

3、可以根据需求更改脚本`main.sh`或者`load_model_gen.sh`中的模型检查点路径`CHECKPOINT_PATH`来实现对不同检查点的加载，例如改成自己新训练的检查点的路径


## 五、文件夹说明
1、`checkpoints`是存储模型训练的检查点，以及测试生成的预测结果

2、`data`文件夹，存储数据集的文件夹，里面的`MilitaryExercise`是本次军演经过处理后的，符合模型输入格式的数据集。`data/MilitaryExercise/data/`里的数据来自于主目录`dataset/data`里

3、`fnlp/bart-base-chinese`是中文bart模型，由于使用的pytorch版本和transformers版本较低，不能通过联网方式下载该预训练模型，并且`fnlp/bart-base-chinese/config.json`也是经过我修改后才能在这个环境中运行（将`max_position_embeddings`的值从512改为了513）

4、`logs`是日志

5、`predicted`是我存储的一些预测结果，是手动复制过来的，程序生成的预测结果不存在这里，程序生成的结果在`checkpoints/gen-MilitaryExercise-pred/`

6、`preprocessed_data`文件存的是`src/genie/Military_data_module.py`代码处理`data`中的数据后的结果

7、`scripts`是训练和测试文本，对应的是不同数据集，军演的数据集是名字带有`military`的脚本

8、`src/genie`是模型用到的源文件，里面有`model.py`、`network.py`等，是对模型结构的定义等

9、`unsupercode`则是小组里其他负责的无监督方法源代码

## 六、文件说明

1、`aida_ontology_cleaned.csv`是定义模型使用的模板

2、`gen-arg-Chinese/data/MilitaryExercise/scorer/event_role_multiplicities.txt`是模型用到的实体定义。定义了事件类型、参数角色以及参数个数

3、请注意，如果修改了上述的模板，则`event_role_multiplicities.txt`也要对应修改

4、如果修改了模板，那么`gen-arg-Chinese/data/MilitaryExercise/data`里的训练和测试数据也需要更改，因为里面的数据是含有模板里定义的参数角色名称的。至于如何修改这些数据，可以根据数据集的格式要求，自己写代码进行处理。我也会提供一个参考代码。参考代码不在`gen-arg-Chinese`里，请看和`gen-arg-Chinese`同级目录下的`dataset`文件夹

## 七、数据集格式要求

1、格式是和`gen-arg-Chinese/data/RAMS_1.0`英文数据集一样的，详细可以看`gen-arg-Chinese/data/RAMS_1.0/README.md`里的说明，以及观察提供的数据集进行参考

2、这里简单汇总说明一下：
* `ent_spans`: Start and end (inclusive) indices and an event/argument/role string.一个参数或角色的开始，结束位置，以及角色。
        示例：`[[position1, position2, [["evt089arg02role", 1.0]]], []]`

* `evt_triggers`: Start and end (inclusive) indices and an event type string.
        示例：同`ent_pans`，不过是触发词的

* `sentences`: Document text
        示例：`[["", "", ""], ["", "", ""]]`，其中""里是一个字

* `gold_evt_links`: A triple of (event, argument, role) following the above format
        示例：`[[[pos1, pos2], [pos3, pos4], "evt01arg01"], []]`，实际上就是<event, argument, role>三元组，event实际是evt_triggers里的位置`[pos1, pos2]`，argument就是参数的位置`[pos3, pos4]`，role就是角色的实体名称

* `source_url`: source of the text，即原文链接

* `split`: Which data split it belongs to. 属于哪个划分：train, test, dev三种的一种

* `doc_key`: Which individual file it corresponds to (nw_ is prepended on all of them). 每个文本的文档id



