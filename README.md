# 说明
基于bart的中文事件元素抽取，军事演习领域。训练好的模型在checkpoints里


文件夹`gen-arg-Chinese`是事件抽取模型的主代码
- 基于bart模型，训练时，请到“`https://huggingface.co/fnlp/bart-base-chinese/tree/main`”里下载`pytorch_model.bin`文件并放到`gen-arg-Chinese/fnlp/bart-base-chinese`路径下，这是中文bart模型，需要自己下载

# 具体的说明
运行main.sh脚本调用训练好的模型进行中文事件元素的抽取
请看文件夹里的README.md文件说明