---
tags:
- text2text-generation
- Chinese
- seq2seq
- BART
language: zh
---

# Chinese BART-Base

## 说明
可以从huggingface中找到这个模型的源文件，链接：https://huggingface.co/fnlp/bart-base-chinese/tree/main

## Model description

This is an implementation of Chinese BART-Base.

[**CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation**](https://arxiv.org/pdf/2109.05729.pdf)

Yunfan Shao, Zhichao Geng, Yitao Liu, Junqi Dai, Fei Yang, Li Zhe, Hujun Bao, Xipeng Qiu

**Github Link:** https://github.com/fastnlp/CPT


## Usage

```python
>>> from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
>>> tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
>>> model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
>>> text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
>>> text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=False)
    [{'generated_text': '北 京 是 中 国 的 首 都'}]
```

**Note: Please use BertTokenizer for the model vocabulary. DO NOT use original BartTokenizer.**

## Citation

```bibtex
@article{shao2021cpt,
  title={CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation}, 
  author={Yunfan Shao and Zhichao Geng and Yitao Liu and Junqi Dai and Fei Yang and Li Zhe and Hujun Bao and Xipeng Qiu},
  journal={arXiv preprint arXiv:2109.05729},
  year={2021}
}
```