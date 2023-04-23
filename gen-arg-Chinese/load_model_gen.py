import argparse 
from src.genie.model import GenIEModel 
import torch
import re
from transformers import BertTokenizer
import datetime

MAX_LENGTH=512

# 加载实体，包括模板、参数角色，事件类型等
def load_ontology():
        # read ontology 
        ontology_dict ={} 
        with open('aida_ontology_cleaned.csv','r') as f:
            for lidx, line in enumerate(f):
                if lidx == 0:# header 
                    continue 
                fields = line.strip().split(',') 
                if len(fields) < 2:
                    break 
                evt_type = fields[0]
                args = fields[2:]
                
                ontology_dict[evt_type] = {
                        'template': fields[1]
                    }
                
                for i, arg in enumerate(args):
                    if arg !='':
                        ontology_dict[evt_type]['arg{}'.format(i+1)] = arg 
                        ontology_dict[evt_type][arg] = 'arg{}'.format(i+1)
        
        return ontology_dict 

def convert_output2list(model_output_str):
    # predicted = {
    # 'doc_key':[],
    # '参与国家':[],
    # '地点':[],
    # '开始时间':[],
    # '结束时间':[],
    # '演习代号':[],
    # '参与人数':[]
    # }
    predicted = {}
    keys = [
    # 'doc_key',
    '参与国家',
    '地点',
    '开始时间',
    '结束时间',
    '演习代号',
    '参与人数'
    ]
    # 根据模板不同需要设计不同的pattern从output_str中获取参数
    patterns = [r'(.*?)等国家和组织', r'在(.*?)等地区',r'从(.*?)开始',r'开始到(.*?)共',r'代号为(.*?)的军事演习',r'共(.*?)参与'] 

    index = 0
    for key in keys:
        if key == 'doc_key':
            # index += 1
            continue
        tmp = re.findall(patterns[index], model_output_str)
        if tmp != [] and tmp[0] == '<arg>':
            # predicted[key].append('')
            predicted[key] = ''
        elif tmp != []:
            # predicted[key].append(tmp[0])
            predicted[key] = tmp[0]
        else:
            # predicted[key].append('')
            predicted[key] = ''
        index += 1
    return predicted
    
    
def EventExtract(context_str):  
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        default='gen',
        # 给的checkpoints是训练gen模型得到的，如果要用constrained-gen模型，需要在训练时指定训练constrained-gen模型
        choices=['gen','constrained-gen'] 
    )
    parser.add_argument(
        "--load_ckpt", # 模型checkpoints的路径
        default=None,
        type=str, 
    )
    parser.add_argument(
        "--eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation." # 一次处理多少个文本，默认是一次处理一条
    )
    parser.add_argument(
        "--eval_only", action="store_true", # 表明是调用checkpoints而不是训练
    )
    
    args = parser.parse_args()
    
    time_start = datetime.datetime.now()
    
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义模型类对象，以及加载checkpoints
    model = GenIEModel(args).to(device)
    model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict'])
    
    # 加载中文bart的tokenizer
    tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
    tokenizer.add_tokens(['<arg>','<tgr>']) # 加token，目的是保证模板里的<arg>不会被分割开
    
    ontology_dict = load_ontology() 
    template = ontology_dict['military.exercise']['template'] # 加载模板
    
    input_template = re.sub(r'<arg\d[\d]?>', '<arg>', template) # 将模板中的<arg4>等换成<arg>
    space_tokenized_input_template = input_template.split(' ') # 将模板中的文字按空格划分，定义的模板每个字要用空格割开

    time0 = datetime.datetime.now()
    
    tokenized_input_template = [] 
    for w in space_tokenized_input_template:
        tokenized_input_template.extend(tokenizer.tokenize(w)) # 将模板进行tokenize
    
    context = tokenizer.tokenize(context_str) # 将输入文本进行tokenize
    
    # 将模板和文本进行拼接，然后一起encode
    input_tokens = tokenizer.encode_plus(tokenized_input_template, context, 
                                add_special_tokens=True,
                                add_prefix_space=True,
                                max_length=MAX_LENGTH,
                                truncation='only_second',
                                padding='max_length')
    
    # 获取模型的结果生成器
    generator = model.getSelfModel()
    # 将list转化为tensor
    input = torch.tensor([input_tokens['input_ids']]).to(device)
    
    # 生成抽取文本
    with torch.no_grad():
        sample_output = generator.generate(input, do_sample=False, 
                                    max_length=100, num_return_sequences=1,num_beams=1,
                                )
    sample_output = sample_output.reshape(input.size(0), 1, -1)
    
    # decode成文本
    output = tokenizer.decode(sample_output[0].squeeze(0), skip_special_tokens=True)
    time1 = datetime.datetime.now()
    output = re.sub(' ', '', output) # 去掉空格
    
    time2 = datetime.datetime.now()
    print('模板：\n', template)
    print('\n输入文本：\n', context_str)
    print('\n输出结果：\n', output)
    durTime = '%dms' % ((time0 - time_start ).seconds * 1000 + (time0 -time_start ).microseconds / 1000)
    print('\n加载模型时间：' + str(durTime))
    durTime = '%dms' % ((time1 - time0 ).seconds * 1000 + (time1 -time0 ).microseconds / 1000)
    print('\n推理时间：' + str(durTime))
    durTime = '%dms' % ((time2 - time_start ).seconds * 1000 + (time2 -time_start ).microseconds / 1000)
    print('\n总耗时：' + str(durTime))
    return output

if __name__ == '__main__':
    context_str = "美国国防部当地时间1月21日宣布，美国与北约部分成员国将于24日在地中海举行“海王星22号打击”大规模海军演习。\r\n据悉，美国将派出“杜鲁门”号航空母舰及其航母打击群和空军联队参与演习。美国防部发言人称，此次演习将持续至2月4日。（央视记者 刘骁骞）"
    output = EventExtract(context_str)
    predicted = convert_output2list(output)
    print('\n格式化输出：\n', predicted)