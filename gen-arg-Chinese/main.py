# -*- coding: utf-8 -*-
import sys
sys.path.append('unsupercode/')
import time

from load_model_gen import EventExtract
from load_model_gen import convert_output2list

if __name__ == '__main__':
    title = '美国防部：将与北约部分成员国在地中海举行大规模军演'
    content = '美国国防部当地时间1月21日宣布，美国与北约部分成员国将于24日在地中海举行“海王星22号打击”大规模海军演习。\r\n据悉，美国将派出“杜鲁门”号航空母舰及其航母打击群和空军联队参与演习。美国防部发言人称，此次演习将持续至2月4日。（央视记者 刘骁骞）'
    public_time = '2022-01-22'
    input_data = {}
    input_data['title'] = title
    input_data['content'] = content
    input_data['public_time'] = public_time
    tStart = time.time()
    output = EventExtract(content)
    supervised_result = convert_output2list(output)
    print('\n格式化输出：\n', supervised_result)
    
    



