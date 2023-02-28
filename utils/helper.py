import tqdm
from collections import defaultdict
import torch
import json
from pytorch_pretrained_bert import BertModel, BertTokenizer
import jieba
from jieba.analyse.tfidf import TFIDF
from jieba.posseg import POSTokenizer
import sys
import re 
import glob
sys.path.append('../')

from model_entity.hanlp_ner import HanlpNER
from utils.preprocessing import split_to_subsents, split_to_sents
from pathlib import Path

def get_entities_by_third_party_tool():
    hanlpner = HanlpNER()
    entities_by_third_party_tool = defaultdict(list)
    for file in tqdm.tqdm(list(glob.glob('data/FR2KG/financial_research_reports/*.txt'))):
        with open(file, encoding='utf-8') as f:
            sents = [[]]
            cur_sent_len = 0
            for line in f:
                for sent in split_to_subsents(line):
                    sent = sent[:hanlpner.max_sent_len]
                    if cur_sent_len + len(sent) > hanlpner.max_sent_len:
                        sents.append([sent])
                        cur_sent_len = len(sent)
                    else:
                        sents[-1].append(sent)
                        cur_sent_len += len(sent)
            sents = [''.join(_) for _ in sents]
            sents = [_ for _ in sents if _]
            for sent in sents:
                entities_dict = hanlpner.recognize(sent)
                for ent_type, ents in entities_dict.items():
                    entities_by_third_party_tool[ent_type] += ents

    for ent_type, ents in entities_by_third_party_tool.items():
        entities_by_third_party_tool[ent_type] = list([ent for ent in set(ents) if len(ent) > 1])
    return entities_by_third_party_tool

def get_entities_by_rules():
    entities_by_rule = defaultdict(list)
    for file in tqdm.tqdm(list(glob.glob('data/FR2KG/financial_research_reports/*.txt'))):
        with open(file, encoding='utf-8') as f:
            found_yanbao = False
            found_fengxian = False
            for lidx, line in enumerate(f):
                # 公司的标题
                ret = re.findall('^[\(（]*[\d一二三四五六七八九十①②③④⑤]*[\)）\.\s]*(.*有限公司)$', line)
                if ret:
                    entities_by_rule['机构'].append(ret[0])

                # 研报
                if not found_yanbao and lidx <= 5 and len(line) > 10:
                    may_be_yanbao = line.strip()
                    if not re.findall(r'\d{4}\s*[年-]\s*\d{1,2}\s*[月-]\s*\d{1,2}\s*日?', may_be_yanbao) \
                            and not re.findall('^[\d一二三四五六七八九十]+\s*[\.、]\s*.*$', may_be_yanbao) \
                            and not re.findall('[\(（]\d+\.*[A-Z]*[\)）]', may_be_yanbao) \
                            and len(may_be_yanbao) > 5 \
                            and len(may_be_yanbao) < 100:
                        entities_by_rule['研报'].append(may_be_yanbao)
                        found_yanbao = True

                # 文章
                for sent in split_to_sents(line):
                    results = re.findall('《(.*?)》', sent)
                    for result in results:
                        entities_by_rule['文章'].append(result)

                # 风险
                for sent in split_to_sents(line):
                    if found_fengxian:
                        sent = sent.split('：')[0]
                        fengxian_entities = re.split('以及|、|，|；|。', sent)
                        fengxian_entities = [re.sub('^[■]+[\d一二三四五六七八九十①②③④⑤]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [re.sub('^[\(（]*[\d一二三四五六七八九十①②③④⑤]+[\)）\.\s]+', '', ent) for ent in
                                             fengxian_entities]
                        fengxian_entities = [_ for _ in fengxian_entities if len(_) >= 4]
                        entities_by_rule['风险'] += fengxian_entities
                        found_fengxian = False
                    if not found_fengxian and re.findall('^\s*[\d一二三四五六七八九十]*\s*[\.、]*\s*风险提示[:：]*$', sent):
                        found_fengxian = True

                    results = re.findall('^\s*[\d一二三四五六七八九十]*\s*[\.、]*\s*风险提示[:：]*(.{5,})$', sent)
                    if results:
                        fengxian_entities = re.split('以及|、|，|；|。', results[0])
                        fengxian_entities = [re.sub('^[■]+[\d一二三四五六七八九十①②③④⑤]+', '', ent) for ent in fengxian_entities]
                        fengxian_entities = [re.sub('^[\(（]*[\d一二三四五六七八九十①②③④⑤]+[\)）\.\s]+', '', ent) for ent in
                                             fengxian_entities]
                        fengxian_entities = [_ for _ in fengxian_entities if len(_) >= 4]
                        entities_by_rule['风险'] += fengxian_entities

    for ent_type, ents in entities_by_rule.items():
        entities_by_rule[ent_type] = list(set(ents))
    return entities_by_rule
            

def custom_collate_fn(data):
    # copy from torch official，无需深究
    import collections.abc as container_abcs
    string_classes = str
    r"""Converts each NumPy array data field into a tensor"""
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        tmp_dict = {}
        for key in data:
            if key in ['sent_token_ids', 'tag_ids', 'mask']:
                tmp_dict[key] = custom_collate_fn(data[key])
                if key == 'mask':
                    tmp_dict[key] = tmp_dict[key].byte()
            else:
                tmp_dict[key] = data[key]
        return tmp_dict
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(custom_collate_fn(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [custom_collate_fn(d) for d in data]
    else:
        return data

def review_model_predict_entities(model_predict_entities):
    word_tag_map = POSTokenizer().word_tag_tab
    idf_freq = TFIDF().idf_freq
    reviewed_entities = defaultdict(list)
    for ent_type, ent_and_sent_list in model_predict_entities.items():
        for ent, sent in ent_and_sent_list:
            start = sent.lower().find(ent)
            if start == -1:
                continue
            start += 1
            end = start + len(ent) - 1
            tokens = jieba.lcut(sent)
            offset = 0
            selected_tokens = []
            for token in tokens:
                offset += len(token)
                if offset >= start:
                    selected_tokens.append(token)
                if offset >= end:
                    break

            fixed_entity = ''.join(selected_tokens)
            fixed_entity = re.sub(r'\d*\.?\d+%$', '', fixed_entity)
            if ent_type == '人物':
                if len(fixed_entity) >= 10:
                    continue
            if len(fixed_entity) <= 1:
                continue
            if re.findall(r'^\d+$', fixed_entity):
                continue
            if word_tag_map.get(fixed_entity, '') == 'v' and idf_freq[fixed_entity] < 7:
                continue
            reviewed_entities[ent_type].append(fixed_entity)
    return reviewed_entities
        
if __name__ == '__main__':
    hanlpner = HanlpNER()
    entities_by_third_party_tool = defaultdict(list)
    ls = glob.glob('../data/FR2KG/financial_research_reports/*.txt')
    for file in list(ls):
        with open(file, encoding='utf-8') as f:
            sents = [[]]
            cur_sent_len = 0
            for line in f:
                for sent in split_to_subsents(line):
                    sent = sent[:hanlpner.max_sent_len]
                    if cur_sent_len + len(sent) > hanlpner.max_sent_len:
                        sents.append([sent])
                        cur_sent_len = len(sent)
                    else:
                        sents[-1].append(sent)
                        cur_sent_len += len(sent)
            sents = [''.join(_) for _ in sents]
            sents = [_ for _ in sents if _]
            for sent in sents:
                entities_dict = hanlpner.recognize(sent)
                print(entities_dict)
                print('*' * 30)
                for ent_type, ents in entities_dict.items():
                    entities_by_third_party_tool[ent_type] += ents

        break