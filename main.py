import base64
import os
import json
from utils.schema import *
from ner_trainer import trainer
import pickle
import argparse
import torch
import tqdm
import glob
import random

from utils.helper import get_entities_by_third_party_tool, get_entities_by_rules


def find_entity(data_path):
    if os.path.exists(data_path):
        entities = {}
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
            for key in data.keys():
                entities[key] = data[key]
    else:
        if 'tool' in data_path:
            entities = get_entities_by_third_party_tool()
        elif 'rule' in data_path:
            entities = get_entities_by_rules()
        with open(data_path, 'w', encoding='utf-8') as file_obj:
            json.dump(entities, file_obj, ensure_ascii=False)
    return entities

def get_ner():
    entities_by_third_party_tool = find_entity(os.path.join('data', 'entity_by_tool.json'))
    print("Get by tool done!")
    entities_by_rules = find_entity(os.path.join('data', 'entity_by_rules.json'))
    print("Get by rules done!")
    return entities_by_third_party_tool, entities_by_rules

def NER_by_model(args, train_entities):
    texts = []
    num = 0
    for file in tqdm.tqdm(list(glob.glob('data/FR2KG/financial_research_reports/*.txt'))):
        with open(file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
        num += 1
        if num == args.select_text_num:
            break
    worker = trainer(args, texts, train_entities)
    for i in range(args.nums_round):
        print('Round: {}'.format(i + 1))
        random.shuffle(texts)
        worker.texts = texts
        worker.train()
        new_entities = worker.update(i)
        worker.train_entities = new_entities
    return new_entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
    parser.add_argument('--word_emb_dim', type=int, default=768, help='Word embedding dimension.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Input and RNN dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Applies to SGD and Adagrad.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese', help='bert模型位置')
    parser.add_argument('--data_dir', type=str, default='./data/ccks2020-stage2-open', help='输入数据文件位置')
    parser.add_argument('--out_dir', type=str, default='./output/6.17', help='输出模型文件位置')
    parser.add_argument('--out_data_dir', type=str, default='./output/data', help='输出数据文件位置')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--total_epoch_nums', type=int, default=20, help='epoch')
    parser.add_argument('--nums_round', type=int, default=5, help='nums_round')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=35)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--train_dev_ratio', type=float, default=0.9)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument('--start_update_round', type=int, default=-1)
    parser.add_argument('--select_text_num', type=int, default=1000)

    args = parser.parse_args()
    args.num_subj_type = len(entity_type2id)

    trained_entities = json.load(open('./data/FR2KG/seedKG/entities.json', 'r', encoding='utf-8'))
    # NER by rules and tool
    ent1, ent2 = get_ner()
    for entity_type, entities in ent1.items():
        trained_entities[entity_type] = list(set(trained_entities[entity_type] + entities))
    for entity_type, entities in ent2.items():
        trained_entities[entity_type] = list(set(trained_entities[entity_type] + entities))

    # NER by BERT Model
    last_entities = NER_by_model(args, trained_entities)
    pickle.dump(last_entities, open('./data/entity_by_ner.pkl', 'wb'))
    