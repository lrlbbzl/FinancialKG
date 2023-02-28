import torch
from utils.preprocessing import Article, preprocess_data
import random
import json
import tqdm
import os
import pickle
from transformers import AutoTokenizer, AutoModel
from dataset import MyDataset
from model_for_ner.ner_models import BertNER, entityModel
from utils.helper import custom_collate_fn, review_model_predict_entities
from utils.extract import extract_entities
from utils.metrics import evaluator
from utils.schema import entity_id2type, entity_type2id, type2query, query2type
from collections import defaultdict

class trainer():
    def __init__(self, args, texts, train_entities):
        self.args = args
        self.texts = texts
        self.texts_num = len(self.texts)
        self.train_entities = train_entities
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)
        self.model = entityModel(self.args)
        self.device = 'cuda' if self.args.cuda else 'cpu'

    def build_dataloader(self, data, shuffle=True):
        dataset = MyDataset(data, self.tokenizer)
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=custom_collate_fn, shuffle=shuffle)

    def evaluate(self, data_loader):
        founded_entities_json = defaultdict(set)
        golden_entities_json = defaultdict(set)
        for batch_id, one_data in enumerate(data_loader):
            T = torch.stack([d['T'] for d in one_data]).to(self.device)
            mask = torch.stack([d['mask'] for d in one_data]).to(self.device)
            subj_start_logits, subj_end_logits = self.model.predict(T, mask)

            for data, subj_start_list, subj_end_list in zip(one_data, subj_start_logits, subj_end_logits):
                for entity_label in data['entity_labels']:
                    golden_entities_json[entity_label['entity_type']].add(entity_label['entity_name'])

                tokens = data['sent_tokens']
                for i, _ss1 in enumerate(subj_start_list):
                    if _ss1 > 0:
                        for j, _ss2 in enumerate(subj_end_list[i:]):
                            if _ss2 == _ss1:
                                entity = ''.join(tokens[i: i + j + 1])
                                entity_type = entity_id2type[_ss2]
                                founded_entities_json[entity_type].add(entity)
                                break

        evaluate_tool = evaluator(golden_entities_json, founded_entities_json)
        scores = evaluate_tool.compute_entities_score()
        return scores

    def test(self, data_loader):
        founded_entities = defaultdict(set)
        for batch_id, one_data in enumerate(tqdm.tqdm(data_loader)):
            T = torch.stack([d['T'] for d in one_data]).to(self.device)
            mask = torch.stack([d['mask'] for d in one_data]).to(self.device)
            subj_start_logits, subj_end_logits = self.model.predict(T, mask)

            for data, subj_start_list, subj_end_list in zip(one_data, subj_start_logits, subj_end_logits):
                sent = data['sent']
                tokens = data['sent_tokens']
                for i, _ss1 in enumerate(subj_start_list):
                    if _ss1 > 0:
                        for j, _ss2 in enumerate(subj_end_list[i:]):
                            if _ss2 == _ss1:
                                entity = ''.join(tokens[i: i + j + 1])
                                entity_type = entity_id2type[_ss2]
                                founded_entities[entity_type].add((entity, sent))
                                break
        result = defaultdict(list)
        for ent_type, ents in founded_entities.items():
            result[ent_type] = list(set(ents))
        return result

    def train(self, ):
        self.train_data = self.texts[ : int(self.texts_num * self.args.train_test_ratio)]
        self.test_data = self.texts[int(self.texts_num * self.args.train_test_ratio) : ]
        self.train_num = len(self.train_data)
        random.shuffle(self.train_data)
        train_data = self.train_data[ : int(self.train_num * self.args.train_dev_ratio)]
        valid_data = self.train_data[int(self.train_num * self.args.train_dev_ratio) : ]
        train_data, valid_data = preprocess_data(self.train_entities, train_data, self.tokenizer), \
                    preprocess_data(self.train_entities, valid_data, self.tokenizer)
        train_dataloader, valid_dataloader = self.build_dataloader(train_data), self.build_dataloader(valid_data)
        pickle.dump(train_dataloader, open('train_dataloader.pkl', 'wb'))
        pickle.dump(valid_dataloader, open('valid_dataloader.pkl', 'wb'))
        pickle.dump(self.test_data, open('test_data.pkl', 'wb'))
        best_evaluate_score = 0

        for epoch in range(self.args.total_epoch_nums):
            # train stage
            pbar = tqdm.tqdm(train_dataloader)
            for batch_id, one_data in enumerate(pbar):
                T = torch.stack([d['T'] for d in one_data]) # tokens 
                S1 = torch.stack([d['S1'] for d in one_data]) # start idx which values entity type
                S2 = torch.stack([d['S2'] for d in one_data]) # end idx which values entity type
                mask = torch.stack([d['mask'] for d in one_data]) # mask 1, 1, ..., 0, 0

                loss = self.model.forward(T, S1, S2, mask)
                pbar.set_description('epoch: {}, loss: {:.3f}'.format(epoch + 1, loss))

            # valid stage
            score = self.evaluate(valid_dataloader)
            f1, precision, recall = score['f'], score['p'], score['r']
            print("F1: {}, Precision: {}, Recall: {}".format(f1, precision, recall))
            if f1 > best_evaluate_score:
                # find best model during training by evaluating on valid dataset
                best_evaluate_score = f1
                self.model.save_model(os.path.join(self.args.save_path, 'best_model.pth'))

    def update(self, num_round):
        self.model.ner_model.load_state_dict(torch.load(os.path.join(self.args.save_path, 'best_model.pth')))
        test_data = preprocess_data(self.train_entities, self.test_data, self.tokenizer, for_train=False)
        test_dataloader = self.build_dataloader(test_data)
        res = self.train_entities
        if num_round > self.args.start_update_round:
            # update entities when train_round > start update round
            predict_entities = self.test(test_dataloader)
            reviewed_entities = review_model_predict_entities(predict_entities)
            update_entities = extract_entities(reviewed_entities, self.train_entities)
            for ent_type, entities in update_entities.items():
                res[ent_type] = list(set(entities + res[ent_type]))
        return res
                