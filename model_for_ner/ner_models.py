import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pytorch_pretrained_bert import BertModel, BertTokenizer
import pickle

class BertNER(nn.Module):
    def __init__(self, args):
        super(BertNER, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.hidden_dim = self.args.word_emb_dim
        self.model = AutoModel.from_pretrained(self.args.bert_model)
        self.top_n = self.args.topn
        self.linear_start = nn.Linear(self.hidden_dim, self.args.num_subj_type + 1)
        self.linear_end = nn.Linear(self.hidden_dim, self.args.num_subj_type + 1)
        nn.init.xavier_uniform_(self.linear_start.weight, gain=1.414)
        nn.init.xavier_uniform_(self.linear_end.weight, gain=1.414)

    def forward(self, inputs):
        # pickle.dump(inputs, open('temp.pkl', 'wb'))
        output = self.model(inputs)
        outputs, pooler_output = output.last_hidden_state, output.pooler_output
        subj_start_logits = self.linear_start(self.dropout(outputs)).squeeze(-1)
        subj_end_logits = self.linear_end(self.dropout(outputs)).squeeze(-1)
        return subj_start_logits, subj_end_logits

    def predict(self, inputs, mask):
        output = self.model(inputs)
        outputs, pooler_output = output.last_hidden_state, output.pooler_output
        subj_start_logits = self.linear_start(outputs)
        subj_start_logits = torch.argmax(subj_start_logits, 2)
        subj_start_logits = subj_start_logits.mul(mask.float())

        subj_end_logits = self.linear_end(outputs)
        subj_end_logits = torch.argmax(subj_end_logits, 2)
        subj_end_logits = subj_end_logits.mul(mask.float())
        return subj_start_logits.squeeze(-1).data.cpu().numpy().tolist(), subj_end_logits.squeeze(-1).data.cpu().numpy().tolist()


class entityModel(object):
    def __init__(self, args):
        super(entityModel, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.ner_model = BertNER(args)
        self.parameters = [p for p in self.ner_model.parameters() if p.requires_grad]
        if self.args.cuda:
            self.ner_model.cuda()
            self.criterion.cuda()
        self.optimizer = torch.optim.Adam(self.ner_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def predict(self, words, mask):
        self.ner_model.eval()
        return self.ner_model.predict(words, mask)

    def save_model(self, path):
        torch.save(self.ner_model.state_dict(), path)
        return None

    def forward(self, T, S1, S2, mask):
        inputs = T
        subj_start_type = S1
        subj_end_type = S2

        self.ner_model.train()
        self.optimizer.zero_grad()

        subj_start_logits, subj_end_logits = self.ner_model(inputs)
        subj_start_logits = subj_start_logits.view(-1, self.args.num_subj_type + 1)
        subj_start_type = subj_start_type.view(-1).squeeze()
        subj_start_loss = self.criterion(subj_start_logits, subj_start_type).view_as(mask)
        subj_start_loss = torch.sum(subj_start_loss.mul(mask.float())) / torch.sum(mask.float())

        subj_end_loss = self.criterion(subj_end_logits.view(-1, self.args.num_subj_type + 1),
                                           subj_end_type.view(-1).squeeze()).view_as(mask)
        subj_end_loss = torch.sum(subj_end_loss.mul(mask.float())) / torch.sum(mask.float())

        loss = subj_start_loss + subj_end_loss

        # backward
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.ner_model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val
