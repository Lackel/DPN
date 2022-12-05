import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import torch.nn.functional as F
from transformers import WEIGHTS_NAME, CONFIG_NAME, AutoTokenizer
from util import accuracy_score
from pytorch_pretrained_bert.optimization import BertAdam
import math


class PretrainModelManager:
    
    def __init__(self, args, data):
        self.set_seed(args.seed)
        self.args = args
        self.data = data
        self.model = BertForModel(args, data.n_known_cls)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.num_train_steps = math.ceil(
            (2 * len(data.train_labeled_examples) + 2 * len(data.train_unlabeled_examples)) / args.pretrain_batch_size) * args.num_pretrain_epochs  
        self.optimizer = self.get_optimizer(args)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_steps)   
        return optimizer

    def save_model(self):
        if not os.path.exists(self.args.pretrain_dir):
            os.makedirs(self.args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(self.args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(self.args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def train(self):
        wait = 0
        best_model = None
        labelediter = iter(self.data.pretrain_labeled_dataloader)
        
        for epoch in range(int(self.args.num_pretrain_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(self.data.pretrain_semi_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                mask_ids, mask_lb = self.mask_tokens(input_ids.cpu(), self.tokenizer)
                mask_ids, mask_lb = mask_ids.cuda(), mask_lb.cuda()
                loss_mlm = self.model(mask_ids, input_mask, segment_ids, labels=mask_lb, mode='mlm')
                
                try:
                    batch = labelediter.next()
                except StopIteration:
                    labelediter = iter(self.data.pretrain_labeled_dataloader)
                    batch = labelediter.next()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_ce, _ = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train") 
                loss = loss_ce + loss_mlm

                loss.backward()
                tr_loss += loss.item()

                self.optimizer.step()
                self.optimizer.zero_grad()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('Epoch {} train_loss: {}'.format(epoch, loss))

            eval_score = self.eval()
            print('Epoch {} eval_score: {}'.format(epoch, eval_score))

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= self.args.wait_patient:
                    break
        
        self.model = best_model
        if self.args.save_model:
            self.save_model()

    def eval(self):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.data.n_known_cls)).to(self.device)

        for batch in self.data.eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                logits = self.model(input_ids, segment_ids, input_mask, mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc

    def mask_tokens(self, inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class BertForModel(nn.Module):
    def __init__(self, args, num_labels):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModelForMaskedLM.from_pretrained(args.bert_model)
        self.config = self.bert.config
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)


    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, mode = None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        encoded_cls = outputs.hidden_states[-1][:,0]
        output = self.dense(encoded_cls)
        output = self.activation(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        
        if mode == 'feature_extract':
            return output
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        elif mode == 'mlm':
            outputs = self.bert(input_ids, attention_mask, token_type_ids, labels=labels)
            return outputs.loss
        elif mode == 'eval':
            return logits
    


