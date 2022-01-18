import numpy as np
from typing import List, Tuple, Dict

from collections import namedtuple

import os

from model import Model
import random

import torch
from torch import nn
import pytorch_lightning as pl

from transformers import BertTokenizer, BertModel

def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return StudentModel(device, "b") 
    #return RandomBaseline()

def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    return StudentModel(device, "ab") 
    # return RandomBaseline(mode='ab')
    # raise NotImplementedError

def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    return RandomBaseline(mode='cd')
    #raise NotImplementedError

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
        ("service", 248),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device, type_) -> None:
        super().__init__()
        
        self.classification_labels = {
            0:"positive",
            1:"negative",
            2:"neutral",
            3:"conflict"
        }

        self.device = device

        # store the type of prediction required from the model
        self.type = type_
        self.model_a = None
        self.model_b = None

        # store the paths of the two models
        self.chk_path_a = os.sep.join(["model", "best_model_A.ckpt"])
        self.chk_path_b = os.sep.join(["model", "best_model_B.ckpt"])

        # download the two tokenizers
        # the tokenizer for the task B has a special token for the aspect term considered
        self.tokenizerB = BertTokenizer.from_pretrained('bert-large-uncased', additional_special_tokens=["<target-term>"])
        self.tokenizerA = BertTokenizer.from_pretrained('bert-large-cased')
        
        # save the id of the special token for the task B tokenizer
        self.target_idB = self.tokenizerB("<target-term>", return_tensors="pt")["input_ids"][0][1]


    # forward pass of the task A
    """
        x: Bert input dict
            input_ids: defines the sequence of word pieces ids of the sentence
            attention_mask
        word_pieces: a list containing for each sentence a list containing for 
                    each token formed by multiple aspect terms the ids of them
        lenghts: the number of word pieces in the sentence

        returns the predicted logits
    """
    @torch.no_grad()
    def forward_a(self, x:Dict, word_pieces:List, lenghts:List):
        return self.model_a(x,word_pieces, lenghts)

    """
        samples: list of dictionaries containing
                text: the sentence
    """
    def predict_ab(self, samples: List[Dict]) -> List[Dict]:

        # since it takes to much to load the model, I load it at the first
        # call to the model
        if(self.model_a == None):
            print("start loading model a")
            
            # load the model from checkpoint
            self.model_a = TaskALightningModule.load_from_checkpoint(self.chk_path_a).model
            
            #put the model in evaluation mode
            self.model_a.eval()
            print("loaded_model_a")
        
        # get the data as dictionary:
        """
            {
                "inputs": the word pieces encoding,
                "rebuild_phrase": the splitted sentence,
                "word_pieces_to_merge": list of word pieces that form a single term,
                "lengths": the number of tokens in the sentence
            }
        """
        encoded_data = self.task_a_init_data(samples)

        #obtain all the data we need
        """
                {
                    "inputs": Bert input: indices_ids and attention mask
                    "lengths": the number of tokens in each sentence 
                    "mask": mask to ignore the pad values
                    "splitted_texts": the splitted sentences 
                    "word_pieces": list of word pieces that form a single term
                    "input_lengths": number of word pieces for each sentence
                }
        """
        batch = self.task_a_collate_fn(encoded_data) 
        
        inputs = batch['inputs']
        # pass the inputs over the device
        inputs["input_ids"].to(self.device)
        inputs["attention_mask"].to(self.device)
        
        lengths = batch['lengths']
        input_lengths = batch["input_lengths"]
        mask = batch['mask']
        splitted_texts = batch['splitted_texts']
        word_pieces = batch["word_pieces"]

        # forward pass: get the logits 
        logits = self.forward_a(inputs, word_pieces, input_lengths)

        #apply the sigmoind and round the values
        preds = torch.sigmoid(logits)
        rounded = torch.round(preds)

        ret = []

        # for each sentence
        for i, row_ in enumerate(rounded):

            # avoid to consider the padding values
            row = row_[:lengths[i]]
            
            # get the indices of terms that are considered aspect terms 
            # we will merge the consecutive terms that are predicted to be
            # aspect terms
            indeces = (row.squeeze()==1).nonzero()

            # computed targets will contain the aspect terms
            # computed_targets_idxs will contain a list of indices of terms
            # for each aspect term
            computed_targets, computed_targets_idxs, last_index = [], [], -1
            if (lengths[i] == 1):
                if (row[0][0] == 1):
                    computed_targets = splitted_texts[i][0]        
                    computed_targets_idxs =  [0]                               
            else:
                # for each index of term considered an aspect term by the model
                for index in indeces:

                    # if the index is the first one or is not the consecutive of the 
                    # precedent index considered, than we add a new aspect term
                    if (index != last_index + 1 or last_index == -1):
                        computed_targets.append(splitted_texts[i][index])
                        computed_targets_idxs.append([index]) 
                    
                    # otherwise we add the term to the last aspect term
                    # if the term or the precedent one are "-" we do not add the space in the middle
                    elif computed_targets[-1][-1] == "-" or splitted_texts[i][index] == "-":
                        computed_targets[-1] += splitted_texts[i][index]
                        computed_targets_idxs[-1].append(index) 
                    # otherwise we add a space between the two terms
                    else:
                        computed_targets[-1] += " " + splitted_texts[i][index]
                        computed_targets_idxs[-1].append(index) 

                    last_index = index 
            
            #list of sentences where the aspect term has been replaced with the special token 
            texts_with_target = []
            for target in computed_targets_idxs:
                # get the sentences before and after the aspect term considered
                splitted_text_pre, splitted_text_post = splitted_texts[i][:target[0]], splitted_texts[i][target[-1]+1:]
                
                # build the text as concatenation of the sentence pre aspecte term + special token + sentence post aspect term
                text_with_target = self.merge_splitted_text(splitted_text_pre) + "<target-term>" + " " + self.merge_splitted_text(splitted_text_post)
                
                texts_with_target.append({"inputs":text_with_target})

            # if there is at least one aspect term, predict the sentiment
            if (texts_with_target != []):
                sentiments = self.predict_b(texts_with_target)

                ret.append({"targets":list(zip(computed_targets, sentiments))})
            else:
                ret.append({"targets":[]})
                
        return ret
    

    """
        samples: list of dicts containing
                {
                    text: sentence
                    targets: list containing [[start idx], aspect term] for each aspect term
                }
    """
    def pre_predict_b(self, samples):
        res = []
        for phrase in samples:
            #target_terms will contain the aspect terms
            # texts_with_target will contain the sentences with the aspect term replaced with
            # the speical token
            target_terms, texts_with_target = [], []        
            for target in phrase["targets"]:
                
                # get the indices of the aspect term
                start_target, end_target = target[0][0], target[0][1]
                
                # replace the aspect term with the special token
                tmp_text = phrase["text"][:start_target] + " <target-term> " + phrase["text"][end_target:]    
                
                # append the data                 
                texts_with_target.append({"inputs":tmp_text})
                target_terms.append(target[1])

            # if there is at least one aspect predict the sentence
            if(target_terms != []):
                sentiments = self.predict_b(texts_with_target)
                
                # associate aspect terms and predicted sentiments
                res.append({"targets":list(zip(target_terms, sentiments))})

            else:
                res.append({"targets":[]})

        return res

    """
        texts_with_target: list of dictionaries containing:
                                {
                                    "inputs": the sentence with the aspect term replaced with the special token
                                }
    """
    @torch.no_grad()
    def predict_b(self, texts_with_target: List[Dict]) -> List[Dict]:
        if(self.model_b == None):
            self.model_b = TaskBLightningModule.load_from_checkpoint(self.chk_path_b, tokenizer=self.tokenizerB).model
            # put the model in eval mode
            self.model_b.eval()

        input_b = self.task_b_collate_fn(texts_with_target)
        logits_b = self.model_b(input_b["inputs"].to(self.device), input_b["indices"])

        # apply softmax in order to obtain a probability distribution over the labels
        preds = nn.Softmax(-1)(logits_b)

        # get the predicted label and the correct one
        arg_max = torch.argmax(preds, dim=-1).detach().cpu().tolist()

        sentiments = self.label_indices2sentiment(arg_max)              

        
        return sentiments

    def predict_cd(self, samples: List[Dict]) -> List[Dict]:
        if(self.model_c == None or self.model_d == None):
            #load_model c and d
            pass
    
    """
        word_pieces: list of word pieces

        returns the sentence where the word pieces 
        relative to the same term
        have been merged into single tokens
        by removiung the ##
    """
    def word_pieces2string(self, word_pieces:List):
        s = ""
        for word_piece in word_pieces:
            if (len(word_piece) > 2 and word_piece[:2] == "##"):
                s += word_piece[2:]
            else:
                s += " " + word_piece
        return s

    # returns for each label the name of the classified class
    def label_indices2sentiment(self, labels:List):
        return [self.classification_labels[i] for i in labels]

    """
        word_pieces: List of word pieces
        start_idx: the starting index to add

        returns 
            word_pieces_to_merge: the list of lists
                            where each internal list is the list of
                            indices of the word pieces forming that 
                            a term
            tot_to_subtract: the number of word pieces starting with ##
    """
    def get_indexes_to_merge(self, word_pieces:List, start_idx:List):

        word_pieces_to_merge = []
        started = False
        tot_to_subtract = 0 # number of word pieces starting with ##
        for i, word_piece in enumerate(word_pieces):
            idx = start_idx + i
            # if the word piece of the term is not the first one
            # that form that term 
            if (len(word_piece)>2 and word_piece[0:2] == "##"):
                # if is not the first word piece of the term with ##
                # than we have only to append the index to the last entry
                if (started):
                    word_pieces_to_merge[-1].append(idx)
                
                #else: this is the first word piece with ## of the term
                # so we have to add both the indices idx-1 and idx
                # since the term has started the word piece before
                else:
                    word_pieces_to_merge.append([idx-1, idx])
                started = True
                tot_to_subtract += 1
            else:
                started = False
        
        return word_pieces_to_merge, tot_to_subtract

    """
        data: list of Dictionaries contining
                text: the sentence
    """
    def task_a_init_data(self, data:List[Dict]):
        encoded_data = list()
        
        for i in range(len(data)):
            # get the sentence
            text = data[i]["text"]

            # begin of sentence
            tokenized = [101]
            rebuild_phrase = ["[start]"]

            # get the word pieces            
            word_pieces = self.tokenizerA.tokenize(text)

            # get the indices of the word pieces that form a single term
            word_pieces_to_merge, _ = self.get_indexes_to_merge(word_pieces, len(tokenized))

            # rebuild the phrase
            rebuild_phrase += self.word_pieces2string(word_pieces).split()
            
            # get the encoding of the word pieces
            tokenized = self.tokenizerA.encode(word_pieces)
            
            # number of tokens
            length = len(rebuild_phrase)

            encoded_data.append({
                "inputs": torch.tensor(tokenized),
                "rebuild_phrase": rebuild_phrase,
                "word_pieces_to_merge": word_pieces_to_merge,
                "lengths":length
            })

        return encoded_data

    """
        splitted_text:List of terms

        returns the reduction of the list by summing the terms with a space
        between them if are not equal to "-"
    """
    def merge_splitted_text(self, splitted_text:List):
        s = ""
        for i, term in enumerate(splitted_text):
            if (i>0 and (term == "-" or splitted_text[i-1] == "-")):
                s += term
            else:
                s += term + " "
        
        return s

    
    """
        data is a list of dictionaries
            {
                "inputs": the encoding of the word pieces of the sentence
                "rebuild_phrase": the splitted sentence
                "word_pieces_to_merge": the indices of word pieces forming a term
                "lengths": the number of terms in the sentence
            }
    """
    def task_a_collate_fn(self, data:List[Dict]):
        # get the encoding of the word pieces of each sentence
        X = [entry["inputs"] for entry in data]

        # get the number of word pieces of each sentence
        input_lengths = [entry["inputs"].shape[0] for entry in data]

        # get the number of tokens of each sentence
        lengths = [entry["lengths"] for entry in data]

        # pad the encoded word pieces
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=0)

        # get a mask to ignore the padded results
        mask = [ True if i<len else False for len in lengths for i in range(max(lengths)) ]

        # get the word pieces indices forming a term for each sentence 
        word_pieces = [entry["word_pieces_to_merge"] for entry in data]
        # get the splitted text for each sentence
        splitted_texts = [entry["rebuild_phrase"] for entry in data]
        
        # build the attention mask in order to ignore the pad values
        attention_masks = torch.tensor([[1]*i_len + [0]*(X.shape[1] - i_len) for i_len in input_lengths])

        # define Bert input
        input = {"input_ids": X, "attention_mask": attention_masks}

        return {"inputs":input,
                "lengths":lengths, "mask":mask, 
                "splitted_texts": splitted_texts, "word_pieces": word_pieces,
                "input_lengths":input_lengths}

    def task_b_collate_fn(self, data):
        print(data)
        # build a list of sentences
        X = [entry["inputs"] for entry in data]
        # tokenize each sentence and pad them 
        X = self.tokenizerB(X, return_tensors="pt", padding=True)
        
        # get theindex for each sentence where the target term is
        index = [(X["input_ids"][i] == self.target_idB).nonzero(as_tuple=True)[0] for i in range(X["input_ids"].shape[0])]

        return {"inputs":X, "indices":index}

    def predict(self, samples: List[Dict]) -> List[Dict]:
        '''
        --> !!! STUDENT: implement here your predict function !!! <--
        Args:
            - If you are doing model_b (ie. aspect sentiment analysis):
                sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza.",
                            "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                            "targets": [[4, 9], "people", [[36, 40], "taste"]]
                        }
                    ]
            - If you are doing model_ab or model_cd:
                sentence: a dictionary that represents an input sentence, for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza."
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                        }
                    ]
        Returns:
            A List of dictionaries with your predictions:
                - If you are doing target word identification + target polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                        }
                    ]
                - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                            "categories": [("food", "conflict")]
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                            "categories": [("service", "positive"), ("food", "positive")]
                        }
                    ]
        '''
        if(self.type == "ab"):
            return self.predict_ab(samples)
        elif(self.type == "b"):
            return self.pre_predict_b(samples)
        else:
            return self.predict_cd(samples)

class TaskALightningModule(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(TaskALightningModule, self).__init__(*args, **kwargs)
        """
          hparams contains 
            dropout value
            modality: MAX, AVG or SUM that define how to deal tokens with more word piece
        """
        self.save_hyperparameters(hparams)
        
        # initialize the binary cross entropy loss 
        self.loss_function = torch.nn.BCELoss()
        
        #initialize the model
        self.model = TaskAModel(self.hparams)

    # This performs a forward pass of the model, as well as returning the predicted index.
    def forward(self, x, word_pieces, lenghts):
        logits = self.model(x, word_pieces, lenghts)
        #predictions = torch.argmax(logits, -1)
        return logits
    # This runs the model in training mode mode, ie. activates dropout and gradient computation. It defines a single training step.
    def training_step(self, batch, batch_nb):
        pass
    # This runs the model in eval mode, ie. sets dropout to 0 and deactivates grad. Needed when we are in inference mode.
    def validation_step(self, batch, batch_nb):
        pass
    # This runs the model in eval mode, ie. sets dropout to 0 and deactivates grad. Needed when we are in inference mode.
    def test_step(self, batch, batch_nb):
        pass
    

class TaskBLightningModule(pl.LightningModule):

    """
        model: the model we want to train.
    """
    def __init__(self, hparams, tokenizer, *args, **kwargs):
        super(TaskBLightningModule, self).__init__(*args, **kwargs)
    
        # save hparams
        self.save_hyperparameters(hparams)
    
        # initialize the loss function as a weighted CrossEntropyLoss for multiclass classification 
        self.loss_function = nn.CrossEntropyLoss(weight=self.hparams.weights)
        
        # initialize the model we want to train
        self.model = TaskBModel(self.hparams, tokenizer)

        self.logits = []
        self.labels_indices = []
        self.arg_maxs = []
    
    # This performs a forward pass of the model
    # returns the predicted logits
    def forward(self, x, indices):
        logits = self.model(x, indices)

        return logits

    # training step
    def training_step(self, batch, batch_nb):
        return 0

    # validation step -> model in eval state 
    def validation_step(self, batch, batch_nb):
        pass
    # test step -> model in eval state
    def test_step(self, batch, batch_nb):
        pass


class TaskAModel(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams):
        super(TaskAModel, self).__init__()
        
        # load the pretrained bert large cased model
        self.bert = BertModel.from_pretrained('bert-large-cased', output_hidden_states=True)

        # define dropout and the FC
        self.dropout = nn.Dropout(hparams.dropout)
        # 4*1024 since 1024 is the size of an hidden layer, 4 cause we 
        # concatenate the last 4 hidden layers
        
        # 1 since I am doing binary classification
        self.lin1 = torch.nn.Linear(4*1024, 1) 

        self.modality = hparams.modality

    """
        x: dict with 
          input_ids: the padded sequence of ids of word pieces
          attention_mask
        word_pieces: list containing the indices of the word pieces that we have
                     to merge in order to get the result for a token
    """
    def forward(self, x, word_pieces, lenghts):
        bert_out = self.bert(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        
        x = x["input_ids"]
        
        # initialize the lists that will contain the embeddings of the tokens
        # (not word pieces)
        last_list       = []
        last_but_one    = []
        last_but_two    = []
        last_but_three  = []
        
        # for each phrase, compute the list of embs of the terms insied it 
        # by considering for each token the avg, sum or max of the embs of the 
        # word pieces 
        for i in range(x.shape[0]):
            last_list_of_embs, last_but_one_list_of_embs, last_but_two_list_of_embs, last_but_three_list_of_embs = [], [], [], []
            
            last = 0
            bo_i = bert_out.hidden_states[-1][i]

            bh_1 = bert_out.hidden_states[-2][i]
            bh_2 = bert_out.hidden_states[-3][i]
            bh_3 = bert_out.hidden_states[-4][i]


            removed = 0

            for word_piece in word_pieces[i]:

                if (self.modality == "SUM"):
                    # with sum modality for each token formed by more than 1 word piece,
                    # compute the sum of the embs of the word piece
                    last_list_of_embs+=bo_i[last:word_piece[0]]
                    last_list_of_embs.append(sum(bo_i[word_piece[0]:word_piece[-1] + 1]))
                    
                    last_but_one_list_of_embs+=bh_1[last:word_piece[0]]
                    last_but_one_list_of_embs.append(sum(bh_1[word_piece[0]:word_piece[-1] + 1]))

                    last_but_two_list_of_embs+=bh_2[last:word_piece[0]]
                    last_but_two_list_of_embs.append(sum(bh_2[word_piece[0]:word_piece[-1] + 1]))

                    last_but_three_list_of_embs+=bh_3[last:word_piece[0]]
                    last_but_three_list_of_embs.append(sum(bh_3[word_piece[0]:word_piece[-1] + 1]))
                
                elif (self.modality == "MAX"):
                    # with max modality for each token formed by more than 1 word piece,
                    # compute the max (element wise) of the embs of the word piece
                    last_list_of_embs+=bo_i[last:word_piece[0]]
                    last_list_of_embs.append(torch.max(bo_i[word_piece[0]:word_piece[-1] + 1], -2)[0])
                    
                    last_but_one_list_of_embs+=bh_1[last:word_piece[0]]
                    last_but_one_list_of_embs.append(torch.max(bh_1[word_piece[0]:word_piece[-1] + 1], -2)[0])

                    last_but_two_list_of_embs+=bh_2[last:word_piece[0]]
                    last_but_two_list_of_embs.append(torch.max(bh_2[word_piece[0]:word_piece[-1] + 1], -2)[0])

                    last_but_three_list_of_embs+=bh_3[last:word_piece[0]]
                    last_but_three_list_of_embs.append(torch.max(bh_3[word_piece[0]:word_piece[-1] + 1],-2)[0])
                  
                elif (self.modality == "AVG"):
                    # with avg modality for each token formed by more than 1 word piece,
                    # compute the avg of the embs of the word piece
                    last_list_of_embs+=bo_i[last:word_piece[0]]
                    last_list_of_embs.append(sum(bo_i[word_piece[0]:word_piece[-1] + 1])/len(word_piece))
                    
                    last_but_one_list_of_embs+=bh_1[last:word_piece[0]]
                    last_but_one_list_of_embs.append(sum(bh_1[word_piece[0]:word_piece[-1] + 1])/len(word_piece))

                    last_but_two_list_of_embs+=bh_2[last:word_piece[0]]
                    last_but_two_list_of_embs.append(sum(bh_2[word_piece[0]:word_piece[-1] + 1])/len(word_piece))

                    last_but_three_list_of_embs+=bh_3[last:word_piece[0]]
                    last_but_three_list_of_embs.append(sum(bh_3[word_piece[0]:word_piece[-1] + 1])/len(word_piece))
                else:
                    raise Exception("Declare the modality")
                
                last = word_piece[-1]+1
            
                removed += len(word_piece)-1

            # take the tokens from the last term formed by multiple word pieces
            # and the end of the sentence
            if (last < bo_i.shape[0]):
                last_list_of_embs+=(bo_i[last:lenghts[i]])
                last_but_one_list_of_embs+=(bh_1[last:lenghts[i]])
                last_but_two_list_of_embs+=(bh_2[last:lenghts[i]])
                last_but_three_list_of_embs+=(bh_3[last:lenghts[i]])
            
            #stack the embs of the terms of the sentence
            last_list_of_embs           = torch.stack(last_list_of_embs)
            last_but_one_list_of_embs   = torch.stack(last_but_one_list_of_embs)
            last_but_two_list_of_embs   = torch.stack(last_but_two_list_of_embs)
            last_but_three_list_of_embs = torch.stack(last_but_three_list_of_embs)

            # add the stacked embs into these lists
            last_list.append(last_list_of_embs)
            last_but_one.append(last_but_one_list_of_embs)
            last_but_two.append(last_but_two_list_of_embs)
            last_but_three.append(last_but_three_list_of_embs)

        # pad the sequences with zeros
        last_padded_embs            = torch.nn.utils.rnn.pad_sequence(last_list, batch_first=True, padding_value=0)
        last_but_one_padded_embs    = torch.nn.utils.rnn.pad_sequence(last_but_one, batch_first=True, padding_value=0)
        last_but_two_padded_embs    = torch.nn.utils.rnn.pad_sequence(last_but_two, batch_first=True, padding_value=0)
        last_but_three_padded_embs  = torch.nn.utils.rnn.pad_sequence(last_but_three, batch_first=True, padding_value=0)
        
        # concatenate the results from the last 4 hidden layers of bert
        padded_embs = torch.cat((last_padded_embs, last_but_one_padded_embs, last_but_two_padded_embs, last_but_three_padded_embs), -1)

        padded_embs = self.dropout(padded_embs)

        out = self.lin1(padded_embs)
        
        
        return out

class TaskBModel(nn.Module):
    
    """
        hparams: dict with dropout and num_classes informations
        tokenizer: the tokenizer used to tokenize the sentences
        
    """

    def __init__(self, hparams, tokenizer):
        super(TaskBModel, self).__init__()

        # download the bert pretrained model        
        self.bert = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
        
        # resize the embeddings since I added a new token (<target-term>)
        self.bert.resize_token_embeddings(len(tokenizer))

        #initialize dropout and two FC layers
        self.dropout = nn.Dropout(hparams.dropout)

        # 1024 is the output size for bert large
        # 4 * 1024 since I am going to concatenate the last 4 layers
        self.lin1 = torch.nn.Linear(4*1024, hparams.num_classes) 

    def forward(self, input, indices):

        bert_out = self.bert(**input)

        # get the output of the last 4 layers relative to the <target-term> word_piece
        last = bert_out.hidden_states[-1][range(input["input_ids"].shape[0]), indices, :]
        last_but_one = bert_out.hidden_states[-2][range(input["input_ids"].shape[0]), indices, :]
        last_but_two = bert_out.hidden_states[-3][range(input["input_ids"].shape[0]), indices, :]
        last_but_three = bert_out.hidden_states[-4][range(input["input_ids"].shape[0]), indices, :]

        # stack the outputs
        bert_out = torch.hstack((last, last_but_one, last_but_two, last_but_three))
        last, last_but_one, last_but_two, last_but_three = None, None, None, None
        
        # dropout
        bert_out = self.dropout(bert_out)

        # FC
        out = self.lin1(bert_out)

        return out