import os
import time
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings

warnings.filterwarnings("ignore")
import gc

gc.enable()
import math
import json
import time
import random
import multiprocessing

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn import model_selection
from string import punctuation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from torch.utils.data.distributed import DistributedSampler

try:
    from apex import amp

    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)
from utils import *
from ensemble import *

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return optimal_value

logging.set_verbosity_warning()
logging.set_verbosity_error()
sys.stdout = Logger('result102_3_no_bias.txt')

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#data = pd.read_csv('../kaggle/chaii-QA/train.csv').sample(frac=1, random_state=10086).reset_index(drop=True)
data = pd.read_csv('data/train.csv').sample(frac=1, random_state=10086).reset_index(drop=True)
valid_size = data.shape[0] // 5
valid = data[:valid_size]
train = data[valid_size:]
print(train.shape)
print(valid.shape)


def generate_emnsemble_outputs(train, ifValid=False):
    train['context'] = train['context'].apply(lambda x: ' '.join(x.split()))
    train['question'] = train['question'].apply(lambda x: ' '.join(x.split()))

    tokenizer = AutoTokenizer.from_pretrained(Config().tokenizer_name)

    train_features = []
    for i, row in train.iterrows():
        train_features += prepare_train_features(Config(), row, tokenizer)

    args = Config()
    train_dataset = DatasetRetriever(train_features, mode='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=SequentialSampler(train_dataset),
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    base_model = '../kaggle/5-folds-roberta792/output/'

    def findFirst1(seq):
        for i in range(len(seq)):
            if seq[i] == 1:
                return i
        return -1

    def make_model(args):
        config = AutoConfig.from_pretrained(args.config_name)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        model = Model(args.model_name_or_path, config=config)
        return config, tokenizer, model

    with torch.no_grad():
        # print(torch.__version__)
        _, _, model = make_model(Config())
        model.cuda()

        ensemble_sequence_outputs = []
        fin_start_logits = None
        fin_end_logits = None
        for model_i in tqdm(range(5)):
            checkpoint_path = 'checkpoint-fold-{}/pytorch_model.bin'.format(model_i)
            model.load_state_dict(
                torch.load(base_model + checkpoint_path)
            )
            sequence_outputs = []
            if ifValid:
                start_logits = []
                end_logits = []
            for i, row in enumerate(tqdm(train_dataloader)):
                model.eval()

                start, end, sequence_output = model(row['input_ids'].cuda(), row['attention_mask'].cuda())
                sequence_outputs.append(sequence_output.detach().cpu())
                if ifValid:
                    start_logits.append(start.detach().cpu().tolist())
                    end_logits.append(end.detach().cpu().tolist())

                del start, end, sequence_output
            gc.collect()
            ensemble_sequence_outputs.append(torch.cat(sequence_outputs, 0).unsqueeze(0))
            if ifValid:
                start_logits = np.vstack(start_logits)
                end_logits = np.vstack(end_logits)

                if fin_start_logits is None:
                    fin_start_logits = start_logits
                    fin_end_logits = end_logits
                else:
                    fin_start_logits += start_logits
                    fin_end_logits += end_logits

        ensemble_sequence_outputs = torch.cat(ensemble_sequence_outputs, 0).permute(1, 0, 2,
                                                                                    3)  # bs, ensemble_num, max_len, dim
        model.to("cpu")
        del model
        for i in range(10):
            torch.cuda.empty_cache()
        gc.collect()
        print(ensemble_sequence_outputs.shape)

    del train_dataset, train_dataloader
    gc.collect()

    train_ensemble_features = []

    for idx in range(ensemble_sequence_outputs.shape[0]):
        feature = dict()
        feature["sequence_output"] = ensemble_sequence_outputs[idx]
        feature["attention_mask"] = train_dataset.features[idx]["attention_mask"]
        feature["offset_mapping"] = train_dataset.features[idx]["offset_mapping"]

        # pooling mask
        sequence_ids = train_dataset.features[idx]["sequence_ids2"]
        feature["pooling_mask"], feature["question_bert_len"], feature["text_bert_len"] = build_pooling_mask(
            sequence_ids)

        # label
        feature["start_position"] = train_dataset.features[idx]["start_position"]
        feature["end_position"] = train_dataset.features[idx]["end_position"]
        feature["binaray_label"] = 1 if (feature["start_position"] > 0 and feature["end_position"] > 0) else 0

        # text
        feature["example_id"] = train_dataset.features[idx]["example_id"]
        feature["context"] = train_dataset.features[idx]["context"]
        feature["answer"] = train_dataset.features[idx]["answer"]

        train_ensemble_features.append(feature)
    return train_ensemble_features, train_features, fin_start_logits, fin_end_logits


print('=' * 60)

args = Config()
config = AutoConfig.from_pretrained(args.config_name)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

real_train_ensemble_features, train_features, _, _ = generate_emnsemble_outputs(train)
valid_ensemble_features, valid_features, valid_start_logits, valid_end_logits = generate_emnsemble_outputs(valid,
                                                                                                           ifValid=True)

init_preds = postprocess_qa_predictions(valid, tokenizer, valid_features, (valid_start_logits, valid_end_logits))
init_test = postprocess_qa_predictions2(init_preds, valid)
init_jaccard_score = compute_jaccard(init_test)
print("init jaccard score:", init_jaccard_score)

for i in range(20):
    torch.cuda.empty_cache()

train_ensemble_dataset = EnsembleDataset(real_train_ensemble_features, mode='train')
train_ensemble_dataloader = DataLoader(
    train_ensemble_dataset,
    batch_size=64,
    num_workers=optimal_num_of_loader_workers(),
    pin_memory=True,
    drop_last=False,
    shuffle=True,
)

valid_ensemble_dataset = EnsembleDataset(valid_ensemble_features, mode='train')
valid_ensemble_dataloader = DataLoader(
    valid_ensemble_dataset,
    batch_size=128,
    sampler=SequentialSampler(valid_ensemble_dataset),
    num_workers=optimal_num_of_loader_workers(),
    pin_memory=True,
    drop_last=False,
)

ensemble_model = EnsembleModel(config=config)
print(ensemble_model)
ensemble_model.to("cuda")

lr = 1e-5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([p for p in ensemble_model.parameters() if p.requires_grad], lr=lr)

epochs = 5000
current_patience = 0
max_patience = 10
min_test_loss, min_epoch = 10000000, 0
max_jaccard, max_epoch = 0, 0

time1 = time.time()
t1 = datetime.datetime.fromtimestamp(time1)
t_tmp = t1
save_dir_best_loss = "best_loss_model102.pt"
save_dir_best_jaccard = "best_jaccard_model102.pt"
train_loss_list = []
valid_loss_list = []
valid_jaccard_list = []
for epoch in tqdm(range(1, epochs + 1)):
    train_loss = []
    for b_idx, b_data in enumerate(train_ensemble_dataloader):
        ensemble_model.train()
        optimizer.zero_grad()

        output = ensemble_model(b_data["sequence_output"].cuda(), b_data["pooling_mask"].cuda(),
                                text_bert_len=b_data["text_bert_len"].cuda(),
                                attention_mask=b_data["attention_mask"].cuda())
        start_logits = output[0]
        end_logits = output[1]
        loss = loss_fn(start_logits, b_data["start_position"].cuda()) + loss_fn(end_logits,
                                                                                b_data["end_position"].cuda())
        loss /= start_logits.shape[0]

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        del start_logits, end_logits, loss
        gc.collect()

    start_logits = []
    end_logits = []
    test_loss = []
    for b_idx, b_data in enumerate(valid_ensemble_dataloader):
        with torch.no_grad():
            ensemble_model.eval()
            output = ensemble_model(b_data["sequence_output"].cuda(), b_data["pooling_mask"].cuda(),
                                    text_bert_len=b_data["text_bert_len"].cuda(),
                                    attention_mask=b_data["attention_mask"].cuda())

            loss = loss_fn(output[0], b_data["start_position"].cuda()) + loss_fn(output[1],
                                                                                 b_data["end_position"].cuda())
            loss /= output[0].shape[0]
            start = output[0].detach().cpu().numpy()
            end = output[1].detach().cpu().numpy()
            start_logits.append(start)
            end_logits.append(end)
            test_loss.append(loss.item())

            del output, start, end
            gc.collect()
    start_logits = np.vstack(start_logits)
    end_logits = np.vstack(end_logits)
    fin_preds = postprocess_qa_predictions(valid, tokenizer, valid_features, (start_logits, end_logits))
    fin_test = postprocess_qa_predictions2(fin_preds, valid)
    jaccard_score = compute_jaccard(fin_test)

    train_loss_epoch = sum(train_loss) / len(train_loss)
    test_loss_epoch = sum(test_loss) / len(test_loss)
    print("train loss:{}".format(train_loss_epoch) + "\tvalid loss:{}".format(test_loss_epoch))
    print("valid jaccard:{}".format(jaccard_score))

    train_loss_list.append(train_loss_epoch)
    valid_loss_list.append(test_loss_epoch)
    valid_jaccard_list.append(jaccard_score)

    if test_loss_epoch < min_test_loss:
        min_test_loss = test_loss_epoch
        min_epoch = epoch
        save(ensemble_model, save_dir_best_loss)
        print("new best loss model saved.")

    if jaccard_score > max_jaccard:
        max_jaccard = jaccard_score
        max_epoch = epoch
        save(ensemble_model, save_dir_best_jaccard)
        print("new best jaccard model saved.")
        current_patience = 0
    else:
        current_patience += 1
        print("current_patience:" + str(current_patience))

    print('use time: ' + str(datetime.datetime.fromtimestamp(time.time()) - t_tmp))
    t_tmp = datetime.datetime.fromtimestamp(time.time())
    print("-" * 60)
    print("-" * 60)

    if max_patience <= current_patience:
        break

time2 = time.time()
t2 = datetime.datetime.fromtimestamp(time2)

print("all epoch use time:" + str(t2 - t1))
print("lowest loss epoch:{}, lowest loss:{}".format(min_epoch, min_test_loss))
print("highest jaccard epoch:{}, highest jaccard:{}".format(max_epoch, max_jaccard))
