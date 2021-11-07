import os
import time
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from sklearn.utils import shuffle
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

data = pd.read_csv('data/train.csv').sample(frac=1, random_state=10086).reset_index(drop=True)

external_mlqa = pd.read_csv('data/mlqa_hindi.csv')
external_xquad = pd.read_csv('data/xquad.csv')
external_train = pd.concat([external_mlqa, external_xquad])
def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=2021)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['language'])):
        data.loc[v_, 'kfold'] = f
    return data

data = create_folds(data, num_splits=5)
external_train["kfold"] = -1
external_train['id'] = list(np.arange(1, len(external_train)+1))
#data = pd.concat([data, external_train]).reset_index(drop=True)

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

    #base_model = '../kaggle/5-folds-roberta792/output/'
    base_model = '/share/apps/kaggle/'
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

    #del train_dataset, train_dataloader
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



def loss_fn(preds, labels):
    start_preds, end_preds = preds
    start_labels, end_labels = labels

    start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    total_loss = (start_loss + end_loss) / 2
    return total_loss


def make_loader(
        args, data,
        tokenizer, fold
):
    train, valid = data[data['kfold'] != fold], data[data['kfold'] == fold]
    print(train.shape)
    print(valid.shape)
    real_train_ensemble_features, train_features, _, _ = generate_emnsemble_outputs(train)
    valid_ensemble_features, valid_features, valid_start_logits, valid_end_logits = generate_emnsemble_outputs(valid,
                                                                                                               ifValid=True)

    init_preds = postprocess_qa_predictions(valid, tokenizer, valid_features, (valid_start_logits, valid_end_logits))
    init_test = postprocess_qa_predictions2(init_preds, valid)
    init_jaccard_score = compute_jaccard(init_test)
    print("init jaccard score:", init_jaccard_score)
    train_sampler = RandomSampler(real_train_ensemble_features)
    valid_sampler = SequentialSampler(valid_ensemble_features)

    train_ensemble_dataset = EnsembleDataset(real_train_ensemble_features, mode='train')
    train_dataloader = DataLoader(
        train_ensemble_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False
    )
    valid_ensemble_dataset = EnsembleDataset(valid_ensemble_features, mode='train')
    valid_dataloader = DataLoader(
        valid_ensemble_dataset,
        batch_size=args.eval_batch_size,
        sampler=valid_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, valid_dataloader,valid,valid_features

class Trainer:
    def __init__(
            self, model, tokenizer,
            optimizer, scheduler
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(
            self, args,
            train_dataloader,
            epoch, result_dict
    ):
        count = 0
        losses = AverageMeter()

        self.model.zero_grad()
        self.model.train()

        fix_all_seeds(args.seed)

        for batch_idx, b_data in enumerate(train_dataloader):

            outputs_start, outputs_end = self.model(
                b_data["sequence_output"].cuda(), b_data["pooling_mask"].cuda(),
                text_bert_len=b_data["text_bert_len"].cuda(),
                attention_mask=b_data["attention_mask"].cuda()
            )

            loss = loss_fn((outputs_start, outputs_end), (b_data["start_position"].cuda(), b_data["end_position"].cuda()))
            loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            #count += input_ids.size(0)
            losses.update(loss.item(), b_data["sequence_output"].size(0))

            # if args.fp16:
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            # else:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

            if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (batch_idx % args.logging_steps == 0) or (batch_idx + 1) == len(train_dataloader):
                _s = str(len(str(len(train_dataloader.sampler))))
                ret = [
                    ('Epoch: {:0>2} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count,
                                                                               len(train_dataloader.sampler),
                                                                               100 * count / len(
                                                                                   train_dataloader.sampler)),
                    'Train Loss: {: >4.5f}'.format(losses.avg),
                ]
                print(', '.join(ret))

        result_dict['train_loss'].append(losses.avg)
        return result_dict
def make_model(args):
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = Model(args.model_name_or_path, config=config)
    return config, tokenizer, model
def init_training(args, data, fold):
    fix_all_seeds(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # model
    model_config = AutoConfig.from_pretrained(args.config_name)
    model = EnsembleModel(config=config)

    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(),
            torch.cuda.get_device_name(0))
        )
        model = model.cuda()
    else:
        raise ValueError('CPU training is not supported')

    # data loaders
    train_dataloader, valid_dataloader,valid ,valid_features = make_loader(args, data, tokenizer, fold)

    # optimizer
    optimizer = make_optimizer(args, model)

    # scheduler
    num_training_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs
    if args.warmup_ratio > 0:
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    else:
        num_warmup_steps = 0
    print(f"Total Training Steps: {num_training_steps}, Total Warmup Steps: {num_warmup_steps}")
    scheduler = make_scheduler(args, optimizer, num_warmup_steps, num_training_steps)

    # mixed precision training with NVIDIA Apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    result_dict = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': np.inf
    }

    return (
        model, model_config, tokenizer, optimizer, scheduler,\
        train_dataloader, valid_dataloader, result_dict,valid,valid_features
    )


class Evaluator:
    def __init__(self, model,valid,valid_features):
        self.model = model
        self.valid = valid
        self.valid_features = valid_features

    def save(self, result, output_dir):
        with open(f'{output_dir}/result_dict.json', 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def evaluate(self, valid_dataloader, epoch, result_dict):
        losses = AverageMeter()
        start_logits = []
        end_logits = []
        for batch_idx, b_data in enumerate(valid_dataloader):
            self.model = self.model.eval()

            with torch.no_grad():
                outputs_start, outputs_end = self.model(
                    b_data["sequence_output"].cuda(), b_data["pooling_mask"].cuda(),
                    text_bert_len=b_data["text_bert_len"].cuda(),
                    attention_mask=b_data["attention_mask"].cuda()
                )

                loss = loss_fn((outputs_start, outputs_end),
                               (b_data["start_position"].cuda(), b_data["end_position"].cuda()))

                losses.update(loss.item(), b_data["sequence_output"].size(0))
                start = outputs_start.detach().cpu().numpy()
                end = outputs_end.detach().cpu().numpy()
                start_logits.append(start)
                end_logits.append(end)
        start_logits = np.vstack(start_logits)
        end_logits = np.vstack(end_logits)
        fin_preds = postprocess_qa_predictions(self.valid, tokenizer, self.valid_features, (start_logits, end_logits))
        fin_test = postprocess_qa_predictions2(fin_preds, self.valid)
        jaccard_score = compute_jaccard(fin_test)



        print("valid jaccard:{}".format(jaccard_score))
        valid_jaccard_list.append(jaccard_score)
        print('----Validation Results Summary----')
        print('Epoch: [{}] Valid Loss: {: >4.5f}'.format(epoch, losses.avg))
        result_dict['val_loss'].append(losses.avg)
        return result_dict
def run(data, fold):
    args = Config()
    model, model_config, tokenizer, optimizer, scheduler, train_dataloader, \
    valid_dataloader, result_dict ,valid ,valid_features = init_training(args, data, fold)

    trainer = Trainer(model, tokenizer, optimizer, scheduler)
    evaluator = Evaluator(model,valid,valid_features)

    train_time_list = []
    valid_time_list = []

    for epoch in range(args.epochs):
        result_dict['epoch'].append(epoch)

        # Train
        torch.cuda.synchronize()
        tic1 = time.time()
        result_dict = trainer.train(
            args, train_dataloader,
            epoch, result_dict
        )
        torch.cuda.synchronize()
        tic2 = time.time()
        train_time_list.append(tic2 - tic1)

        # Evaluate
        torch.cuda.synchronize()
        tic3 = time.time()
        result_dict = evaluator.evaluate(
            valid_dataloader, epoch, result_dict
        )
        torch.cuda.synchronize()
        tic4 = time.time()
        valid_time_list.append(tic4 - tic3)

        output_dir = os.path.join(args.output_dir, f"checkpoint-fold-{fold}")
        if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
            print("{} Epoch, Best epoch was updated! Valid Loss: {: >4.5f}".format(epoch, result_dict['val_loss'][-1]))
            result_dict["best_val_loss"] = result_dict['val_loss'][-1]

            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
            model_config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saving model checkpoint to {output_dir}.")

        print()

    evaluator.save(result_dict, output_dir)

    print(
        f"Total Training Time: {np.sum(train_time_list)}secs, Average Training Time per Epoch: {np.mean(train_time_list)}secs.")
    print(
        f"Total Validation Time: {np.sum(valid_time_list)}secs, Average Validation Time per Epoch: {np.mean(valid_time_list)}secs.")

    torch.cuda.empty_cache()
    del trainer, evaluator
    del model, model_config, tokenizer
    del optimizer, scheduler
    del train_dataloader, valid_dataloader, result_dict
    gc.collect()

valid_jaccard_list = []
for fold in range(1):
    print();print()
    print('-'*50)
    print(f'FOLD: {fold}')
    print('-'*50)
    run(data, fold)