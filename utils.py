import torch
import numpy as np
import pandas as pd
from string import punctuation
import sys

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

class Config:
    # model
    model_type = 'xlm_roberta'
    model_name_or_path = "../kaggle/xlm-roberta-large-squad-v2"
    config_name = "../kaggle/xlm-roberta-large-squad-v2"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    # tokenizer
    tokenizer_name = "../kaggle/xlm-roberta-large-squad-v2"
    max_seq_length = 400
    doc_stride = 135

    # train
    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    # optimzer
    optimizer_type = 'AdamW'
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    # scheduler
    decay_name = 'linear-warmup'
    warmup_ratio = 0.1

    # logging
    logging_steps = 10

    # evaluate
    output_dir = 'output'
    seed = 2021


def build_pooling_mask(sequence_ids):
    question_start = 1
    # question_end = ?

    for i in range(question_start, len(sequence_ids)):
        if sequence_ids[i] is None:
            question_end = i - 1
            break

    text_start = question_end + 3
    # text_end = ?
    for i in range(text_start, len(sequence_ids)):
        if sequence_ids[i] is None:
            text_end = i - 1
            break

    res = torch.zeros(len(sequence_ids), len(sequence_ids))
    for i in range(text_start, text_end + 1):
        for j in range(question_start, question_end + 1):
            res[i][j] = 1

    return res, question_end - question_start + 1, text_end - text_start + 1



def save(model, filename):
    params = {
        'model': model.state_dict(),
    }
    # try:
    torch.save(params, filename)
    print("model saved to {}".format(filename))


import collections


def postprocess_qa_predictions(examples, tokenizer, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions


def postprocess_qa_predictions2(fin_preds, test):
    submission = []
    for p1, p2 in fin_preds.items():
        p2 = " ".join(p2.split())
        p2 = p2.strip(punctuation)
        submission.append((p1, p2))

    sample = pd.DataFrame(submission, columns=["id", "PredictionString"])

    test_data = pd.merge(left=test, right=sample, on='id')

    bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
    bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

    tamil_ad = "கி.பி"
    tamil_bc = "கி.மு"
    tamil_km = "கி.மீ"
    hindi_ad = "ई"
    hindi_bc = "ई.पू"

    cleaned_preds = []
    for pred, context in test_data[["PredictionString", "context"]].to_numpy():
        if pred == "":
            cleaned_preds.append(pred)
            continue
        while any([pred.startswith(y) for y in bad_starts]):
            pred = pred[1:]
        while any([pred.endswith(y) for y in bad_endings]):
            if pred.endswith("..."):
                pred = pred[:-3]
            else:
                pred = pred[:-1]
        if pred.endswith("..."):
            pred = pred[:-3]

        if any([pred.endswith(tamil_ad), pred.endswith(tamil_bc), pred.endswith(tamil_km), pred.endswith(hindi_ad),
                pred.endswith(hindi_bc)]) and pred + "." in context:
            pred = pred + "."

        cleaned_preds.append(pred)

    test_data["PredictionString"] = cleaned_preds
    # test_data[['id', 'PredictionString']].to_csv('submission.csv', index=False)
    return test_data

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard(fin_test):
    size = fin_test.shape[0]
    s = 0
    for i in range(size):
        answer_truth = fin_test.loc[i, "answer_text"]
        answer_predict = fin_test.loc[i, "PredictionString"]

        s += jaccard(answer_truth, answer_predict)
    return s / size





class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, 'a')

    def write(self, message):
        '''print实际相当于sys.stdout.write'''
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



def prepare_train_features(args, example, tokenizer):
    example["question"] = example["question"].lstrip()
    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_example.pop("offset_mapping")

    features = []
    for i, offsets in enumerate(offset_mapping):
        feature = {}

        input_ids = tokenized_example["input_ids"][i]
        attention_mask = tokenized_example["attention_mask"][i]

        feature['input_ids'] = input_ids
        feature['attention_mask'] = attention_mask
        feature['offset_mapping'] = offsets
        feature['sequence_ids'] = [0 if i is None else i for i in tokenized_example.sequence_ids(i)]
        feature['sequence_ids2'] = tokenized_example.sequence_ids(i)
        feature["example_id"] = example['id']
        feature['context'] = example['context']
        feature['answer'] = example['answer_text']

        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_example.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = example["answer_text"]

        if len(str(example["answer_start"])) == 0:
            raise
            feature["start_position"] = cls_index
            feature["end_position"] = cls_index
        else:
            start_char = example["answer_start"]
            end_char = start_char + len(answers)

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                feature["start_position"] = cls_index
                feature["end_position"] = cls_index
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                feature["start_position"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                feature["end_position"] = token_end_index + 1

        features.append(feature)
    return features




