from typing import Dict, Any
import json
import logging
import random
import re
import sys


from transformers import AutoTokenizer


from overrides import overrides

# from minimal_allennlp import Field, Instance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/

logger = logging.getLogger("dataloader")  # pylint: disable=invalid-name

logger.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler("dataloader.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


dataset_dir = (
    "/data/hzz5361/raw_data/rule-reasoning-dataset-V2020.2.5.0/original/depth-3/"
)

train_data_path = dataset_dir + "train.jsonl"
validation_data_path = dataset_dir + "dev.jsonl"
test_data_path = dataset_dir + "test.jsonl"


class RuleTakerDataset(Dataset):
    def __init__(
        self,
        pretrained_model: str,
        path: str,
        max_pieces: int = 384,
        syntax: str = "rulebase",
        add_prefix: Dict[str, str] = None,
        skip_id_regex: str = None,
        scramble_context: bool = False,
        use_context_full: bool = False,
        sample: int = -1,
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        self._max_pieces = max_pieces
        self._add_prefix = add_prefix
        self._scramble_context = scramble_context
        self._use_context_full = use_context_full
        self._sample = sample
        self._syntax = syntax
        self._skip_id_regex = skip_id_regex

        self.data = [dic for dic in self.read_generator(path)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.text_to_instance(**self.data[index])

    def read_generator(self, file_path: str):
        counter = self._sample + 1
        debug = 1
        is_done = False

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                if is_done:
                    break
                item_json = json.loads(line.strip())
                item_id = item_json.get("id", "NA")
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                if self._syntax == "rulebase":
                    questions = item_json["questions"]
                    if self._use_context_full:
                        context = item_json.get("context_full", "")
                    else:
                        context = item_json.get("context", "")
                elif self._syntax == "propositional-meta":
                    questions = item_json["questions"].items()
                    sentences = [x["text"] for x in item_json["triples"].values()] + [
                        x["text"] for x in item_json["rules"].values()
                    ]
                    if self._scramble_context:
                        random.shuffle(sentences)
                    context = " ".join(sentences)
                else:
                    raise ValueError(f"Unknown syntax {self._syntax}")

                for question in questions:
                    counter -= 1
                    debug -= 1
                    if counter == 0:
                        is_done = True
                        break
                    if debug > 0:
                        logger.debug(item_json)
                    if self._syntax == "rulebase":
                        text = question["text"]
                        q_id = question.get("id")
                        label = None
                        if "label" in question:
                            label = 1 if question["label"] else 0

                    yield {
                        "item_id": q_id,
                        "question_text": text,
                        "context": context,
                        "label": label,
                        "debug": debug,
                    }

    def text_to_instance(
        self,  # type: ignore
        item_id: str,
        question_text: str,
        label: int = None,
        context: str = None,
        debug: int = -1,
    ):

        encoding = self.tokenizer.encode_plus(
            text=question_text,
            text_pair=context,
            add_special_tokens=True,
            max_length=self._max_pieces,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        data = {
            "token_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "segment_ids": encoding["token_type_ids"],
        }
        metadata = {
            "id": item_id,
            "question_text": question_text,
            "tokens": [x for x in question_text.split()],
            "context": context,
        }
        assert label is not None
        if label is not None:
            data["label"] = label
            data["correct_answer_index"] = label

        data["metadata"] = metadata
        return data


if __name__ == "__main__":
    logger.debug("debug")
    backbone = "roberta-base"
    reader = RuleTakerDataset(path=test_data_path, pretrained_model=backbone)
    for i in range(20):
        logger.debug(reader[i])
