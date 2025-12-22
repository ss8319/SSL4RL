# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the dataset to parquet format
"""

import argparse
import os
import datasets
from utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="our_datasets/MMBench_PositionQA")
    parser.add_argument("--data_source", default="datasets/MMBench_PositionQA")

    args = parser.parse_args()
    data_source = args.data_source

    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["test"]
    test_dataset = dataset["validation"]

    print(train_dataset[0])

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("query")
            prompt = problem
            answer = example.pop("answer")
            images = example.pop("images")

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    print("Successfully save datasets!")