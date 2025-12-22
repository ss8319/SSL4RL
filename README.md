<p align="center">
  <img src="logo.png" alt="SSL4RL Logo" width="400">
</p>

# SSL4RL: Revisiting Self-supervised Learning as Intrinsic Reward for Visual-Language Reasoning

<a target="_blank" href="https://arxiv.org/pdf/2510.16416">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="#">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://huggingface.co/collections/PKU-ML/ssl4rl">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>

<br>
<span>
Xiaojun Guo<sup>*</sup>,
Runyu Zhou<sup>*</sup>,
<a class="name" target="_blank" href="https://yifeiwang77.com/">Yifei Wang<sup>*</sup></a>,
Qi Zhang,
Chenheng Zhang,
<a class="name" target="_blank" href="https://people.csail.mit.edu/stefje/">Stefanie Jegelka</a>,
Xiaohan Wang,
Jiajun Chai,
Guojun Yin,
Wei Lin,
<a class="name" target="_blank" href="https://yisenwang.github.io/">Yisen Wang<sup>&Dagger;</sup></a>
<br>
<sup>*</sup>Equal Contribution.
<sup>&Dagger;</sup>Correspondence.
</span>

## 📊 Overview

We propose ***SSL4RL***, a novel framework that leverages self-supervised learning (SSL) tasks as a source of verifiable rewards for RL-based fine-tuning. Our approach reformulates SSL objectives—such as predicting image rotation or reconstructing masked patches—into dense, automatic reward signals, eliminating the need for human preference data or unreliable AI evaluators. Experiments show that SSL4RL substantially improves performance on both **vision-centric** and **vision-language reasoning benchmarks**, with encouraging potentials on **open-ended image-captioning tasks**. Through systematic ablations, we identify key factors—such as **data volume, model scale, model choice, task difficulty, and semantic alignment with the target domain** — that influence the effectiveness of SSL4RL tasks, offering new design principles for future work. We also demonstrate the framework’s generality by applying it to graph learning, where it yields significant gains. SSL4RL establishes a versatile and effective paradigm for aligning multimodal models using verifiable, self-supervised objectives.

## 📌 Key Takeaways

1️⃣ **SSL as Intrinsic Reward Sharpens VLM Reasoning**. The SSL4RL paradigm demonstrably enhances vision-language reasoning by repurposing SSL tasks as intrinsic rewards. It deepens the perception and understanding of the image itself, leading towards more precise visual attention and less language bias.

2️⃣ **Task Choice is Critical**. SSL tasks show effectiveness when their inherent semantic aligns with core reasoning skills, while an inappropriate task may induce negative transfer and hinder downstream performance.

3️⃣ **Goldilocks Principle of Task Difficulty**. The effectiveness of an SSL task is contingent on its difficulty being appropriately matched to the model's capacity. Insufficient challenge provides a weak learning signal, while excessive difficulty leads to negative transfer.

4️⃣ **Non-additivity of Rewards**. A naive combination of multiple SSL rewards does not yield cumulative improvements, indicating potential optimization conflicts and underscoring the need for sophisticated integration strategies rather than simple averaging.

## 🔥 Open-source Collections

Our models are released in the huggingface collection [PKU-ML/SSL4RL](https://huggingface.co/collections/PKU-ML/ssl4rl):

- `SSL4RL-MMbench-Position-3B`: [PKU-ML/SSL4RL-MMbench-Position-3B](https://huggingface.co/PKU-ML/SSL4RL-MMBench-Position-3B)
- `SSL4RL-MMbench-Rotation-3B`: [PKU-ML/SSL4RL-MMbench-Rotation-3B](https://huggingface.co/PKU-ML/SSL4RL-MMBench-Rotation-3B)
- `SSL4RL-MMbench-Jigsaw-3B`: [PKU-ML/SSL4RL-MMbench-Jigsaw-3B](https://huggingface.co/PKU-ML/SSL4RL-MMBench-Jigsaw-3B)
- `SSL4RL-MMbench-Contrastive-3B`: [PKU-ML/SSL4RL-MMbench-Contrastive-3B](https://huggingface.co/PKU-ML/SSL4RL-MMBench-Contrastive-3B)

## 🚀 Environment Setups

Our implementation is based on the library [verl 0.3.0](https://github.com/volcengine/verl/tree/v0.3.x) developed by ByteDance Seed team.

1. Requirements:

    - **Python**: Version >= 3.9
    - **CUDA**: Version >= 12.1

2. For installing the dependencies, we recommend to use a fresh new conda environment:

    ```bash
    conda create -n verl python==3.10
    conda activate verl
    ```

3. Execute the install script.

    ```bash
    git clone https://github.com/PKU-ML/SSL4RL.git
    cd SSL4RL
    bash scripts/install_vllm_sglang_mcore.sh
    ```

    We provide the version of key required packages here:

    ```markdown
      - accelerate==1.8.1
      - datasets==4.0.0
      - flash-attn==2.7.4.post1
      - pyarrow==20.0.0
      - qwen-vl-utils==0.0.11
      - tokenizers==0.21.1
      - torch==2.6.0
      - torchvision==0.21.0
      - transformers==4.52.4
      - verl==0.3.0.post1
      - vllm==0.8.5.post1
      - xformers==0.0.29.post2
    ```

4. Install our package with some lightweight dependencies in setup.py:

    ```bash
    pip3 install -e .
    ```

If you encounter any issues during installation, please refer to the [Installation Guide](https://verl.readthedocs.io/en/v0.3.x/start/install.html) provided by Verl. If problems persist, don’t hesitate to [report them to us](https://github.com/PKU-ML/SSL4RL/issues).

## 🎯 Build SSL4RL Tasks

We provide the codes for building SSL4RL tasks, including Position, Rotation, Contrastive, and Jigsaw.

Take the MMBench as an example:

1. Download the MMBench benchmark from [HuggingFaceM4/MMBench](https://huggingface.co/datasets/HuggingFaceM4/MMBench) and save it to local directorys `datasets/MMBench`.

2. To build the SSL4RL dataset, execute the codes provided in `build_benchmarks`. Each code will generate an SSL4RL dataset that can be directedly load with `datasets.load_dataset`. By default, the dataset will be saved in the directory `datasets`. You can assign the target directory by changing the `save_dir` in the code.

```bash
cd build_benchmarks

## For Position Task
python makeposition_mmbench.py
## For Rotation Task
python makerotation_mmbench.py
## For Jigsaw Task
python makejigsaw_mmbench.py
## For Contrastive Task
python makecontrastive_mmbench.py
```

We provide the download link of benchmarks used in the paper here:

- MMBench: [HuggingFaceM4/MMBench](https://huggingface.co/datasets/HuggingFaceM4/MMBench)
- SEED-Bench: [Opencompass/SEEDBench](https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv)
- BLINK: [BLINK-Benchmark/BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)
- MME-RealWorld: [yifanzhang114/MME-RealWorld-Lite](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite)
- MMStar: [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar)
- RealWorldQA: [Opencompass/RealWorldQA](https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv)
- ImageNet: [PKU-ML/ImageNet-Subset](https://huggingface.co/datasets/PKU-ML/ImageNet-Subset)
- CapArena: [NJU-Desk/CapArena](https://box.nju.edu.cn/f/a79c42c9c10e4acb83e7/)

## 🧩 Running Reinforcement Learning

Follow these steps to reproduce our SSL4RL implementation:

1. **Preprocess the dataset for RL training.** Run the preprocessing script to convert the dataset format:
  
    ```bash
    cd verl

    python preprocess.py --data_source datasets/MMBench_PositionQA  --local_dir our_datasets/MMBench_PositionQA
    ```
  
    - `data_source`: The directory of SSL4RL datasets.
    - `local_dir`: Output directory for the processed dataset.

2. **Launch RL Training.**

   Execute the training script (Position as an example):

   ```bash
   bash run_mmbench_position.sh
   ```

   **Configuration notes for the training script:**

    - `SAVE_DIR`: Output directory for the trained model.
    - `train_path` and `test_path`: Paths to the processed dataset.
    - Logging: Defaults to [Tensorboard](https://www.tensorflow.org/tensorboard). To use [Weights & Biases](https://wandb.ai/site/), set `trainer.logger = ['console','wandb']`.
    - `trainer.n_gpus_per_node`: Your actual GPU count.
    - Our paper used 8×A800 GPUs. For limited GPU resources, reduce these parameters (may affect performance):

      ```bash
      actor_rollout_ref.actor.ppo_mini_batch_size
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu
      ```
  
3. **Convert RL-trained Checkpoints to HuggingFace Format**

   Merge the model checkpoints into a HuggingFace-compatible format:

   ```bash
   python scripts/merge_models.sh
   ```

## 🌊 Inference and Evaluation

We utilize [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/v0.1) to evaluate SSL4RL models.

1. **Install VLMEvalKit**

    ```bash
      git clone https://github.com/open-compass/VLMEvalKit.git
      cd VLMEvalKit
      pip install -e .
    ```

2. **Customize Prompt**

    To accurately extract the answer, we customize the prompt in `vlmeval/dataset/image_mcq.py`

    > Answer this multiple-choice question. Think step by step before answering. The last line of your response should be of the following format: \<think\>step-by-step reasoning\</think\> \<answer\>$LETTER\</answer\>, where LETTER is one of the options.

3. **Write Configs**

    Add configuration files to the `configs` directory.

    ```json
    {
        "model": {
          "ssl4rl_qwen_3b_mmbench_position_step300": {
            "class": "Qwen2VLChat",
            "model_path": "our_models/ssl4rl_qwen_3b_mmbench_position_step300",
            "min_pixels": 35840,
            "max_pixels": 12845056,
            "use_custom_prompt": false
          }
        },
        "data": {
          "MMBench": {
            "class": "ImageMCQDataset",
            "dataset": "MMBench_DEV_EN"
          }
        }
      }
    ```

4. **Run Evaluation**

   We provide evaluation scripts as below:

    ```bash
    torchrun --nproc-per-node=8 run.py \
      --config configs/mmbench_position.json \
      --work-dir eval_results/SSL4RL \
      --mode infer \
    ```

## 🎨 Customization Guide

To adapt SSL4RL for your needs, we recommend modifying these key files from our verl-based implementation:

```markdown
verl/utils/                # Supporting utilities
  - reward_score/__init__.py   # Reward normalization/scaling
  - reward_score/ssl4rl.py      # Scoring metrics
```

## Citation

If you find this work useful, please give us a free cite:

```bibtex
@article{guo2025ssl4rl,
  title={SSL4RL: Revisiting Self-supervised Learning as Intrinsic Reward for Visual-Language Reasoning},
  author={Guo, Xiaojun and Zhou, Runyu and Wang, Yifei and Zhang, Qi and Zhang, Chenheng and Jegelka, Stefanie and Wang, Xiaohan and Chai, Jiajun and Yin, Guojun and Lin, Wei and others},
  journal={arXiv preprint arXiv:2510.16416},
  year={2025}
}
```
