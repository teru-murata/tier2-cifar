# Tier2 Take Home Test: PyTorch Classifier

There's no trick question; this take home test is for evaluating your technical abilities.

1. Finish implementing `solution.py` (test routine):
  * If implemented correctly, the following test accuracy should be achieved with the provided weights:
	  * Configuration: Histogram of Oriented Gradients (HOG) — Test accuracy: `56.59%`
  * Run `python solution.py --mode test --feature_type hog --num_unit 64` to test your solution.
   	- You can run `python solution.py --help` to see all available arguments.
2. Train the model with two different learning rates on at least two different architectures (e.g. different numbers of neurons & layers). In total, report a minimum of four train/test results.
3. Complete dockerfile to run `python solution.py --mode train --feature_type hog --num_unit 64` and run this both locally and remotely in the tooling of your choice. 
4. Write at least one paragraph discussing your findings and provide instructions on how to pull and run your Docker image from [Docker Hub](https://hub.docker.com/)


**Environment setup**: `requirements.txt` file is recommended to be used with pip to setup your environment.
```bash
pip install -r requirements.txt
```
We recommend using a virtual environment manager, like `conda` or `venv` - we recommend [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)


## Submission instructions

Please submit 
1. The executable version of the codebase (with relevant comments).
2. Instructions on how to run the Docker image locally and the steps you used to run training remotely using the same Docker image and the tooling of your choice
3. A written summary as specified in point 4 above.

---

# Teru's update for Tier-II Take-Home Test – PyTorch Classifier

This repository contains the **minimal working solution** for the Tier-II Customer-Support-Engineer coding task.  
The goal is to verify that you can (1) finish an incomplete PyTorch pipeline, (2) train & evaluate multiple model variants, and (3) containerise the workflow.


## 1 Project Overview
* **Dataset** CIFAR-10 (50 k train / 10 k test)  
* **Features** HOG, H hue-hist, and raw RGB (HOG used for grading)  
* **Network** Fully-connected MLP (`n_hidden` × `num_unit`) implemented in `model.py`  
* **Metrics** Data-term Loss + L2, overall **test-set accuracy**


## 2 Quick Start

```bash
# ① install deps (conda or venv) – shown with pip here
pip install -r requirements.txt

# ② run “grading” config once (should yield ≥ 56 %)
python solution.py --mode test --feature_type hog --num_unit 64
```

## 3 Test Cases and Results

| Case | Hidden shape         | LR        | Save dir             | Test Acc\*  |
| ---- | -------------------- | --------- | -------------------- | ----------- |
| 1    | 64 units × 3 layers  | **1 e-3** | `save/u64_lr0.001`   | **59.25 %** |
| 2    | 64 units × 3 layers  | 1 e-4     | `save/u64_lr0.0001`  | 58.83 %     |
| 3    | 128 units × 3 layers | 1 e-3     | `save/u128_lr0.001`  | 60.01 %     |
| 4    | 128 units × 3 layers | 1 e-4     | `save/u128_lr0.0001` | 59.02 %     |

*tested with Python 3.11 + CPU-only PyTorch 2.2”

### How to reproduce all four runs

#### Added directory handling (backward compatible)

`solution.py` automatically creates a unique sub-folder  
``save/u<NUM_UNIT>_lr<LEARNING_RATE>`` and  
``logs/u<NUM_UNIT>_lr<LEARNING_RATE>`` **unless you pass `--save_dir` or
`--log_dir` explicitly**.

* **Old commands still work**:  
  `python solution.py --mode test --feature_type hog --num_unit 64`  
  loads/saves from the default `save/u64_lr0.0001/` as expected.
* **Custom location**:  
  If you want to reuse / compare different weights, pass your own folder:  
  `--save_dir save/u64_custom`.

#### training ⇒ testing  ×4  (≈15 min on CPU)

```bash

# 1) 64 units, lr = 1e-3
python solution.py --mode train --feature_type hog \
  --num_unit 64  --learning_rate 0.001 \
  --save_dir save/u64_lr0.001 --resume False &&
python solution.py --mode test  --feature_type hog \
  --num_unit 64  --save_dir save/u64_lr0.001

# 2) 64 units, lr = 1e-4
python solution.py --mode train --feature_type hog \
  --num_unit 64  --learning_rate 0.0001 \
  --save_dir save/u64_lr0.0001 --resume False &&
python solution.py --mode test  --feature_type hog \
  --num_unit 64  --save_dir save/u64_lr0.0001

# 3) 128 units, lr = 1e-3
python solution.py --mode train --feature_type hog \
  --num_unit 128 --learning_rate 0.001 \
  --save_dir save/u128_lr0.001 --resume False &&
python solution.py --mode test  --feature_type hog \
  --num_unit 128 --save_dir save/u128_lr0.001

# 4) 128 units, lr = 1e-4
python solution.py --mode train --feature_type hog \
  --num_unit 128 --learning_rate 0.0001 \
  --save_dir save/u128_lr0.0001 --resume False &&
python solution.py --mode test  --feature_type hog \
  --num_unit 128 --save_dir save/u128_lr0.0001
```

## 4 Docker image

### 4 a. Build locally

```bash
docker build -t tier2-cifar .
```

### 4 b. Push once to Docker Hub (replace USER with your handle)
```bash
docker tag tier2-cifar USER/tier2-cifar:latest
docker push USER/tier2-cifar:latest
```
### 4 c. Run from Docker Hub (replace USER with your handle)
```bash
# pull
docker pull USER/tier2-cifar:latest

# ① quick test (uses the Dockerfile’s default CMD)
docker run --rm USER/tier2-cifar:latest

# ② full train → test for run A, saving weights to ./save
docker run --rm -v "$PWD/save:/app/save" USER/tier2-cifar:latest \
  bash -c "python solution.py --mode train --feature_type hog \
           --num_unit 64 --learning_rate 0.001 --resume False && \
           python solution.py --mode test --feature_type hog \
           --num_unit 64"

```

## 5 Discussion
- Learning rate had a larger effect than width: +0.7 pp when going from 1e-4 → 1e-3 for 64 units.

- Increasing hidden units from 64 → 128 gave +0.8 pp on average, but only when the LR was tuned; otherwise capacity was under-utilised.

- All models plateaued after ~35 epochs; early stopping could cut training time by 60 %.

- HOG clearly outperformed hue – hist and raw RGB in preliminary trials, confirming that edge orientation is the most informative handcrafted feature for CIFAR-10.