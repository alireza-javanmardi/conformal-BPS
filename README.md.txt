# Reproducing Results for Optimal Conformal Prediction under Epistemic Uncertainty

This repository contains the code necessary to reproduce the results presented in the paper.

---

## 🔧 Setup

Please create a virtual environment and install the required packages using:

```bash
pip install -r requirements.txt
```

---

## 📁 Folder Structure

* **experiments**: Contains all experiment scripts.
* **models**: Used to store pretrained or newly trained models.
* **training**: Model training scripts for CIFAR-10 and CIFAR-100.
* **epiuc**: Source code and utilities for models, uncertainty wrappers, evaluation routines, and other core components.

---

## 🧪 Section 6.1 – A Synthetic Experiment Given Valid Second-Order Predictions

Run the experiment with:

```bash
python illustration.py
```

---

## 🔄 Section 6.2 – Conformalized Credal Sets with Probabilistic Validity

Run the following script:

```bash
python sets_from_predictions_credal.py
```

**Inputs:**

1. Model name: `ensemble`
2. CP alpha
3. Beta
4. Calibration size
5. Random seed: a number between 0 to 9

To extract averaged results over seeds:

```bash
python results_from_sets_credal.py
```

**Inputs:**

1. Model name: `ensemble`
2. CP alpha
3. Beta
4. Calibration size

---

## 🌍 Section 6.3 – Real-World Experiments Beyond Valid Second-Order Predictions

### Step 1: Train or Use Pretrained Models

You can train models for each dataset and save them under the **models** folder.
Alternatively, use the pretrained models provided.

### Step 2: Generate Predictions

```bash
python predict_from_model.py
```

**Inputs:**

1. Data: `cifar10` or `cifar100`
2. Model: `ensemble`, `mc`, or `evidential`

### Step 3: Create Prediction Sets

```bash
python sets_from_predictions.py
```

**Inputs:**

1. Data: `cifar10` or `cifar100`
2. Model: `ensemble`, `mc`, or `evidential`
3. CP alpha
4. Calibration size
5. Random seed: a number between 0 to 9

### Step 4: Aggregate Results

```bash
python results_from_sets.py
```

**Inputs:**

1. Data: `cifar10` or `cifar100`
2. Model: `ensemble`, `mc`, or `evidential`
3. CP alpha
4. Calibration size

---

## 📊 Appendix D – A Synthetic Experiment

### Step 1: Generate Synthetic Data

```bash
python synthetic(APS_paper).py
```

**Inputs:**

1. Data size
2. Rarity parameter (probability that `X1 == 1`)
3. Random seed: 0

### Step 2: Generate Prediction Sets

```bash
python sets_from_predictions_synthetic_APS.py
```

**Inputs:**

1. Data size
2. Rarity parameter (probability that `X1 == 1`)
3. Random seed (for the data): 0
4. Model: `ensemble`, `mc`, or `evidential`
5. CP alpha
6. Calibration size
7. Random seed: a number between 0 to 9

### Step 3: Aggregate Results

```bash
python results_from_sets_synthetic_APS.py
```

**Inputs:**

1. Data size
2. Rarity parameter (probability that `X1 == 1`)
3. Random seed (for the data): 0
4. Model: `ensemble`, `mc`, or `evidential`
5. CP alpha
6. Calibration size

