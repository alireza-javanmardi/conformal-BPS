# Optimal Conformal Prediction under Epistemic Uncertainty

[![arXiv](https://img.shields.io/badge/arXiv-2505.19033-b31b1b.svg)](https://arxiv.org/abs/2505.19033)

This repository contains the code for the paper: [Optimal Conformal Prediction under Epistemic Uncertainty](PAPER_LINK_PLACEHOLDER),
written by Alireza Javanmardi, Soroush H. Zargarbashi, Santo M. A. R. Thies, Willem Waegeman, Aleksandar Bojchevski, and Eyke Hüllermeier.
This paper is published at the Conference on Uncertainty in Artificial Intelligence (UAI) 2026.

This paper studies conformal prediction (CP) when the underlying model is a credal set predictor that outputs a set of label distributions for each input, representing both aleatoric and epistemic uncertainty. We introduce **Bernoulli Prediction Sets (BPS)**, a method that takes a credal set as input and outputs a set of labels. Depending on what is known about the credal sets, BPS provides different guarantees:

1. **Valid credal sets.** When the credal sets are valid, the resulting prediction sets satisfy a conditional coverage guarantee. No calibration is required.
2. **Partially-valid credal sets.** When the credal sets are only partially valid, the resulting prediction sets satisfy a PAC-style conditional coverage guarantee. No calibration is required.
3. **Unknown validity.** When the validity of the credal sets is unknown, we apply calibration with first-order data to achieve a PAC-style conditional coverage guarantee.

Here is the general procedure of how BPS maps a credal set to a prediction set:

![BPS overview](3%20cases.png "general procedure of BPS for credal sets")

## Setup

1. Clone the repository
2. Create a new virtual environment and install the requirements:
```shell
 pip install -r requirements.txt
```
## Repository structure

### `experiments/`

- **`real_credal_predictions`** — construct credal sets for a given dataset, a model (`CreRL` or `CreEns`), a model seed, and a parameter `gamma`.
- **`real_sets_from_credal`** — construct BPS-based prediction sets given the credal sets, for a given dataset, a model (`CreRL` or `CreEns`), a model seed, the parameter `gamma`, the conformal prediction `alpha`, the desired conditional coverage, and the conformal calibration–test split seed.
- **`read_APS_sets_from_credal`** — same as `real_sets_from_credal`, but constructs APS-based prediction sets.
- **`real_sets_from_conformal_credal`** — construct conformalized credal sets and then the BPS-based and APS-based prediction sets given the conformalized credal sets, for a given dataset, a model (`CreRL` or `CreEns`), a model seed, the parameter `gamma`, the conformal credal set `epsilon`, the desired conditional coverage, and the (conformal credal) calibration–test split seed.

### `notebooks/`

- **`valid_credals.ipynb`** — create synthetic valid credal sets and apply BPS and APS on top of them to obtain the prediction sets.
- **`toy_example.ipynb`** — Toy Example 1 in the paper, comparing the adaptivity of BPS to APS with respect to epistemic uncertainty.
- **`plots.ipynb`** — generate all the plots used in the paper.
- **`tables.ipynb`** — generate all the tables used in the paper.
## Citation

If you use this code, please cite our paper:
```
@inproceedings{javanmardi2026optimal,
      title     = {Optimal Conformal Prediction under Epistemic Uncertainty},
      author    = {Javanmardi, Alireza and Zargarbashi, Soroush H. and Thies, Santo M. A. R. and Waegeman, Willem and Bojchevski, Aleksandar and H{\"u}llermeier, Eyke},
      booktitle = {Proceedings of the Forty-Second Conference on Uncertainty in Artificial Intelligence (UAI)},
      series    = {Proceedings of Machine Learning Research},
      year      = {2026},
      publisher = {PMLR}
}
```