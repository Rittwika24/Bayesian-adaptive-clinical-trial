# Bayesian Adaptive Clinical Trial Design & Analysis

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyMC Version](https://img.shields.io/badge/PyMC-~5.0-red.svg)](https://www.pymc.io/)
[![ArviZ Version](https://img.shields.io/badge/ArviZ-~0.17-green.svg)](https://arviz-devs.github.io/arviz/)

---

## 1. Project Overview

This project demonstrates the design, simulation, and Bayesian analysis of an adaptive Phase II clinical trial for a hypothetical new analgesic drug for chronic pain reduction. It showcases a robust application of statistical modeling and experimental design, emphasizing data-driven decision-making through Bayesian inference in a pharmaceutical research context.

The primary goal is to illustrate how adaptive trial designs, powered by Bayesian methods, can lead to more efficient and ethical drug development by allowing for early stopping for efficacy or futility based on accumulating evidence.

## 2. Key Features & Skills Demonstrated

* **Bayesian Analysis (PyMC & ArviZ):** Application of Bayesian statistical methods for robust parameter estimation, uncertainty quantification, and direct probability statements about treatment effects.
* **Generalized Linear Models (GLM):** Implementation and fitting of a Normal GLM to model continuous outcomes (pain reduction) while accounting for multiple covariates (treatment group, baseline pain, age, sex, biomarker).
* **Adaptive Trial Design:** Simulation of a multi-stage adaptive trial with interim analyses and predefined Bayesian stopping rules (for success or futility).
* **Data Simulation:** Creation of realistic synthetic clinical trial data incorporating various patient characteristics and true treatment effects.
* **Hypothesis Testing:** Bayesian hypothesis testing to quantify the probability of a drug being superior to placebo or to other drugs.
* **Python Programming:** Modular and object-oriented Python development for building reusable simulation and modeling components.
* **Statistical Interpretation:** Drawing actionable insights from posterior distributions, credible intervals, and treatment effect probabilities.

## 3. Problem Statement / Background

Traditional fixed-design clinical trials can be inefficient, rigid, and sometimes ethically challenging, especially if a treatment is clearly effective or ineffective partway through the trial. Bayesian adaptive designs offer a flexible alternative by allowing trial characteristics (like sample size or treatment arm allocation) to be modified during the trial based on accumulating data. This can lead to:

* **Increased Efficiency:** Reaching conclusions faster, potentially reducing trial duration and cost.
* **Enhanced Ethics:** Minimizing patient exposure to ineffective treatments or accelerating access to highly effective ones.
* **Improved Decision Making:** Providing more comprehensive probabilistic evidence for decision-makers at interim stages.

This project simulates such an adaptive design to demonstrate these advantages in a practical scenario.

## 4. Methodology

### a. Data Generation (Trial Simulation)

The `ClinicalTrialSimulator` class in `src/trial_simulator.py` generates synthetic patient data. Key aspects of the simulated data include:
* **Patient Characteristics:** Age, sex, baseline pain scores, and a continuous biomarker.
* **Treatment Arms:** Placebo, Drug A, Drug B (or similar nomenclature).
* **Primary Outcome:** Pain reduction, modeled as a continuous variable with a Normal distribution. The true mean pain reduction for each treatment arm and the influence of patient covariates are predefined.

### b. Bayesian Generalized Linear Model (GLM)

The core statistical model is a Bayesian GLM, implemented in the `BayesianGLM` class in `src/glm_model.py`. The pain reduction ($Y_k$) for patient $k$ is modeled as:

$$Y_k \sim \text{Normal}(\mu_k, \sigma^2)$$

Where the linear predictor $\mu_k$ is defined as:
$$\mu_k = \text{intercept} + \beta_{\text{group}}[\text{group}_k] + \beta_{\text{baseline\_pain}} \cdot \text{baseline\_pain}_k + \beta_{\text{age}} \cdot \text{age}_k + \beta_{\text{sex}} \cdot \text{sex\_encoded}_k + \beta_{\text{biomarker}} \cdot \text{biomarker}_k$$

* **Priors:** Weakly informative Normal priors are placed on all regression coefficients (intercept, group effects, covariate effects). A Half-Normal prior is used for the standard deviation ($\sigma$).
* **Inference:** Markov Chain Monte Carlo (MCMC) sampling, specifically the No-U-Turn Sampler (NUTS) implemented in `PyMC`, is used to draw samples from the posterior distributions of all model parameters.

### c. Adaptive Decision Rules

At predefined interim analysis points (e.g., after each stage of patient enrollment):
* The Bayesian GLM is fitted to the accumulated data.
* Posterior probabilities for efficacy are calculated (e.g., $P(\text{Drug A effect} > \text{Placebo effect} | \text{data})$).
* **Stopping Rules:**
    * **Efficacy:** If the probability of a drug being superior to placebo exceeds a high threshold (e.g., $> 0.95$), the trial might stop early for efficacy for that drug.
    * **Futility:** If the probability of a drug being sufficiently better than placebo falls below a low threshold (e.g., $< 0.10$), the trial might stop early for futility for that drug.
* The trial continues if no stopping rule is met.

## 5. Project Structure

The project is organized into modular components for clarity and reusability:
