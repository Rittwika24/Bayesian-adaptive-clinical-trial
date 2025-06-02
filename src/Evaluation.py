import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm # For progress bars

# Import your custom modules
#from src.trial_simulator import ClinicalTrialSimulator
#from src.glm_model import BayesianGLM

# --- 1. Define Trial Parameters and True Effects ---
# These are the *true* underlying effects for your simulated world.
# A realistic scenario would have Drug1 being better than Placebo, and Drug2 being even better.
TRUE_PAIN_REDUCTION_PLACEBO = 5.0 # Mean pain reduction for placebo group
TRUE_TREATMENT_EFFECT_DRUG1 = 8.0 # Mean pain reduction for Drug Dose 1
TRUE_TREATMENT_EFFECT_DRUG2 = 12.0 # Mean pain reduction for Drug Dose 2 (stronger effect)

BASELINE_PAIN_MEAN = 7.0 # Average baseline pain score (e.g., on a 0-10 scale)
BASELINE_PAIN_STD = 1.5
BIOMARKER_MEAN = 50.0
BIOMARKER_STD = 10.0
NOISE_STD = 2.0 # Standard deviation of random noise in pain reduction

N_PATIENTS_PER_STAGE = 40 # Total patients per stage, divided by number of groups
N_STAGES = 4 # Maximum number of stages in the trial

# Adaptive Stopping Rules (Bayesian criteria)
SUCCESS_THRESHOLD = 0.95 # P(Drug1 > Placebo | data) > 0.95
FUTILITY_THRESHOLD = 0.05 # P(Drug1 > Placebo | data) < 0.05
# You can add more complex rules, e.g., for Dose 2 vs Dose 1, or multiple efficacy endpoints.

# --- 2. Initialize Trial Simulator ---
simulator = ClinicalTrialSimulator(
    n_patients_per_stage=N_PATIENTS_PER_STAGE,
    true_treatment_effect_drug1=TRUE_TREATMENT_EFFECT_DRUG1,
    true_treatment_effect_drug2=TRUE_TREATMENT_EFFECT_DRUG2,
    true_pain_reduction_placebo=TRUE_PAIN_REDUCTION_PLACEBO,
    baseline_pain_mean=BASELINE_PAIN_MEAN,
    baseline_pain_std=BASELINE_PAIN_STD,
    biomarker_mean=BIOMARKER_MEAN,
    biomarker_std=BIOMARKER_STD,
    noise_std=NOISE_STD
)

# --- 3. Simulate and Analyze the Adaptive Trial ---
# This part simulates *one* trial run. For robust results, you'd run this many times (e.g., 1000 simulations).

trial_data = pd.DataFrame()
trial_history = [] # To store decisions and probabilities at each stage

print("--- Starting Single Adaptive Trial Simulation ---")

for stage in range(1, N_STAGES + 1):
    print(f"\n--- Stage {stage} ---")
    new_data = simulator.generate_stage_data(stage_num=stage)
    trial_data = pd.concat([trial_data, new_data], ignore_index=True)

    print(f"Total patients enrolled: {len(trial_data)}")

    # Instantiate and fit Bayesian GLM
    glm_model = BayesianGLM(trial_data.copy()) # Pass a copy to avoid modifying original df for next stages
    print("Building and fitting Bayesian GLM...")
    trace = glm_model.fit_model(draws=2000, tune=1000, chains=4, target_accept=0.9) # Reduced draws for speed

    # Get Posterior Summary
    summary = glm_model.get_posterior_summary()
    # print(summary) # Uncomment to see full summary at each stage

    # Evaluate Treatment Effect for Decision Making (e.g., Drug1 vs Placebo)
    prob_drug1_better_than_placebo, mean_diff_drug1_placebo, hdi_drug1_placebo = \
        glm_model.get_treatment_effect_probability(group_of_interest='Drug1', comparison_group='Placebo')

    prob_drug2_better_than_placebo, mean_diff_drug2_placebo, hdi_drug2_placebo = \
        glm_model.get_treatment_effect_probability(group_of_interest='Drug2', comparison_group='Placebo')

    print(f"P(Drug1 > Placebo | data): {prob_drug1_better_than_placebo:.3f}")
    print(f"Mean Difference (Drug1 - Placebo): {mean_diff_drug1_placebo:.2f} (95% HDI: {hdi_drug1_placebo})")
    print(f"P(Drug2 > Placebo | data): {prob_drug2_better_than_placebo:.3f}")
    print(f"Mean Difference (Drug2 - Placebo): {mean_diff_drug2_placebo:.2f} (95% HDI: {hdi_drug2_placebo})")


    # --- Adaptive Stopping Rules ---
    decision = "Continue"
    if prob_drug1_better_than_placebo >= SUCCESS_THRESHOLD or prob_drug2_better_than_placebo >= SUCCESS_THRESHOLD:
        decision = "Success: Drug(s) found effective!"
        break
    elif prob_drug1_better_than_placebo <= FUTILITY_THRESHOLD and prob_drug2_better_than_placebo <= FUTILITY_THRESHOLD:
        decision = "Futility: No drug appears effective."
        break
    elif stage == N_STAGES:
        decision = "Completed Max Stages: No clear decision (requires further investigation or larger trial)."

    print(f"Trial Decision: {decision}")
    trial_history.append({
        'stage': stage,
        'patients_enrolled': len(trial_data),
        'prob_drug1_vs_placebo': prob_drug1_better_than_placebo,
        'prob_drug2_vs_placebo': prob_drug2_better_than_placebo,
        'decision': decision
    })

    if decision != "Continue":
        break

print("\n--- Trial Completed ---")
print(f"Final Decision: {decision}")
print(f"Total Patients: {len(trial_data)}")

# --- 4. Post-Trial Analysis and Visualization ---

# Plot posterior distributions of key parameters
print("\n--- Plotting Posterior Distributions ---")
az.plot_posterior(trace, var_names=['group_coeffs', 'beta_baseline_pain', 'beta_age', 'beta_sex', 'beta_biomarker', 'sigma'])
plt.suptitle('Posterior Distributions of Model Parameters')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('results/plots/posterior_distributions.png')
plt.show()

# Customize group_coeffs plot with actual names
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_posterior(trace, var_names=['group_coeffs'], ref_val=0, ax=ax)
# Manually relabel x-axis ticks if needed, based on glm_model.group_names order
ax.set_xticks(np.arange(len(glm_model.group_names)))
ax.set_xticklabels(glm_model.group_names)
ax.set_title('Posterior Distributions of Group Coefficients')
plt.tight_layout()
#plt.savefig('results/plots/group_coeffs_posterior.png')
plt.show()


# Plot a subset of MCMC traces for convergence diagnostics
print("\n--- Plotting MCMC Traces ---")
az.plot_trace(trace, var_names=['group_coeffs', 'intercept', 'sigma'])
plt.suptitle('MCMC Traces')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('results/plots/mcmc_traces.png')
plt.show()

# Visualize trial history
trial_history_df = pd.DataFrame(trial_history)
#print("\nTrial History:")
#print(trial_history_df)
#trial_history_df.to_csv('results/tables/trial_history.csv', index=False)

