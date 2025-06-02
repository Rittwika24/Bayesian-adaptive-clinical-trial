import numpy as np
import pandas as pd

class ClinicalTrialSimulator:
    def __init__(self, n_patients_per_stage, true_treatment_effect_drug1, true_treatment_effect_drug2,
                 true_pain_reduction_placebo, baseline_pain_mean, baseline_pain_std,
                 biomarker_mean, biomarker_std, noise_std):
        """
        Initializes the clinical trial simulator with true underlying parameters.

        Args:
            n_patients_per_stage (int): Number of patients to enroll per stage.
            true_treatment_effect_drug1 (float): True average pain reduction due to Drug Dose 1.
            true_treatment_effect_drug2 (float): True average pain reduction due to Drug Dose 2.
            true_pain_reduction_placebo (float): True average pain reduction in placebo group.
            baseline_pain_mean (float): Mean of baseline pain scores.
            baseline_pain_std (float): Standard deviation of baseline pain scores.
            biomarker_mean (float): Mean of baseline biomarker.
            biomarker_std (float): Standard deviation of baseline biomarker.
            noise_std (float): Standard deviation of the noise in pain reduction (residual error).
        """
        self.n_patients_per_stage = n_patients_per_stage
        self.true_effects = {
            'Placebo': true_pain_reduction_placebo,
            'Drug1': true_treatment_effect_drug1,
            'Drug2': true_treatment_effect_drug2
        }
        self.baseline_pain_mean = baseline_pain_mean
        self.baseline_pain_std = baseline_pain_std
        self.biomarker_mean = biomarker_mean
        self.biomarker_std = biomarker_std
        self.noise_std = noise_std
        self.patient_id_counter = 0

    def generate_stage_data(self, stage_num):
        """
        Generates simulated data for a single stage of the trial.
        Patients are equally split among groups for simplicity.
        """
        n_groups = len(self.true_effects)
        n_patients = self.n_patients_per_stage # // n_groups # Adjust if not equal split

        data = []
        for group_idx, group_name in enumerate(self.true_effects.keys()):
            # Simulate n_patients for each group
            current_n_patients = n_patients
            # Assign unique Patient IDs
            patient_ids = [f"P{self.patient_id_counter + i}" for i in range(current_n_patients)]
            self.patient_id_counter += current_n_patients

            # Simulate baseline characteristics
            baseline_pain = np.random.normal(self.baseline_pain_mean, self.baseline_pain_std, current_n_patients)
            age = np.random.normal(55, 10, current_n_patients).clip(20, 80) # Age between 20 and 80
            sex = np.random.choice(['Male', 'Female'], size=current_n_patients)
            biomarker = np.random.normal(self.biomarker_mean, self.biomarker_std, current_n_patients).clip(0) # Non-negative biomarker

            # Simulate outcome (pain reduction)
            # Basic linear model for pain reduction
            # Outcome = True_Effect_Group + beta_pain*Baseline_Pain + beta_age*Age + beta_biomarker*Biomarker + noise
            # For simplicity, let's just use true group effect + noise for now,
            # and let the GLM learn effects of covariates.
            # True pain reduction values will be around (e.g.) 5 for placebo, 10 for drug1, 15 for drug2

            # True underlying model for pain reduction
            # Example: Pain_Reduction = Intercept + Group_Effect + 0.1 * Baseline_Pain - 0.05 * Age + 0.02 * Biomarker + Noise
            # To simplify simulation, let's just use the group effects and noise for the direct outcome.
            # The GLM will then estimate the effects of covariates.
            # This makes the simulation independent of the GLM's covariate structure for outcome generation.

            # For a more sophisticated simulation, you'd specify beta coefficients for covariates:
            # true_beta_pain = 0.1
            # true_beta_age = -0.05
            # true_beta_biomarker = 0.02
            # expected_pain_reduction = self.true_effects[group_name] + \
            #                           true_beta_pain * (baseline_pain - self.baseline_pain_mean) + \
            #                           true_beta_age * (age - 55) + \
            #                           true_beta_biomarker * (biomarker - self.biomarker_mean)

            expected_pain_reduction = self.true_effects[group_name] # Simple direct effect for simulation

            pain_reduction = expected_pain_reduction + np.random.normal(0, self.noise_std, current_n_patients)

            for i in range(current_n_patients):
                data.append({
                    'patient_id': patient_ids[i],
                    'stage': stage_num,
                    'group': group_name,
                    'baseline_pain': baseline_pain[i],
                    'age': age[i],
                    'sex': sex[i],
                    'biomarker': biomarker[i],
                    'pain_reduction': pain_reduction[i]
                })

        return pd.DataFrame(data)