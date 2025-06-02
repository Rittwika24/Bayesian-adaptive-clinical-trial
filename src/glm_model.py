import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import arviz as az

class BayesianGLM:
    def __init__(self, data_df):
        """
        Initializes the Bayesian GLM with the current accumulated trial data.

        Args:
            data_df (pd.DataFrame): DataFrame containing patient data up to current stage.
        """
        self.data = data_df
        self.model = None
        self.trace = None

    def build_model(self):
        """
        Builds the Bayesian Generalized Linear Model.
        Pain reduction is assumed to be normally distributed.
        Covariates: group (categorical), baseline_pain, age, sex, biomarker.
        """
        # Encode categorical variables
        self.data['group_encoded'] = self.data['group'].astype('category').cat.codes
        group_names = self.data['group'].astype('category').cat.categories.tolist()
        n_groups = len(group_names)

        self.data['sex_encoded'] = (self.data['sex'] == 'Female').astype(int) # 0 for Male, 1 for Female

        with pm.Model() as self.model:
            # Priors for intercept and coefficients
            intercept = pm.Normal('intercept', mu=0, sigma=10)

            # Group effects (treatment effects)
            # Use a hierarchical prior or non-centered parametrization for better sampling if many groups
            # For simplicity, using separate Normal priors here, with placebo as reference.
            # Assuming 'Placebo' is the first category after encoding (cat.codes sorts alphabetically by default)
            # If not, ensure you pick the correct reference category for interpretation
            # E.g., if groups are ['Drug1', 'Drug2', 'Placebo'], placebo might not be 0.
            # Make sure 'Placebo' is the reference level (coefficient is 0 for placebo, others are differences)
            # We explicitly define effects for each group and then use a contrast or index approach.

            # Alternative: One effect for reference, then diffs for others.
            # base_effect = pm.Normal('base_effect', mu=0, sigma=5) # For Placebo
            # drug1_effect_diff = pm.Normal('drug1_effect_diff', mu=0, sigma=5)
            # drug2_effect_diff = pm.Normal('drug2_effect_diff', mu=0, sigma=5)
            # group_effects = pm.Deterministic('group_effects', pt.stack([base_effect, base_effect + drug1_effect_diff, base_effect + drug2_effect_diff]))
            # Make sure the indexing matches your group_encoded values.

            # Simpler approach for now: separate effects for each group, will compare to placebo post-hoc
            group_coeffs = pm.Normal('group_coeffs', mu=0, sigma=5, shape=n_groups)

            # Covariate effects
            beta_baseline_pain = pm.Normal('beta_baseline_pain', mu=0, sigma=2) # Effect of baseline pain
            beta_age = pm.Normal('beta_age', mu=0, sigma=0.5) # Effect of age
            beta_sex = pm.Normal('beta_sex', mu=0, sigma=2) # Effect of sex
            beta_biomarker = pm.Normal('beta_biomarker', mu=0, sigma=2) # Effect of biomarker

            # Expected pain reduction (mu)
            mu = (intercept +
                  group_coeffs[self.data['group_encoded'].values] +
                  beta_baseline_pain * self.data['baseline_pain'].values +
                  beta_age * self.data['age'].values +
                  beta_sex * self.data['sex_encoded'].values +
                  beta_biomarker * self.data['biomarker'].values)

            # Standard deviation of the outcome
            sigma = pm.HalfNormal('sigma', sigma=5)

            # Likelihood (Observed pain reduction)
            pain_reduction_obs = pm.Normal('pain_reduction_obs', mu=mu, sigma=sigma,
                                           observed=self.data['pain_reduction'].values)

        self.group_names = group_names
        return self.model

    def fit_model(self, draws=2000, tune=1000, chains=2, target_accept=0.9):
        """
        Fits the Bayesian GLM model using NUTS sampler.
        """
        if self.model is None:
            self.build_model()

        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains,
                                   target_accept=target_accept,
                                   return_inferencedata=True,
                                   random_seed=42)
        return self.trace

    def get_posterior_summary(self):
        """
        Returns a summary of the posterior distributions.
        """
        if self.trace is None:
            raise ValueError("Model has not been fitted yet. Call fit_model() first.")
        return az.summary(self.trace)

    def get_treatment_effect_probability(self, group_of_interest='Drug1', comparison_group='Placebo'):
        """
        Calculates the posterior probability that group_of_interest is better than comparison_group.
        (Assumes higher pain reduction is better).
        """
        if self.trace is None:
            raise ValueError("Model has not been fitted yet. Call fit_model() first.")

        # Get the posterior samples for group coefficients
        group_coeffs_samples = self.trace.posterior['group_coeffs'].values

        # Find the indices of the groups
        try:
            idx_group_of_interest = self.group_names.index(group_of_interest)
            idx_comparison_group = self.group_names.index(comparison_group)
        except ValueError as e:
            print(f"Error: {e}. Check group names: {self.group_names}")
            return None

        # Calculate the difference in effects
        effect_difference = (group_coeffs_samples[:, :, idx_group_of_interest] -
                             group_coeffs_samples[:, :, idx_comparison_group])

        # Calculate the probability that the difference is positive (Drug1 > Placebo)
        prob_better = (effect_difference > 0).mean()

        # You can also return the mean and HPD interval of the difference
        mean_diff = effect_difference.mean()
        hdi_diff = az.hdi(effect_difference, hdi_prob=0.95)

        return prob_better, mean_diff, hdi_diff

    def get_parameter_hdi(self, param_name, hdi_prob=0.95):
        """
        Returns the HPD interval for a given parameter.
        """
        if self.trace is None:
            raise ValueError("Model has not been fitted yet. Call fit_model() first.")
        import arviz as az
        return az.hdi(self.trace.posterior[param_name], hdi_prob=hdi_prob)

    def get_parameter_posterior_mean(self, param_name):
        """
        Returns the posterior mean for a given parameter.
        """
        if self.trace is None:
            raise ValueError("Model has not been fitted yet. Call fit_model() first.")
        return self.trace.posterior[param_name].mean().item()
    