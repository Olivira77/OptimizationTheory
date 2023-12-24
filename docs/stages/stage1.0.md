# Detecting Domain Shifts

## Concepts

### Domain Shift

- Probabilities 
  - Source Distribution $p(x, y)$
  - Target Distribution $q(x, y)$
- Data
  - Training Examples $(x_1, y_1), …, (x_n, y_n) \sim p(x, y)$
  - Test Examples $(x'_1, …, x'_n) \sim q(x)$
- Problem
  - $p(x,y) \ne q(x,y)$

- Objective
  - Predict well on the test distribution, **WITHOUT** seeing any labels $y_i \sim q(y)$

#### Label Shift / Prior probability shift

- $p(x|y) = q(x|y)$ and $p(y) \ne q(y)$
- Makes anti-causal assumption, (y causes x) [Diseases cause symptoms]
- But how can we estimate $q(y)$ without any samples $y_i \sim q(y)$?
- Detection: Black Box Shift Detection (BBSD)

#### Covariate Shift

- $p(y|x) = q(y|x)$ and $p(x) \ne q(x)$
- Implicitly assumes that x causes y [While symptoms don't cause diseases]
- Appealing because we have samples $x_i \sim p(x)$ and $x'_j \sim q(x)$
- Natural to estimate $q(x)/p(x)$ -> use for importance-weighted ERM
- Detection: Refers to 'Failing Loudly'

#### Concept Shift / Concept Drift

- $p(y|x)\ne q(y|x) \; \text{and} \; p(x) = q(x)$
- or $p(x|y) \ne q(x|y) \; \text{and} \; p(y) = q(y)$

#### Dataset Shift

- None of the above hold.
- $p(x,y) \ne q(x,y)$

### Black Box Shift Detection 

#### Assumptions

1. The label shift assumption 
   $$
   p(x|y)=q(x|y), \forall x \in \mathcal X, y \in \mathcal Y
   $$

2. For every $y \in \mathcal Y$ with $q(y) > 0$ we require $p(y) > 0$

3. Access to a black box predictor $f : \mathcal X \rightarrow \mathcal Y$ where the expected confusion matrix $C_p(f)$ is invertible 

$$
C_p(f) := p(f(x), y) \in R^{|\mathcal Y| \times |\mathcal Y|}
$$

#### Pipeline

1. Train a Black-Box model ($f$) on the train data.
2. Generate predictions on train ($p(\hat y)$) and test data ($q(\hat y)$).
3. Utilize 2-sample tests on $p(\hat y)$ and $q(\hat y)$.

### 2-Sample Kolmogorov-Smirnov (KS) Test + Bonferroni Correction

#### 2-Sample Kolmogorov-Smirnov (KS) Test

1. **Hypotheses:**
   - **Null Hypothesis (H0):** The two samples come from the same continuous distribution.
   - **Alternative Hypothesis (H1):** The two samples do not come from the same continuous distribution.
2. **Test Statistic Calculation:**
   - The KS test statistic is the maximum vertical distance (D) between the cumulative distribution functions (CDFs) of the two samples.
   - Mathematically, $D=max|F_1(x)-F_2(x)|$, where $F_1(x)$ and $F_2(x)$ are the CDFs of the two samples.
3. **Critical Value or P-value:**
   - The KS test compares the calculated test statistic (D) to a critical value from the Kolmogorov-Smirnov distribution or provides a p-value.
   - If the p-value is below a chosen significance level (commonly 0.05), the null hypothesis is rejected.
4. **Interpretation:**
   - If the null hypothesis is rejected, it suggests that there is a significant difference between the distributions of the two samples.

#### Bonferroni Correction

1. **Multiple Testing Problem:**
   - When conducting multiple statistical tests simultaneously (e.g., testing several hypotheses or comparing multiple groups), the likelihood of obtaining at least one significant result purely by chance increases.
2. **Hypotheses and Significance Level:**
   - Assume you are testing *k* hypotheses with a significance level (*α*) of, for example, 0.05.
3. **Individual Test Significance Level:**
   - Without correction, you would test each hypothesis at the significance level of $\frac \alpha k$ to maintain an overall significance level of *α*.
4. **Bonferroni Correction Formula:**
   - The Bonferroni Correction adjusts the significance level for each individual test. The corrected significance level ($α_{corr}$) is calculated as $α_{corr}=\frac \alpha k$.
5. **Interpretation:**
   - For each test, compare the p-value to the adjusted significance level.
   - If the p-value is less than or equal to the adjusted significance level, you reject the null hypothesis for that specific test.
6. **Conservative Nature:**
   - The Bonferroni Correction is conservative, meaning it reduces the chance of Type I errors (false positives) but increases the likelihood of Type II errors (false negatives).
7. **Use Case:**
   - Commonly applied in situations with multiple pairwise comparisons or when conducting multiple statistical tests to maintain an overall family-wise error rate.

```python
# Assuming 'p_values' is a list/array of p-values from multiple tests
alpha = 0.05
k = len(p_values)

# Bonferroni Correction
alpha_corr = alpha / k

# Compare each p-value to the corrected significance level
reject_null_hypothesis = [p <= alpha_corr for p in p_values]
```

## Pipeline

### Method 1

1. Train a Deep Neural Network (Black Box Predictor) on the train data.
2. Generate logits (output of the Neural Network) on train and test data.
3. Generate predictions ($argmax(logits)$) on train ($p(\hat y)$) and test data ($q(\hat y)$).
4. Utilize two-sample Kolmogorov-Smirnov test to test label shift.

    ```python
    from scipy.stats import ks_2samp
    ks_statistic, p_value = ks_2samp(data1, data2)
    ```

5. Obtain the result whether $p(\hat y) \ne q(\hat y) \Rightarrow$ Label Shift holds.
6. According to 'Failing Loudly', conduct a `2-Sample Kolmogorov-Smirnov (KS) Test + Bonferroni Correction` on the Softmaxed C-dimension logits, and get the result whether $p(x) \ne q(x) \Rightarrow$ Covariate Shift holds.
7. If Label Shift and Covarite Shift hold, then Concept Shift can't hold.

### Method 2 (deprecated)

Domain Classifier

1. Label all the train data with 0 and test data with 1.
2. Split both train and test data into 2 halves.
3. Train a classifier (AdaBoostClassifier) on the first half of the data.
4. Test on the other half of the data, obtain the prediction accuracy `acc`.
5. Doing Binomial Testing: $H_0: acc = 0.5 \; vs \; H_1: acc \ne 0.5$. Under the null hypothesis, $acc \sim \text {Bin}(N_{held-out, 0.5})$. 

```python
from scipy.stats import binomtest

# Example data
x = 55  # Number of successes
N = 100  # Total number of trials
p_value = binom_test(x, n=N, p=0.5, alternative='two-sided')

# Make a decision based on the p-value
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

