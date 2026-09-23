# Decision Tree Classification (Entropy-Based)

This project trains a **Decision Tree Classifier** for **multi-class classification**. The model learns a hierarchy of decision rules that split the data on feature values to predict a target class.

Unlike linear models that learn coefficients, the tree structure itself is the learned model.

---

## Problem Type

- **Learning type**: supervised
- **Task**: multi-class classification
- **Model**: `DecisionTreeClassifier`
- **Splitting criterion**: entropy (information gain)

---

## How a Decision Tree Works

A decision tree predicts a class by asking a sequence of yes/no questions (IF–THEN rules). During training, it learns which questions to ask and in what order, by comparing all possible thresholds of all features and picking the split that most reduces class uncertainty. This repeats until no useful split is left.

Example:
```
IF feature_4 <= 14.0:
    IF feature_2 <= 0.5:
        predict Class A
    ELSE:
        predict Class B
ELSE:
    predict Class C
```

Each internal node is a decision rule, and each leaf is a final prediction.

---

## Building the Tree

1. **Start with all training rows** at the root. Classes are mixed, so uncertainty is high.

2. **Measure uncertainty with entropy:**

   ```
   H = - Σ p_i log2(p_i)
   ```

   where p_i is the proportion of class i in the node. Entropy is 0 when all samples in a node belong to one class.

3. **Try all possible splits.** For each feature and candidate threshold, split rows into left/right groups and compute the weighted entropy.

4. **Pick the split with the highest information gain:**

   ```
   IG = H_parent − (n_left / n) * H_left − (n_right / n) * H_right
   ```

   This gives a rule like `feature_k <= threshold`.

5. **Split the rows.** Rows that satisfy the rule go left, the rest go right.

6. **Repeat** on each child node.

7. **Stop** when a node is pure, the maximum depth is reached, no split improves information gain, or too few samples remain.

Note that the tree splits rows (samples), not features or classes.

---

## Reading the Tree

Each node shows:

- **Decision rule**, e.g. `x[4] <= 14.027`
- **Entropy**: remaining uncertainty
- **Samples**: number of training rows at the node
- **Value**: class counts, e.g. `[15, 11, 13, 38, 63]`

The prediction at a node is the class with the highest count.

Example leaf:
```
entropy = 0.0
samples = 28
value = [0, 0, 0, 28, 0]
```

All 28 samples belong to one class, so any new sample that reaches this leaf is predicted as that class.

---

## Prediction

To classify a new sample, start at the root, apply the rule, move left or right, and repeat until a leaf is reached. The leaf's class is the prediction.

---

## Model Configuration

- Criterion: `entropy`
- Maximum depth limited to prevent overfitting

---

## Characteristics

- Interpretable
- Non-linear
- No feature scaling needed
- Overfits without depth control

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
