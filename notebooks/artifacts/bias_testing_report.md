# Content Moderation Bias Testing Report
Model: roberta-base
Test Date: 2025-12-28
Test Set Size: 31915 samples

## 1. Identity Term Bias

**Overall False Positive Rate:** 4.00%

Tested {len(bias_test_cases)} neutral statements containing identity terms.

### Results by Category:

- **Race**: 0.00% false positive rate
- **Religion**: 0.00% false positive rate
- **Gender**: 0.00% false positive rate
- **Sexuality**: 20.00% false positive rate
- **Nationality**: 0.00% false positive rate

### Highest Risk Identities:

- **lesbian**: 40.00% (2/5 flagged)
- **gay**: 40.00% (2/5 flagged)
- **heterosexual**: 20.00% (1/5 flagged)
- **bisexual**: 20.00% (1/5 flagged)
- **African American**: 0.00% (0/5 flagged)

## 2. False Positive Analysis

Analysis of incorrectly flagged content:

- **Toxic**: 495 false positives (1.71% of true negatives)
- **Severe Toxic**: 286 false positives (0.90% of true negatives)
- **Obscene**: 344 false positives (1.14% of true negatives)
- **Threat**: 58 false positives (0.18% of true negatives)
- **Insult**: 444 false positives (1.46% of true negatives)
- **Identity Hate**: 183 false positives (0.58% of true negatives)

## 3. Context Sensitivity

**Accuracy:** 80.0% (8/10 tests passed)

The model was tested on:
- Reclaimed slurs in positive contexts
- Educational/meta-discussion of harmful language
- AAVE and modern slang
- Passionate but non-toxic criticism

### Failed Tests:

- Positive usage: Expected NOT_TOXIC, got TOXIC
- Slang (positive): Expected NOT_TOXIC, got TOXIC

## 4. Recommendations

✅ **Low identity bias** (<5% false positive rate)
- Current performance is acceptable
- Continue monitoring in production

⚡ **Moderate context sensitivity** (70-90% accuracy)
- Some contextual understanding, but room for improvement

