# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.

## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

In this section, provide your interpretation of the Logistic Regression model's performance on the imbalanced dataset. Consider:

- Which metric performed best and why?
- Which metric performed worst and why?
- How much did the class imbalance affect the results?
- What does the confusion matrix tell you about the model's predictions?

The logistic regression model trained on the imbalanced dataset showed strong accuracy (91.68%), but this was misleading due to the underlying class imbalance. While accuracy appeared high, recall was particularly low (30.07%), meaning the model failed to identify a large number of true positive cases. Precision also suffered, and the F1 score reflected this imbalance with only moderate performance. The confusion matrix revealed that the model predicted the majority of the negative class well but missed many actual positive cases, indicating that the model was biased toward the majority class. This showcases the limitations of relying on accuracy alone in imbalanced classification problems.

## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

In this section, compare the performance of the Random Forest and XGBoost models:

- Which model performed better according to AUC score?
- Why might one model outperform the other on this dataset?
- How did the addition of time-series features (rolling mean and standard deviation) affect model performance?

Introducing time-series features and using more powerful models significantly improved performance. Both Random Forest and XGBoost achieved high AUC scores, with XGBoost outperforming Random Forest (0.9953 vs. 0.9735). This performance gain can be attributed to XGBoost's ability to capture complex relationships through gradient boosting. The addition of rolling mean and standard deviation features for heart rate helped both models learn temporal trends, leading to better model discrimination.

## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

In this section, analyze the improvements gained by addressing class imbalance:

- Which metrics showed the most significant improvement?
- Which metrics showed the least improvement?
- Why might some metrics improve more than others?
- What does this tell you about the importance of addressing class imbalance?

After applying SMOTE to balance the training dataset, the logistic regression model’s recall jumped from 30.07% to 86.01%, a dramatic improvement in its ability to identify positive cases. The F1 score also improved, reflecting a better balance between precision and recall. Accuracy remained strong (85.68%), and although AUC only increased slightly, the overall quality of predictions improved, particularly for the minority class. 

## Overall Conclusions

Summarize your key findings from all three parts of the assignment:

- What were the most important factors affecting model performance?
- Which techniques provided the most significant improvements?
- What would you recommend for future modeling of this dataset?

Across all three parts of the assignment, the most impactful factors influencing model performance were class imbalance and the use of engineered features. The logistic regression model performed poorly on the imbalanced dataset but improved significantly after applying SMOTE. Tree-based models, especially XGBoost, showed good results even before balancing due to their inherent robustness and the use of time-series features. In future modeling efforts with similar datasets, it is important to use class-balancing techniques like SMOTE and to engineer features that capture domain-specific patterns. Additionally, ensemble models like XGBoost should be considered for their ability to capture complex interactions in the data.