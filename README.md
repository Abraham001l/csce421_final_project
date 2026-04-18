Determining the goal for the model:

For a classification model there are three primary metrics which give insight into the performance of the model. Those being recall, precision, and F1-score which is the combination of the prior two metrics. To determine what scores our model should have in order to be considered successful we have to first consider all four cases:

- True positive: successful model execution

- False positive: leads to time waste by decoders (cost = time)

- True negative: successful model execution

- False negative: incredibly disastrous, loss of revenue (cost = money)

Under these circumstances we can identify that the success of our model depends on its ability to avoid False positives and negatives. 

We first look at the more concerning case of false negatives, this can be analyzed in the recall score of the model. More specifically we can make the claim that we will at most be willing to sacrifice 5% of our revenue, which then maps to a 95% recall score. 

Now we look at the false positive case, this can be analyzed in the precision score of the model. To do this we can make the claim that we are at most willing to waste 15% of the time of the decoders, under this claim we can determine that the model would need a precision score of at least 85%.

These numbers at the moment make heavy assumptions. The 95% recall score assumes that the cost of all procedures (ICD codes) is the exact same, which in reality is not true. The 85% precision score assumes that all procedures (ICD codes) cost the exact same time, which is also not true. Our team's reasoning behind these assumptions is to simply develop reasonable numbers for our models to aim for. Once a final model is made, more exact cost analysis can be performed. For the money cost the MEPS (Medical Expenditure Panel Survey) dataset can be used to map ICD codes to approximate monetary cost allowing us to quantify exactly what percentage of revenue our model’s classification’s cost. As for the time cost, that would require a more involved approach to understand how much time it takes to analyze certain procedures (ICD codes).
