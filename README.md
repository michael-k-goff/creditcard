# Credit Card Fraud Detection

Credit card fraud is a major and growing problem, with [direct losses](https://chargebacks911.com/credit-card-fraud-statistics/) exceeding $28 billion worldwide in 2020 and $11 billion in the United States. World losses have more than tripled from 2011 to 2021, and more growth is expected in the next few years. There is a critical need for financial institutes to provide robust anti-fraud protection, for which fraud detection is a vital tool.

The goal of this project is to build a classifier to detect credit card fraud. This project is based on a dataset of 213605 credit card transactions, of which 213236 are labeled as valid and 369 are labeled as fraudulent. Each transaction is identified by 28 features, the nature of which cannot be determined, as well as a timestamp and a transaction amount.

After data preprocessing, several classification algorithms were attempted. Of these, the random forest showed the most promising results, and it is the only result attempted here that I would consider for further development if I was a cybersecurity manager at a bank.

## Business considerations

For use in a production environment, [data drift](https://www.sciencedirect.com/science/article/pii/S1319157822004062) would be a critical issue, as fraudsters would likely quickly adapt to any fixed fraud detection tool, and thus the kind of static dataset that was used for this project would be insufficient.

An important business consideration is relative weighting of type I and type II errors. Clearly a false negative (type II error, or classifying a fraudulent transaction as safe) is more harmful than a false positive (type I error, or classifying a safe transaction as fraudulent), and the magnitude of this difference should be accounted in scoring.

## Tools used

The following algorithms were used in the course of this project.

- Random forest classifier
- Support vector classifier
- Calibrated classifier
- Lazy predictor
- Data normalization
- Outlier detection
- Oversampling via SMOTE and undersampling
- F-score
- Confusion matrix