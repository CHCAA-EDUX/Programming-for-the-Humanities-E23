# Binary Email Classifier

This project contains a Python script that classifies emails as either spam or not spam using a Naive Bayes classifier.

author: 
group members: x, y, z

## Project Structure

```bash
├── dat                                                                                                  │
│   └── emails.csv                                                                                       │
├── figs                                                                                                 │
│   └── cm.png                                                                                           │
├── README.md                                                                                            │
├── requirements.txt                                                                                     │
└── src                                                                                                  │
    └── my_binary_classifier.py 
```


## Requirements

This project requires the following Python packages:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

You can use the BinaryEmailClassifier class in the my_binary_classifier.py script to classify emails. Here's an example:

For Python 3:


```python
from my_binary_classifier import BinaryEmailClassifier

bec = BinaryEmailClassifier()
bec.load_data('dat/emails.csv')
bec.preprocess_data()
bec.train()
bec.evaluate()
```

For CMD:

```bash
python my_binary_classifier.py
```