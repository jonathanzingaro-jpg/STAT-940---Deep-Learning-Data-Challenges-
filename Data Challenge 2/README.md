# Automated Essay Scoring with an Ensemble of Fine-Tuned Transformers

Fine-tunes four pretrained language models from Hugging Face (**BERT, DistilBERT, ALBERT and GPT-2**) to predict the score of student essays, then averages their predictions. Built for Data Challenge 2 of **STAT 940 – Deep Learning** at the University of Waterloo ([Kaggle competition](https://www.kaggle.com/competitions/uwstat940f23dc2)).

## Results

**Kaggle Test Score: 0.335 RMSE**

Validation RMSE on a 10% hold-out (300 essays, lower is better):

| Model | Validation RMSE |
|---|---|
| Baseline (predict the mean) | 0.654 |
| BERT | 0.356 |
| DistilBERT | 0.360 |
| ALBERT | **0.335** |
| GPT-2 | 0.369 |
| **Ensemble (Average)** | **0.337** |

## Approach

- **Data:** 3,000 scored essays for training (split 90/10 into training and validation) and 500 test essays. Scores have a mean of 3.49 and a standard deviation of 0.66.
- **Models:** `bert-base-uncased`, `distilbert-base-uncased`, `albert-base-v2` and `gpt2`. Each gets a single-output regression head and is fine-tuned end to end on essays of up to 512 tokens.
- **Training:** MSE loss on standardized scores, AdamW with a linear warm-up/decay schedule, gradient clipping and bfloat16 mixed precision. Each model trains for 4 epochs on an NVIDIA RTX 5090, and the epoch with the lowest validation RMSE is kept.
- **Ensemble:** the four models' predictions are averaged.

## Author

Giovanni (Jonathan) Zingaro · [GitHub](https://github.com/jonathanzingaro-jpg) · [LinkedIn](https://www.linkedin.com/in/jonathanzingaro/)
