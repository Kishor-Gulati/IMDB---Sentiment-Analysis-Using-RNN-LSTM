# IMDB---Sentiment-Analysis-Using-RNN-LSTM

## Introduction
This project aims to perform sentiment analysis on the IMDB movie review dataset using Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) cells. Sentiment analysis involves classifying text data into positive or negative sentiment categories based on the content of the text.

## Dataset
The dataset used for this project is the IMDB movie review dataset, which contains movie reviews labeled as positive or negative sentiment. Each review is preprocessed and tokenized before being fed into the model for training and evaluation.

## Model Architecture
The model architecture consists of an embedding layer, which converts words into dense vectors, followed by one or more layers of RNNs with LSTM cells. The LSTM layers help capture sequential information from the text data, allowing the model to learn long-range dependencies.

## Training
The model is trained using the Adam optimizer with a binary cross-entropy loss function. During training, the model's performance is evaluated on a separate validation dataset to monitor for overfitting. Early stopping is implemented to prevent overfitting and improve generalization.

## Evaluation
The trained model is evaluated on a separate test dataset to assess its performance in classifying sentiment in unseen movie reviews. Performance metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate the model's effectiveness.

## Performance
# Model Performance
The trained model achieved an accuracy of 87.02% on the test dataset, effectively classifying sentiment in movie reviews as positive or negative.

## Classification Report
```
Classification report
              precision    recall  f1-score   support

           0       0.88      0.86      0.87      4993
           1       0.86      0.88      0.87      5007

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000

-----------------------------------------------------
Accuracy of the model:  0.8702
```

## Future Scope

1. **Parameter Tuning**: Fine-tune hyperparameters such as learning rate, batch size, and optimizer settings to optimize model performance.

2. **Stack Two or More LSTM Layers**: Increase model complexity by stacking multiple LSTM layers to improve representation learning and capture more intricate patterns in the text data.

3. **Add Dropout**: Implement dropout regularization to prevent overfitting and improve model generalization by randomly dropping neurons during training.

4. **Change Learning Rate and Epochs**: Explore different learning rates and epochs to find the optimal training schedule for the model.

5. **Deployment**: Deploy the trained model into production environments for real-world applications.

### Using Simple LSTM

```
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True), # Word2Vec
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1) # tf.keras.layers.Dense(3, activation="softmax")
])
```
### Stack Two or More LSTM Layers

```
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True), # Word2Vec
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1) # tf.keras.layers.Dense(3, activation="softmax")
])
```
