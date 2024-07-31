# F1 Race Position Prediction

## Introduction
This project aims to develop a machine learning model capable of predicting the position order of drivers in the next Formula 1 race. By leveraging historical race data and various features such as driver performance, team performance, the model will provide insights and predictions to help understand and anticipate the outcomes of upcoming races.
<br>
This project was realized as a way to get a hold of the vast world of Data science and Machine learning, with the constant help of Juan Kurucz telling me what could be a good approach or what technology to add next, it really helped me get a valuable insight into industry best practices.

## Technologies
- Optuna
- Sklearn
- Pandas
- MLFlow
- Autogluon or FastAi (WIP)
- Gradio (WIP)

## How to run
Use poetry to manage the dependencies, and then run the "predictioner.py" file.
It will run a prediction of the Spa race, and print the results.

## What I learned

### Data Cleaning and Preparation
- **Importance of Data Quality**: Ensured the dataset was clean and free from inconsistencies, highlighting the importance of high-quality data. E.g., there were races from the past with more than 20 drivers, which ended in the model trying to predict in the range from 1 to n (being n>20) rather than 1 to 20.
- **Feature Engineering**: Created meaningful features from raw data, understanding the critical role feature engineering plays in model performance, as I could see how the model improved with more recent and meaningful data.

### Experiment Tracking with MLFlow
- **Experiment Management**: Used MLFlow to track experiments, making it easier to compare different model runs and their respective parameters.
- **Reproducibility**: Learned to maintain reproducibility of experiments by recording model configurations.

### Hyperparameter Optimization with Optuna
- **Optimization Techniques**: Leveraged Optuna for hyperparameter tuning, gaining insights into the benefits of automated optimization techniques.
- **Efficiency**: Noted significant improvements in model performance by efficiently exploring the hyperparameter space.

### Model Training and Evaluation
- **Model Comparison**: Trained and evaluated multiple models, including Gradient Boosting Classifier, Random Forest Classifier, and Multinomial Naive Bayes.
- **Performance Metrics**: Assessed models using appropriate metrics, learning to select the best model based on evaluation results rather than assumptions.

### Practical Insights
- **Iterative Process**: Recognized the iterative nature of machine learning projects, where continuous improvements and refinements lead to better outcomes.

## Conclusion
This project is nowhere near ending, and this was a common feeling all along the way. However, I'm not saying this in a bad way. As I advanced, my horizon broadened while also the scope of the project, wanting each iteration to improve what was already done, understanding something new that could be useful, it was a really exciting thing to do. Even where it seemed really distant the next thing, mostly as I lacked knowledge of the technologies, it was still an attractive challenge to take on.

## References
[Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) <br>
I took plenty of references of EDA and data manipulation from [this kaggle post](https://www.kaggle.com/code/yanrogerweng/formula-1-race-prediction#Check-dataframe)

