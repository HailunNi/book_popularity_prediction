# Fantansize

## Predicting fantasy novel popularity

### Problem
Most publishers select only a very few books to publish each year. One receives 10,000 manuscripts and publishes only 10 books a year. Therefore, as a publisher, you would want to select manuscripts that will be popular in order to maximize sales. 

### Method
The Fantasize web app predicts whether a fantasy novel will be popular or not using a model built upon both metadata and synopsis data from 84,000 Goodreads fantasy novels. The app uses natural language processing techniques (BERT) to analyze synopsis, and uses a boosted tree model to predict popularity. 

### Workflow
* Built models on 84,000 Goodreads fantasy novels.<br>
* Cleaned dataset and conducted exploratory data analysis.<br>
* Explored different ways to vectorize synopsis data, including bag-of-words and TF-IDF models.<br>
* Constructed popularity classifiers using both the state-of-the-art NLP model, BERT, and boosted tree models.<br>
* Boosted popularity classification accuracy from 58% to 67%. <br>

### References
This web app is built upon the following data and code sources.
* <a href="https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home">UCSD Book Graph</a> <br>
* <a href="https://www.chrismccormick.ai/">Chris McCormick NLP</a> <br>
* <a href="https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e#:~:text=Remove%20all%20irrelevant%20characters%20such,%2C%20and%20%E2%80%9CHELLO%E2%80%9D%20the%20same">How to solve 90% of NLP problems: a step-by-step guide</a> <br>
