![Cover image](/images/cover_image.png)

# Fantasize

## Predicting fantasy novel popularity

### Problem
Most publishers select only a very few books to publish each year. One receives 10,000 manuscripts and publishes only 10 books a year. Therefore, as a publisher, you would want to select manuscripts that will be popular in order to maximize sales. The Fantasize web app helps publishers select the right manuscripts to publish. The app layout is as follows.

![App layout](/images/app_layout.png)

### Method
The Fantasize web app predicts whether a fantasy novel will be popular or not using a model built upon both metadata and synopsis data from 84,000 Goodreads fantasy novels. The app uses natural language processing techniques (BERT) to analyze synopsis, and uses a boosted tree model to predict popularity. The model structure is as shown in the picture below. 

![Model structure](/images/model_structure.png)

### Workflow
* Built models on 84,000 Goodreads fantasy novels.
* Cleaned dataset and conducted exploratory data analysis.
* Explored different ways to vectorize synopsis data, including bag-of-words and TF-IDF models.
* Constructed popularity classifiers using both the state-of-the-art NLP model, BERT, and boosted tree models.
* Boosted popularity classification accuracy from 58% to 67%.
* The web app is built using Streamlit and hosted on AWS.
* Given more time, various future directions below can be pursued to further increase model performance. 

![Future directions](/images/future_directions.png)

### References
This web app is built upon the following data and code sources.
* [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
* [Chris McCormick NLP](https://www.chrismccormick.ai/)
* [How to solve 90% of NLP problems: a step-by-step guide](https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e#:~:text=Remove%20all%20irrelevant%20characters%20such,%2C%20and%20%E2%80%9CHELLO%E2%80%9D%20the%20same)
