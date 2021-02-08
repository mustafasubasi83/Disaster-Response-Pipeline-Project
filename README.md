# Disaster-Response-Pipeline-Project


This project implements a classifier model to categorize messages sent by people during natural disasters. After classification, the messages can be directed to the appropriate disaster relief agency. The training data provided by Figure Eight was mined using ETL and natural language processing pipelines.


### Project description

The Project is divided into the following Sections:

#### 1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure.

A Python script, `process_data.py`, contains the data cleaning pipeline that:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in an SQLite database

#### 2. Machine Learning Pipeline to train a model which is able to classify text messages in 36 categories

A Python script, `train_classifier.py`, that holds the code of the machine learning pipeline that:

- Loads data from the SQLite database saved before
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### 3. Web Application using Flask to show model results and predictions in real time.

Holds the code of the Web Interface from which the end-user will enter the message's text to get the predictions. This page will guide the emergency worker providing predictions about the type of emergency at hand.

#### Files description

The list of the files used in this project are:

```sh
- app
|  - template
|  |- master.html   # main page of web app
|  |- go.html       # classification result page of web app
|- run.py           # Flask file that runs the app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv    # data to process
|- process_data.py          # process data and saves it into a database
|- DisasterResponse.db      # database to save clean data to

- models
|- train_classifier.py  # machine learning modeling and prediction
|- DisasterResponseModel.pkl     # saved model


- README.md
```

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that loads the source CSV files, cleans them and stores the cleaned resulting Pandas Dataframe in a database. Run the following under the `data` directory.

        ```
        python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
        ```

    - To run ML pipeline that trains classifier and saves the model in a pickle file run the following from the `models` directory:

        ```
        python train_classifier.py ../data/DisasterResponse.db classifier.pkl
        ```

2. Run the following command in the `app` directory to run the web application in order to do predictions:

    ```
    python run.py
    ```

3. Then go to http://0.0.0.0:3001/ to launch the web interface of the application

4. In the box on the top enter the message to classify and click on the "Classify Message" button.


### Web App Screen Shots:

- The main page shows some graphs about training dataset, provided by Figure Eight


![Main Page](/WebApp_ScreenShots/Disaster Response Project _ Web App Main Page.png)


- This is an example of a message we can type to test the performance of the model. After clicking Classify Message, we can see the categories which the message belongs to highlighted in green

![alt Disaster-Response-Pipeline-Project]](WebApp_ScreenShots/Disaster Response Project _ Analyzing message data)


<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model

