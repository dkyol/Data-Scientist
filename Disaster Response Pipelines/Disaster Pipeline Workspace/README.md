# Disaster Response Pipeline Project
This project was designed to assist first responders prioritize their response to messages.

File: process_data.py
This file loads two sources of data in CSV format
Merges the sources and pre-processes the data into a SQL table that is loaded into a database (DisasterResponse.db)

File: Train_Classifier.py 
This file loads the SQL table from the database 
Separated the contents of the table into independent and dependent variables
The messages, the independent variables, are tokenized. Punctuation is removed, upper case lettering are removed, common stop words defined by the nltk library are removed and the words are lemmatized.
The TfidfVectorizer process the tokens returning essentially a weighted array of term frequency that can be fed into a classifier
A pipeline is used here to prevent the risk of data leakage, and combination of Multioutput classifiers and Random Forest Classifier are included in the pipeline. The Random Forest Classifier was chosen because it is one of the classifiers that accepts multiclass outputs
GridSearch is used to optimize the parameters of the above models, and the ideal model is saved to a pickle file
Running the run.py app with the following prompts launches the dashboard
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
File: run.py 

Add additional visualizations using the SQL table that was loaded 
python run.py
env|grep WORK
[displays the space domain & space ID]
https://SPACEID-3001.SPACEDOMAIN


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
