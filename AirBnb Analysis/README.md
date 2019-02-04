Airbnb Rental Data from Seattle 

Business Understanding

Airbnb is a technology company that has disputed the  hotel industry by providing a platform for people to rent out their homes.

Data Understanding

As a result, of proving a popular platform the company has access to a wealth of information about how people travel.  
Some descriptive analysis was completed to better undersand the data, and to propose questions that if answered could be used to improve the business.   

Prepare Data

Much of the data preparation included refromating the data, changing strings to ints/floats, and creating basic graphics to compare the data. One of Airbnb's primary challenges
is to ensure that their are enough reliable hosts offering rental services.  They have classified their host as either Regular host or Superhost, based on the reviews that they receive
this helps to ensure that people seeking to rent places receive an optimal experience.  However, it wasn't clear if there was any incentive for host to strive for a SuperHost designation.   

The questions that are explored are:
1. Is it worthwhile to become a SuperHost?
2. What characterizes a Superhost  
3. Are there things that one can do in order to become a Superhost?


Data Modeling

A Logistic regression model was implemented to conduct feature analysis which revealed they types of feature that could be used to predict if a host was a SuperHost.  
Airbnb explains that they use reviews to identify Superhost, because it is understand that the quality and number of customer reviews would affect a host's status that data was not included in the prediction model.
However, analysis of the customer review data did show clear differnces amoung the two classes of host. 


Evaluate the Results

Results suggests that SuperUser are able to charge more an incur lower expenses for comprably sized rentals when compared to Non-superusers resulting in higher profits.

read more here: https://medium.com/@dkylemiller/the-number-one-thing-that-you-should-know-if-you-want-to-make-more-money-on-airbnb-6605070a4dca

Libraries Used:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Motivation for Project - To better understand characteristics associated with Superuser on Airbnb

Dataset found here on Kaggle: https://www.kaggle.com/airbnb/seattle

