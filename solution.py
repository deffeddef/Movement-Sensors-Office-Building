"""
The purpose of this program is to predict the activation of movement sensors based on past data. The input is a time
stamp containing year/month/day hour/minute/second with the target being a 1 or 0, which indicates an activation of the
sensor device at said timestamp.

I approach this task not as a single classification problem, but rather as 7 separate ones as there 7 devices. As just
mentioned, I will model this as a binary classification rather than a classic time series, using the scikit learn
implementation of a gradient boosting decision tree algorithm, as it is inherently non-parametric compared to linear
models. Surely, other algorithms might potentially yield pleasant results, too, but gradient boosting trees have
established themselves industry wide as providing good results out of the box.

Step by step, I will do the following:
"""

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, make_scorer
import category_encoders as ce
from skopt import BayesSearchCV
import sys
import itertools
import warnings
pd.options.mode.chained_assignment = None

print('Prediction in Progress...Please Wait!')

"""Write a function to deconstruct the timestamp elements in to separate columns. I have chosen to only include day 
of the month, weekday name and hour, since the prediction will take place on full hours only and thus minutes and 
seconds will always have the value 0. If we had a year round dataset, we could also include the month to account for
seasonality, but here this is not the case as we only have data for 2 months."""
def time_decon(df, col):
    df['day'] = df[col].dt.day
    df['day_name'] = df[col].dt.day_name()
    # df['month'] = df[col].dt.month
    df['hour'] = df[col].dt.hour
    return df


def predict_future_activation(current_time, previous_readings):
    """Create Instances of the tools we are going to use: Undersampling to counteract class imbalance, Binary Encoding
    for the weekday name categorical feature, Gradient Boosting Classifier, log loss as the loss function for K-Fold
    Cross Validation and Bayesian Hyperparameter Optimization"""
    us = RandomUnderSampler(sampling_strategy=1, random_state=42)
    be = ce.BinaryEncoder()

    gbc = GradientBoostingClassifier()
    # Parameter Space for Gradient Boosting Algorithm
    params_gb = {
        "n_estimators": [10, 20, 50, 100, 500],
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.],
        "max_features": list(np.arange(0.05, 1.01, 0.05)),
        "min_samples_split": list(range(2, 21)),
        "min_samples_leaf": list(range(1, 21))
    }

    lls = make_scorer(log_loss)

    bcv = BayesSearchCV(gbc,
                        params_gb,
                        n_iter=50,
                        random_state=42,
                        verbose=0,
                        n_jobs=-1,
                        n_points=50,
                        scoring=lls)


    """Setting up the data. Creating a dataframe containing all time slots to perform prediction on"""
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')
    dateframe = pd.DataFrame(next_24_hours, columns=['time'])
    dateframe = time_decon(dateframe, 'time')

    """Ingesting the CSV, converting the timestamp to a datetime object and sorting the data by time"""
    df = pd.read_csv(previous_readings)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['time'], inplace=True, ascending=True)

    """Creating the dataframe in which the 168 predictions will be saved into at the end"""
    device_names = sorted(df['device'].unique())
    xproduct = list(itertools.product(next_24_hours, device_names))
    preds = pd.DataFrame(xproduct, columns=['time', 'device'])
    preds.set_index('time', inplace=True)
    preds.sort_values(by=['device', 'time'], inplace=True)

    "We then create a column for each device in the df and fill it with a blank"
    pred_list = []
    for device in sorted(df['device'].unique()):
        df[device] = ''

        """Then we iterate through every row and check whether the value in the column 'device' (containing all device 
        names) is equal to the column names we just created. If so, we insert a value 1 at the corresponding new column,
        otherwise a 0. That way, for each device in the df, we now have the explicit information whether a device has 
        been activated or not during a time indicated in the df's timestamps. For example on 2016-07-01 04:23:32 the 
        column 'device_3' will be a 0, on 2016-07-01 08:07:57 the column 'device_3' will be 1. This provides us with a 
        non-sparse data set for each device on which we can train the algorithm"""
        for row in df['device'].iteritems():
            if row[1] == device:
                df[device][row[0]] = int(1)
            else:
                df[device][row[0]] = int(0)
            df = df[['time', 'device', device]]

        """We use the timestamp deconstructor on our timestamp, so it can be used for training"""
        df = time_decon(df, 'time')

        """Next is fitting the binary encoder on the categorical weekday features. This will convert them into Binary
        strings (i.e. '0010') across multiple columns, with each column holding one integer of the binary string. 
        Despite fitting on the entire data, data leakage is not an issue as this a fully unsupervised method. Never 
        before see categories are just assigned a separate generic binary string. I prefer this method one hot encoding
        as it usually needs fewer columns to represent the data(i.e. lower dimensionality and less noise), while 
        yielding at least similar if not better predictive power in most cases."""
        be.fit(df['day_name'], df[device])
        dnt = be.transform(df['day_name'])
        df = pd.concat([df, dnt], axis=1).drop(['day_name'], axis=1)

        y = df[device].astype('int')
        X = df.drop(['time', device, 'device'], axis=1)

        """Since we are dealing with highly imbalanced data, which can throw off our prediction significantly, we 
        decide to under-sample the 'non-activations' or 0 values, while not touching the informative 'activations' or 
        1 values, so that for each device we get a perfectly balanced dataframe. In my opinion, it is not problematic to
        remove some of these 0 values, because they have been somewhat randomly and heuristically generated and are not
        'original data' that we are tampering with. With random under-sampling they become a perfect counterweight to 
        our activations and should help to improve the predictive power of the model. Often such steps in data 
        preparation help improve a model's performance more than the choice of algorithm and the extensive tuning of its
        parameters"""
        X, y = us.fit_sample(X, y)


        """We then perform K-Fold cross validation in conjuction with Bayesian hyperparameter tuning in order to avoid 
        overfitting and increase predictive power by finding suitable parameters.
        
        Instead of using accuracy as the loss function for the optimization algorithm, log loss has been chosen, as it 
        is more nuanced than just accuracy by not just checking whether an instance has been classified 
        correctly/incorrectly, but also taking into account the underlying probabilities of a classification and 
        'punishing' classifiers, that assign a high probability to an incorrect classification and vice versa reward 
        classifiers that assign a high probability to a correct classification. This should in theory also improve 
        overall accuracy.

        NOTE: In this particular instance, I have not performed a train/test split for two reasons:
        a) over-fitting has already been addressed by performing k-fold cross validation in the previous step.
        b) in my opinion, the train/test split is more part of the model evaluation and validation process than the 
        actual process of model deployment, in which, again in my opinion, it is better to train on all the available 
        data, so that the predictive power in production is optimized."""
        with warnings.catch_warnings():
            #The Bayes Optimizer Library throws some pretty lenghty FutureWarnings during fitting. We will ignore them.
            warnings.filterwarnings("ignore")
            bcv.fit(X, y)

        """Before we can pass the unseen data to the trained algotrithm, we must also encode its weekday names into
        binary"""
        dntt = be.transform(dateframe['day_name'])
        dntt = pd.concat([dateframe, dntt], axis=1).drop(['day_name', 'time'], axis=1)

        """For each of the 7 devices we make a prediction on 24 input timestamps, which yields us 168(=24*7) predictions
        in total. The 24 predictions per device are then appended to a list outside the for loop, so that in the end, it
        will contain all 168 predictions"""
        pred = bcv.predict(dntt)
        pred_list.append(pred)

        print("Model Training and Prediction for", device, 'Successful!')

    """Next, we unpack the lists within the list, giving us a single big list and append it as a new column to our
    previously created prediction dataframe. Mind you, that in Line 85 we have sorted this dataframe by device first and
    then by time so that when we now append the predictions, they are perfectly aligned. Only after this appendage, we 
    sort the data again by time first and by device second."""
    pred_list = list(itertools.chain(*pred_list))
    preds['activation_predicted'] = pred_list
    predcount = preds['activation_predicted'].value_counts()
    print('For', len(preds), 'Time Slots', predcount[1], 'Activations have been Predicted')
    preds.sort_values(by=['time', 'device'], inplace=True)

    return preds



if __name__ == '__main__':

    current_time, in_file, out_file = sys.argv[1:]

    previous_readings = pd.read_csv(in_file)
    result = predict_future_activation(current_time, in_file)
    result.to_csv(out_file)
    print ('Prediction Successful!...Please Check CSV File for Results')

"""Question
Are there any points you could think of that could help improve your result (e.g. what if you had more data)?

Of course, more data is helpful most of the time (; But as I have mentioned already, I am quite sure that the activation
of the sensor devices is subject to a certain seasonality and that it is not just dependent on the hours and weekdays, 
but also whether it is summer or winter for example. Thus having year round activation data and not just for July/August
would have been very helpful to capture the effect of seasonality as with the current data set, you basically have to 
predict winter behavior with summer data and that might not yield the best results."""



