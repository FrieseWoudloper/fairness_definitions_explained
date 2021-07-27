# logistic_regression.py
# Required Python Packages

import pandas as pd
import numpy as np
# import pdb
# import plotly.plotly as py
# import plotly.graph_objs as go
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict # Werkt alleen in python 2.7 
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import copy
# import seaborn as sns
from random import randint
from sklearn.feature_selection import RFE

# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)


# Files
DATA_SET_PATH = "../inputs/german.csv"


def dataset_headers(dataset):
    """
    To get the dataset header names
    :param dataset: loaded dataset into pandas DataFrame
    :return: list of header names
    """
    return list(dataset.columns.values)



def unique_observations(dataset, header, method=1):
    """
    To get unique observations in the loaded pandas DataFrame column
    :param dataset:
    :param header:
    :param method: Method to perform the unique (default method=1 for pandas and method=0 for numpy )
    :return:
    """
    try:
        if method == 0:
            # With Numpy
            observations = np.unique(dataset[[header]])
        elif method == 1:
            # With Pandas
            observations = pd.unique(dataset[header].values.ravel())
        else:
            observations = None
            print "Wrong method type, Use 1 for pandas and 0 for numpy"
    except Exception as e:
        observations = None
        print "Error: {error_msg} /n Please check the inputs once..!".format(error_msg=e.message)
    return observations



def feature_target_frequency_relation(dataset, f_t_headers):

    """
    To get the frequency relation between targets and the unique feature observations
    :param dataset:
    :param f_t_headers: feature and target header
    :return: feature unique observations dictionary of frequency count dictionary
    """

    # print f_t_headers

    feature_unique_observations = unique_observations(dataset, f_t_headers[0])
    unique_targets = unique_observations(dataset, f_t_headers[1])
    # print feature_unique_observations, unique_targets

    frequencies = {}
    for feature in feature_unique_observations:
        frequencies[feature] = {unique_targets[0]: len(
            dataset[(dataset[f_t_headers[0]] == feature) & (dataset[f_t_headers[1]] == unique_targets[0])]),
            unique_targets[1]: len(
                dataset[(dataset[f_t_headers[0]] == feature) & (dataset[f_t_headers[1]] == unique_targets[1])])}
    return frequencies



def feature_target_histogram(feature_target_frequencies, feature_header):
    """

    :param feature_target_frequencies:
    :param feature_header:
    :return:
    """
    keys = feature_target_frequencies.keys()
    y0 = [feature_target_frequencies[key][0] for key in keys]
    y1 = [feature_target_frequencies[key][1] for key in keys]

    trace1 = go.Bar(
        x=keys,
        y=y0,
        name='Clinton'
    )
    trace2 = go.Bar(
        x=keys,
        y=y1,
        name='Dole'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group',
        title='Feature :: ' + feature_header + ' Clinton Vs Dole votes Frequency',
        xaxis=dict(title="Feature :: " + feature_header + " classes"),
        yaxis=dict(title="Votes Frequency")
    )
    fig = go.Figure(data=data, layout=layout)
    # plot_url = py.plot(fig, filename=feature_header + ' - Target - Histogram')
    # py.image.save_as(fig, filename=feature_header + '_Target_Histogram.png')



def train_logistic_regression(train_x, train_y):
    """
    Training logistic regression model with train dataset features(train_x) and target(train_y)
    :param train_x:
    :param train_y:
    :return:
    """

    logistic_regression_model = LogisticRegression(penalty='l2', C=1.0)
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model



def model_accuracy(trained_model, features, targets):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:
    """
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score


# I need a separate test generation module for causal definitions and verify the model

Calibration = {0:{}, 1:{}}
BNC = [[], []]
BPC = [[], []]

NPV = [[], []]
FOR = [[], []]
PPV = [[], []]
FDR = [[], []]
FPR = [[], []]
TNR = [[], []]
FNR = [[], []]
TPR = [[], []]
Statistical = [[], []]
conditioned_statistical = [[], []]
overall_accuracy = [[], []]
treatment_equality = [[], []]
weights = {}
ranks = {}

def MyFn(a):
    return abs(a[1])


def cal_avg(times, headers):
    global Calibration, NPV, FOR, BNC, BPC, PPV, FDR, FPR, TNR, FNR, TPR, Statistical, overall_accuracy, treatment_equality, weights
      
    # for dep in range(11):
    #     if len(Calibration[0][dep]) != 0:
    #         Calibration[0][dep] = sum(Calibration[0][dep])*1.0/len(Calibration[0][dep])
    #     else:
    #         Calibration[0][dep] = "null"
    #     if len(Calibration[1][dep]) != 0:
    #         Calibration[1][dep] = sum(Calibration[1][dep])*1.0/len(Calibration[1][dep])
    #     else:
    #         Calibration[1][dep] = "null"

    # BPC = [sum(BPC[0])*1.0/len(BPC[0]), sum(BPC[1])*1.0/len(BPC[1])]
    # BNC = [sum(BNC[0])*1.0/len(BNC[0]), sum(BNC[1])*1.0/len(BNC[1])]

    # NPV = [sum(NPV[0])*1.0/len(NPV[0]), sum(NPV[1])*1.0/len(NPV[1])]
    # FOR = [sum(FOR[0])*1.0/len(FOR[0]), sum(FOR[1])*1.0/len(FOR[1])]
    # PPV = [sum(PPV[0])*1.0/len(PPV[0]), sum(PPV[1])*1.0/len(PPV[1])]
    # FDR = [sum(FDR[0])*1.0/len(FDR[0]), sum(FDR[1])*1.0/len(FDR[1])]
    # FPR = [sum(FPR[0])*1.0/len(FPR[0]), sum(FPR[1])*1.0/len(FPR[1])]
    # TNR = [sum(TNR[0])*1.0/len(TNR[0]), sum(TNR[1])*1.0/len(TNR[1])]
    # TPR = [sum(TPR[0])*1.0/len(TPR[0]), sum(TPR[1])*1.0/len(TPR[1])]
    # FNR = [sum(FNR[0])*1.0/len(FNR[0]), sum(FNR[1])*1.0/len(FNR[1])]
    # Statistical = [sum(Statistical[0])*1.0/len(Statistical[0]), sum(Statistical[1])*1.0/len(Statistical[1])]
    # overall_accuracy = [sum(overall_accuracy[0])*1.0/len(overall_accuracy[0]), sum(overall_accuracy[1])*1.0/len(overall_accuracy[1])]
    # treatment_equality = [sum(treatment_equality[0])*1.0/len(treatment_equality[0]), sum(treatment_equality[1])*1.0/len(treatment_equality[1])]
    weight = []
    for i in weights:
        weights[i] = sum(weights[i])*1.0/len(weights[i])
        weight.append((i, weights[i]))
    weight = sorted(weight, key=MyFn, reverse=True)
        # 
        # sm += weights[i]
    # print weight, type(weight), raw_input()
    for i in weight:
        print i
    print '\n\n\n'
    # rank = []
    # for i in range(len(headers)):
    #     ranks[headers[i]] = sum(ranks[headers[i]])*1.0/len(ranks[headers[i]])
    #     rank.append((headers[i], ranks[headers[i]]))
    
    # rank = sorted(rank, key=lambda x: x[1])
    # for i in range(len(rank)):
    #     print i, rank[i]

    # raw_input()
    # print sm
    # for dep in range(11):
    #     print "Calibration for S = %d   "%(dep), Calibration[0][dep], Calibration[1][dep]

    # print "Balance for positive class for the two classes: ", BPC
    # print "Balance for negative class for the two classes: ", BNC

    # print "NPV's for the two classes: ", NPV
    # print "FOR's for the two classes: ", FOR
    # print "PPV's for the two classes: ", PPV
    # print "FDR's for the two classes: ", FDR
    # print "FPR's for the two classes: ", FPR
    # print "TNR's for the two classes: ", TNR
    # print "TPR's for the two classes: ", TPR
    # print "FNR's for the two classes: ", FNR
    # print "Statistical Parity : ", Statistical
    # print "Overall Accuracy equality : ", overall_accuracy
    # print "Treatment equality : ", treatment_equality


# to test fairness by awarness, we consider three ranges of difference in age, 5, 10, 15 and test very similar to function in test_unaware.
def test_aware(dataset):
    df1 = dataset.copy()         # this is deepcopy, so changing df will not affect dataset
    df2 = dataset.copy()        # I create two copies of dataset, df1 and df2, i will make changes in df2 so that df1 and df2 just differ in Personal-status-and-sex column
    df3 = dataset.copy()
    # make Personal-status-and-sex for dataset = 0, train and now test on df1 and df2
    df1.drop('target', axis=1, inplace=True)
    df2.drop('target', axis=1, inplace=True)
    df3.drop('target', axis=1, inplace=True)
    # print max(df1['Age']), min(df1['Age']), raw_input()
    # replaced status column by 0 for all instances, hence now shouldn't affect the model while training.
    # print dataset['Personal-status-and-sex'], raw_input()
    dataset = pd.get_dummies(dataset, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"]) 
    
    # now take avg of output of df2 and df3 and substract from output of df1, this will give you the answer for discrimination
    for i in range(len(dataset)):
        age = df1.loc[:,('Age')][i]
        df2.loc[:,('Age')][i] = age + 50          # set age + 5
        df3.loc[:,('Age')][i] = age - 50          # set age - 5
        print i

    df1 = pd.get_dummies(df1, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])
    df2 = pd.get_dummies(df2, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])
    df3 = pd.get_dummies(df3, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])

    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=None)
    df1 = df1.as_matrix()            # converts it to a numpy array which is easier to handle
    df2 = df2.as_matrix()            # converts it to a numpy array which is easier to handle
    df3 = df3.as_matrix()            # converts it to a numpy array which is easier to handle

    z = dataset.target.as_matrix()               # they are changed to numpy matrix in corresonding order, so no problem
    dataset.drop('target', axis=1, inplace=True)
    d1 = dataset.as_matrix()

    num_diffs = []
    for train_index, test_index in rkf.split(d1):       # train on dataset
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_x, test_x = d1[train_index], d1[test_index]       # use this for training, Personal-status-and-sex is all 0
        train_y, test_y = z[train_index], z[test_index]
        trained_logistic_regression_model = train_logistic_regression(train_x, train_y)     
        # now this model is trained excluding gender stuff
        a = 0
        for i in range(len(test_index)):        # this is 1000 instances, now compare with the predicted outcome for df and real outcome given by dataset
            # print trained_logistic_regression_model.predict_proba(df1[int(test_index[i])].reshape(1,-1))[0][1], trained_logistic_regression_model.predict_proba(df2[int(test_index[i])].reshape(1,-1)), trained_logistic_regression_model.predict_proba(df3[int(test_index[i])].reshape(1,-1))[0], raw_input()
            b = (trained_logistic_regression_model.predict_proba(df2[int(test_index[i])].reshape(1,-1))[0][1] + trained_logistic_regression_model.predict_proba(df3[int(test_index[i])].reshape(1,-1))[0][1])*1.0/2 - trained_logistic_regression_model.predict_proba(df1[int(test_index[i])].reshape(1,-1))[0][1]
            print b
            if b > 0.5/56:
                a += 1
        num_diffs.append(a)
        # print a
    print num_diffs
    print sum(num_diffs)*1.0/len(num_diffs), " now over fairness through unawareness ", len(num_diffs)


def create_tests_causal(dataset):
    df = dataset.copy()         # this is deepcopy, so changing df will not affect dataset
    df.drop('target', axis=1, inplace=True)
    # here you can mutate the df 'Personal-status-and-sex' attribute
    for i in range(len(df)):
        v = df.loc[:,('Personal-status-and-sex')][i]
        if v == 'A91' or v == 'A93' or v == 'A94':      # male
                df.loc[:,('Personal-status-and-sex')][i] = 'A92'   # change into female
        elif v == 'A92':        # female
            r = randint(1, 3)
            if r == 1:
                df.loc[:,('Personal-status-and-sex')][i] = 'A91'   # change into male, randomly into any of the 3 types
            elif r == 2:
                df.loc[:,('Personal-status-and-sex')][i] = 'A93' 
            elif r == 3:
                df.loc[:,('Personal-status-and-sex')][i] = 'A94'
            else:
                assert False
        else:
            assert False
        # print i, v, df.loc[:,('Personal-status-and-sex')][i]

    # so now df and dataset differ in gender column

    df = pd.get_dummies(df, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"]) 
    dataset = pd.get_dummies(dataset, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"]) 
    # print dataset.shape, raw_input()
    # print dataset_headers(df)
    # now for each instance in df, we can change its personal status and sex attribute, 
    # keep other attributes the same and this entire set of 1000 instances will be a used as test for the trained model
    # so this function just creates another dataframe with corresponding indexes, here training will not be same as 
    # other definitions, it will exclude the column of personal status and sex. I will use 10 fold cross validation
    # training has exluded the status column, now test on pairs of 100 test cases in each iteration, which only differ in status column, other columns are same
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=None)
    df = df.as_matrix()            # converts it to a numpy array which is easier to handle
    z = dataset.target.as_matrix()               # they are changed to numpy matrix in corresondong order, so no problem
    dataset.drop('target', axis=1, inplace=True)
    d1 = dataset.as_matrix()

    num_diffs = []
    for train_index, test_index in rkf.split(df):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_x, test_x = d1[train_index], d1[test_index]
        train_y, test_y = z[train_index], z[test_index]
        trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
        a = 0
        for i in range(len(test_index)):        # this is 1000 instances, now compare with the predicted outcome for df and real outcome given by dataset
            if trained_logistic_regression_model.predict(d1[int(test_index[i])].reshape(1,-1))[0] != trained_logistic_regression_model.predict(df[int(test_index[i])].reshape(1,-1))[0]:
                a += 1
        num_diffs.append(a)
        # print a
    print sum(num_diffs)*1.0/len(num_diffs), " now over causal discrimination ", len(num_diffs)



# to test for this definition of fairness, replace the Personal-status-and-sex by 0 for all instances
def test_unaware(dataset):
    df1 = dataset.copy()         # this is deepcopy, so changing df will not affect dataset
    df2 = dataset.copy()        # I create two copies of dataset, df1 and df2, i will make changes in df2 so that df1 and df2 just differ in Personal-status-and-sex column
    # make Personal-status-and-sex for dataset = 0, train and now test on df1 and df2
    df1.drop('target', axis=1, inplace=True)
    df2.drop('target', axis=1, inplace=True)
    # replaced status column by 0 for all instances, hence now shouldn't affect the model while training.
    # print dataset['Personal-status-and-sex'], raw_input()
    dataset = pd.get_dummies(dataset, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"]) 
    dataset['Personal-status-and-sex_A91'].replace(1, 0, inplace=True)
    dataset['Personal-status-and-sex_A92'].replace(1, 0, inplace=True)
    dataset['Personal-status-and-sex_A93'].replace(1, 0, inplace=True)
    dataset['Personal-status-and-sex_A94'].replace(1, 0, inplace=True)
    # print dataset['Personal-status-and-sex_A92'], dataset['Personal-status-and-sex_A91'], dataset['Personal-status-and-sex_A94'], raw_input()
    
    # here change Personal-status-and-sex column for df2

    for i in range(len(df2)):
        v = df2.loc[:,('Personal-status-and-sex')][i]
        if v == 'A91' or v == 'A93' or v == 'A94':      # male
                df2.loc[:,('Personal-status-and-sex')][i] = 'A92'   # change into female
        elif v == 'A92':        # female
            r = randint(1, 3)
            if r == 1:
                df2.loc[:,('Personal-status-and-sex')][i] = 'A91'   # change into male, randomly into any of the 3 types
            elif r == 2:
                df2.loc[:,('Personal-status-and-sex')][i] = 'A93' 
            elif r == 3:
                df2.loc[:,('Personal-status-and-sex')][i] = 'A94'
            else:
                assert False
        else:
            assert False
        # print i, v, df2.loc[:,('Personal-status-and-sex')][i], df1.loc[:,('Personal-status-and-sex')][i]

    df1 = pd.get_dummies(df1, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])
    df2 = pd.get_dummies(df2, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])

    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=None)
    df1 = df1.as_matrix()            # converts it to a numpy array which is easier to handle
    df2 = df2.as_matrix()            # converts it to a numpy array which is easier to handle
    z = dataset.target.as_matrix()               # they are changed to numpy matrix in corresondong order, so no problem
    dataset.drop('target', axis=1, inplace=True)
    d1 = dataset.as_matrix()

    num_diffs = []
    for train_index, test_index in rkf.split(d1):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_x, test_x = d1[train_index], d1[test_index]       # use this for training, Personal-status-and-sex is all 0
        train_y, test_y = z[train_index], z[test_index]
        trained_logistic_regression_model = train_logistic_regression(train_x, train_y)     
        # now this model is trained excluding gender stuff
        a = 0
        for i in range(len(test_index)):        # this is 1000 instances, now compare with the predicted outcome for df and real outcome given by dataset
            if trained_logistic_regression_model.predict(df1[int(test_index[i])].reshape(1,-1))[0] != trained_logistic_regression_model.predict(df2[int(test_index[i])].reshape(1,-1))[0]:
                a += 1
        num_diffs.append(a)
        # print a
    # print num_diffs
    print sum(num_diffs)*1.0/len(num_diffs), " now over fairness through unawareness ", len(num_diffs)



# keep Credit-history, credit amount, job and age same, flip other attributes and test. In this case we are testing on equal number of males and females as we will flip the gender for testing and test on both males and females
def conditional_parity(dataset):
    df1 = dataset.copy()         # this is deepcopy, so changing df will not affect dataset
    df2 = dataset.copy()         # I create two copies of dataset, df1 and df2, i will make changes in df2 so that df1 and df2 just differ in Personal-status-and-sex column
    
    dataset = pd.get_dummies(dataset, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"]) 
    # here i take out 16 characteristics, 4 are on which the probability is conditioned, I will flip all these 16 and then test
    
    for i in range(len(dataset)):
        # chk_account = dataset.loc[:, ('Checking-account')][i]
        df1.loc[:, ('Checking-account')][i] = randint(0, 3)      # choose random nos between 1 and 3, which is range for Checking-account 
        df2.loc[:, ('Checking-account')][i] = randint(0, 3)

        # months = dataset.loc[:, ('months')][i]
        df1.loc[:, ('Months')][i] = randint(0, 50)
        df2.loc[:, ('Months')][i] = randint(0, 50)

        # sav_account = dataset.loc[:, ('Savings-account')][i]
        df1.loc[:, ('Savings-account')][i] = randint(1, 5)  # some random value
        df2.loc[:, ('Savings-account')][i] = randint(1, 5)

        # employment_since = dataset.loc[:, ('Present-employment-since')][i]
        df1.loc[:, ('Present-employment-since')][i] = randint(0, 7)     # from the ranges in dataset
        df2.loc[:, ('Present-employment-since')][i] = randint(0, 7)

        # rate = dataset.loc[:, ('Installment-rate')][i]
        df1.loc[:, ('Installment-rate')][i] = randint(1, 6)         # from the ranges in dataset
        df2.loc[:, ('Installment-rate')][i] = randint(1, 6)

        # residence_since = dataset.loc[:, ('Present-residence-since')][i]
        df1.loc[:, ('Present-residence-since')][i] = randint(1, 7)          # from the ranges in dataset
        df2.loc[:, ('Present-residence-since')][i] = randint(1, 7)

        # existing_credits = dataset.loc[:, ('Number-of-existing-credits')][i]
        df1.loc[:, ('Number-of-existing-credits')][i] = randint(1, 5)
        df2.loc[:, ('Number-of-existing-credits')][i] = randint(1, 5)

        # liable = dataset.loc[:, ('Number-of-people-being-liable')][i]
        df1.loc[:, ('Number-of-people-being-liable')][i] = randint(1,3)
        df2.loc[:, ('Number-of-people-being-liable')][i] = randint(1,3)

        # telephone = dataset.loc[:, ('Telephone')][i]
        df1.loc[:, ('Telephone')][i] = randint(1, 2)
        df2.loc[:, ('Telephone')][i] = randint(1, 2)

        # for_worker = dataset.loc[:, ('Foreign-worker')][i]
        df1.loc[:, ('Foreign-worker')][i] = randint(1, 2)
        df2.loc[:, ('Foreign-worker')][i] = randint(1, 2)

        # purpose = dataset.loc[:, ('Purpose')][i], purpose_a47 doesn't exist
        p = ['A40', 'A41', 'A410', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49']
        r = randint(0, 10)
        while(r == 7):
            r = randint(0, 10)
        df1.loc[:, ('Purpose')][i] = 'A4' + str(r)
        
        r = randint(0, 10)
        while(r == 7):
            r = randint(0, 10)
        df2.loc[:, ('Purpose')][i] = 'A4' + str(r)



        # gender = dataset.loc[:, ('Personal-status-and-sex')][i]
        p = ['A91', 'A92', 'A93', 'A94']
        df2.loc[:, ('Personal-status-and-sex')][i] = 'A92'      # female
        r = randint(1, 4)
        while(r == 2):
            r = randint(1, 4)
        df1.loc[:, ('Personal-status-and-sex')][i] = 'A9' + str(r)      # male

        
        # debators = dataset.loc[:, ('Other-debtors')][i]
        p = ['A101', 'A102', 'A103']
        df1.loc[:, ('Other-debtors')][i] = 'A10' + str(randint(1, 3))
        df2.loc[:, ('Other-debtors')][i] = 'A10' + str(randint(1, 3))

        
        # prop = dataset.loc[:, ('Property')][i]
        p = ['A121', 'A122', 'A123', 'A124']
        df1.loc[:, ('Property')][i] = 'A12' + str(randint(1, 4))
        df2.loc[:, ('Property')][i] = 'A12' + str(randint(1, 4))

        
        # installment_plans = dataset.loc[:, ('Other-installment-plans')][i]
        p = ['A141', 'A142', 'A143']
        df1.loc[:, ('Other-installment-plans')][i] = 'A14' + str(randint(1,3))
        df2.loc[:, ('Other-installment-plans')][i] = 'A14' + str(randint(1,3))

        
        # house = dataset.loc[:, ('Housing')][i]
        p = ['A151', 'A152', 'A153']
        df1.loc[:, ('Housing')][i] = 'A15' + str(randint(1,3))
        df2.loc[:, ('Housing')][i] = 'A15' + str(randint(1,3))

        # I am getting problem when entire range of each variable is not generated

        # df3 = pd.get_dummies(df1, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])
        # df4 = pd.get_dummies(df2, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])
        # if len(dataset_headers(df3)) != 49 or len(dataset_headers(df4)) != 49:
        #     print dataset_headers(df3), '\n', dataset_headers(df4), '\n', dataset_headers(dataset)
        #     assert False
        print i
        print len(dataset_headers(df1)), len(dataset_headers(df2)) 
        # now we flip each of these 16 randomly, except for the males and put them determininstically, males in df1 and females df2


    p = ['A40', 'A41', 'A410', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49']
    a = set(df1.loc[:, ('Purpose')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df1.loc[:, ('Purpose')][itr] = l
    
    a = set(df2.loc[:, ('Purpose')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df2.loc[:, ('Purpose')][itr] = l



    # gender = dataset.loc[:, ('Personal-status-and-sex')][i]
    p = ['A91', 'A92', 'A93', 'A94']
    a = set(df2.loc[:, ('Personal-status-and-sex')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df2.loc[:, ('Personal-status-and-sex')][itr] = l

    df1.loc[:, ('Personal-status-and-sex')][i] = 'A9' + str(r)      # male
    a = set(df1.loc[:, ('Personal-status-and-sex')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df1.loc[:, ('Personal-status-and-sex')][itr] = l

    
    # debators = dataset.loc[:, ('Other-debtors')][i]
    p = ['A101', 'A102', 'A103']
    a = set(df1.loc[:, ('Other-debtors')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df1.loc[:, ('Other-debtors')][itr] = l

    a = set(df2.loc[:, ('Other-debtors')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df2.loc[:, ('Other-debtors')][itr] = l

    
    # prop = dataset.loc[:, ('Property')][i]
    p = ['A121', 'A122', 'A123', 'A124']
    a = set(df1.loc[:, ('Property')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df1.loc[:, ('Property')][itr] = l

    a = set(df2.loc[:, ('Property')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df2.loc[:, ('Property')][itr] = l

    
    # installment_plans = dataset.loc[:, ('Other-installment-plans')][i]
    p = ['A141', 'A142', 'A143']
    a = set(df1.loc[:, ('Other-installment-plans')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df1.loc[:, ('Other-installment-plans')][itr] = l

    a = set(df2.loc[:, ('Other-installment-plans')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df2.loc[:, ('Other-installment-plans')][itr] = l

    
    # house = dataset.loc[:, ('Housing')][i]
    p = ['A151', 'A152', 'A153']
    a = set(df1.loc[:, ('Housing')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df1.loc[:, ('Housing')][itr] = l

    a = set(df2.loc[:, ('Housing')].as_matrix())        # for unique elements
    b = list(set(p) - set(a))       # finds the set difference and assigns it from start of dataframe
    for itr, l in enumerate(b):
        df2.loc[:, ('Housing')][itr] = l



    # dataset will be used for training, df1 for males, df2 for females, no need of df3, make edits in df1 and df2 accordingly
    df1 = pd.get_dummies(df1, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])
    df2 = pd.get_dummies(df2, columns = ["Credit-history","Purpose","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job"])
    print dataset_headers(df1), '\n', dataset_headers(df2), '\n', dataset_headers(dataset)
    print len(dataset_headers(df1)), len(dataset_headers(df2)), len(dataset_headers(dataset))
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=None)

    z = dataset.target.as_matrix()               # they are changed to numpy matrix in corresonding order, so no problem
    dataset.drop('target', axis=1, inplace=True)
    df1.drop('target', axis=1, inplace=True)
    df2.drop('target', axis=1, inplace=True)
    # print dataset_headers(df1), '\n', dataset_headers(df2), '\n', dataset_headers(dataset)
    df1 = df1.as_matrix()            # converts it to a numpy array which is easier to handle
    df2 = df2.as_matrix()            # converts it to a numpy array which is easier to handle
    d1 = dataset.as_matrix()


    # for i in range(0, len(dataset)):
    #     if len(df1[i]) != 48:
    #         print i, df1[i], len(df1[i])
    #         assert False
    #     if len(df2[i]) != 48:
    #         print i, df2[i], len(df2[i])
    #         assert False
    # # raw_input()
    
    for train_index, test_index in rkf.split(d1):       # train on dataset
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_x, test_x = d1[train_index], d1[test_index]       # use this for training, Personal-status-and-sex is all 0
        train_y, test_y = z[train_index], z[test_index]
        
        trained_logistic_regression_model = train_logistic_regression(train_x, train_y)     
        # now this model is trained excluding gender stuff
        m, f = 0, 0

        for i in range(len(test_index)):        # this is 1000 instances, now compare with the predicted outcome for df and real outcome given by dataset
            # print trained_logistic_regression_model.predict_proba(df1[int(test_index[i])].reshape(1,-1))[0][1], trained_logistic_regression_model.predict_proba(df2[int(test_index[i])].reshape(1,-1)), trained_logistic_regression_model.predict_proba(df3[int(test_index[i])].reshape(1,-1))[0], raw_input()
            # b = (trained_logistic_regression_model.predict_proba(df2[int(test_index[i])].reshape(1,-1))[0][1] + trained_logistic_regression_model.predict_proba(df3[int(test_index[i])].reshape(1,-1))[0][1])*1.0/2 - trained_logistic_regression_model.predict_proba(df1[int(test_index[i])].reshape(1,-1))[0][1]
            # print df1[int(test_index[i])], df1[i]
            if trained_logistic_regression_model.predict(df1[int(test_index[i])].reshape(1,-1))[0] == 1:
                m += 1
            if trained_logistic_regression_model.predict(df2[int(test_index[i])].reshape(1,-1))[0] == 1:
                f += 1
        conditioned_statistical[0].append(m)
        conditioned_statistical[1].append(f)


    # print conditioned_statistical

    for i in range(len(conditioned_statistical)):
        # print conditioned_statistical[i], sum(conditioned_statistical[i])
        conditioned_statistical[i] = sum(conditioned_statistical[i])*1.0/10000
        print i, conditioned_statistical[i]
    # raw_input()

    # print sum(num_diffs)*1.0/len(num_diffs), " now over fairness through unawareness ", len(num_diffs)



def training(dataset, y):
    global Calibration, NPV, FOR, BNC, BPC, PPV, FDR, FPR, TNR, FNR, TPR, Statistical, overall_accuracy, treatment_equality, weights
    # dataset.drop('Personal-status-and-sex_A91', axis=1, inplace=True) 
    # dataset.drop('Personal-status-and-sex_A92', axis=1, inplace=True) 
    # dataset.drop('Personal-status-and-sex_A93', axis=1, inplace=True) 
    # dataset.drop('Personal-status-and-sex_A94', axis=1, inplace=True) 
    # dataset.drop('target', axis=1, inplace=True) 
    headers = dataset_headers(dataset)
    # print headers, raw_input()
    training_features = headers        # the training should include answers or not depends on which definition of fairness one is using
    target = ['target']
    # print dataset, raw_input()
    # print training_features, target, raw_input()
    # print dataset[dataset.columns[-2]], raw_input()       # used for printing a specific column of dataframe
    # first = dataset.columns.get_loc('Personal-status-and-sex_A91')
    # sec = dataset.columns.get_loc('Personal-status-and-sex_A92')
    # third = dataset.columns.get_loc('Personal-status-and-sex_A93')
    # fourth = dataset.columns.get_loc('Personal-status-and-sex_A94')
    rkf = RepeatedKFold(n_splits=10, n_repeats = 10, random_state=None)
    df = dataset.as_matrix()            # converts it to a numpy array which is easier to handle
    z = y.as_matrix()              # they are changed to numpy matrix in corresondong order, so no problem

    sc = StandardScaler()

    for train_index, test_index in rkf.split(df):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_x, test_x = df[train_index], df[test_index]
        train_y, test_y = z[train_index], z[test_index]
        # print train_x, train_y, raw_input()
        sc.fit(train_x)
        X_train_std = sc.transform(train_x)
        X_test_std = sc.transform(test_x)
        trained_logistic_regression_model = train_logistic_regression(X_train_std, train_y)      # how come not scale the second guys
        # print train_x, train_y, raw_input()
        # print headers, np.transpose(trained_logistic_regression_model.coef_[0])
        coefficients = np.transpose(trained_logistic_regression_model.coef_[0])
        # coefficients = pd.DataFrame({"Feature":dataset.columns,"Coefficients":np.transpose(trained_logistic_regression_model.coef_[0])})
        # print len(coefficients), len(headers), headers
        # raw_input()
        # we can use this to find the average weights assigned to different features while training.
        # print trained_logistic_regression_model.coef_[0]
        for f in range(len(coefficients)):
            if headers[f] in weights:
                weights[headers[f]].append(coefficients[f])
            else:
                weights[headers[f]] = [coefficients[f]]           # create a list here
        
        # rfe = RFE(LogisticRegression(), 10)
        # rfe = rfe.fit(train_x, train_y)
        # # summarize the selection of the attributes
        # for f in range(len(headers)):
        #     if headers[f] in ranks:
        #         ranks[headers[f]].append(rfe.ranking_[f])
        #     else:
        #         ranks[headers[f]] = [rfe.ranking_[f]]
        # print weights, raw_input()

        l = {1:0, 2:0}          # 1 for male, 2 for female : in dataset 1,3,4 is male; 2,5 is female
        r = {1:0, 2:0}
        k = {1:0, 2:0}
        p = {1:0, 2:0}
        q = {1:0, 2:0}
        y_0 = {1:[], 2:[]}      # this will be used to calculate BPC and BNC for y == 0 and y == 1 repectively
        y_1 = {1:[], 2:[]}
        
        # case = np.array([2,12,1567,1,3,1,1,22,1,1,2,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0])
        
        # print trained_logistic_regression_model.predict_proba(case.reshape(1,-1)), trained_logistic_regression_model.predict(case.reshape(1,-1)), raw_input()
        '''
        for i in range(len(test_x)):
            # print test_x.iloc[i]['a9'], trained_logistic_regression_model.predict(test_x.iloc[i].values.reshape(1,-1))[0]     # this prints the result on each individual test case
            # print test_x[i]['Personal-status-and-sex_A92'], raw_input()
            a = test_x[i][first]       # take out this value for each person (i) 
            b = test_x[i][sec]
            c = test_x[i][third]
            d = test_x[i][fourth]
            # e = test_x.iloc[i]['Personal-status-and-sex_A95']     # this is absent
            # print  test_index[i], test_x[i], df[test_index[i]], test_y[i], z[test_index[i]], raw_input()
            # this function returns the probability estimates of each person belonging to target class 1 or 0, this is to be used in place of S
            # print trained_logistic_regression_model.predict_proba(test_x[i].reshape(1,-1)), trained_logistic_regression_model.predict(test_x[i].reshape(1,-1)), raw_input()

            if a == 1 or c == 1 or d == 1:      # for males
                l[1] += 1
                # print test_y[i], raw_input()
                if test_y[i] == 0:         # y == 0
                    y_0[1].append(trained_logistic_regression_model.predict_proba(test_x[i].reshape(1,-1)))
                    # d == 1
                    if trained_logistic_regression_model.predict(test_x[i].reshape(1,-1))[0] == 1:
                        k[1] += 1
                    # d == 0
                    else:
                        r[1] += 1
                
                elif test_y[i] == 1:       # y == 1
                    y_1[1].append(trained_logistic_regression_model.predict_proba(test_x[i].reshape(1,-1)))
                    # d == 1
                    if trained_logistic_regression_model.predict(test_x[i].reshape(1,-1))[0] == 1:
                        p[1] += 1
                    # d == 0
                    else:
                        q[1] += 1
                else:
                    assert False
            
               
            elif b == 1:      # for females
                l[2] += 1
                if test_y[i] == 0:     # y == 0
                    y_0[2].append(trained_logistic_regression_model.predict_proba(test_x[i].reshape(1,-1)))
                    # print y_0[2], trained_logistic_regression_model.predict_proba(test_x.iloc[i].values.reshape(1,-1))
                    # d == 1
                    if trained_logistic_regression_model.predict(test_x[i].reshape(1,-1))[0] == 1:  # S,d == 1
                        k[2] += 1
                    # d == 0
                    else:
                        r[2] += 1

                elif test_y[i] == 1:      # y == 1
                    y_1[2].append(trained_logistic_regression_model.predict_proba(test_x[i].reshape(1,-1)))
                    # d == 1
                    if trained_logistic_regression_model.predict(test_x[i].reshape(1,-1))[0] == 1:  # S,d == 1
                        p[2] += 1
                    # d == 0
                    else:
                        q[2] += 1
                else:
                    assert False

            
            else:
                assert False

        # print l[1], k[1], r[1], p[1], q[1], l[2], k[2], r[2], p[2], q[2]

        # so now i have to print 4 probabilities and 2 inequations
        
        # 1. P(a = 1 | S = 1, G = M ) = P(a = 1 | S = 1, G = F)
        # 2. P(a = 1 | S = 2, G = M ) = P(a = 1 | S = 2, G = F)
        # P(a = 2 | S = 1, G = M) = (r[1])/ (k[1])
        # P(a = 2 | S = 1, G = F) = (r[2])/ (k[2])

        # P(a = 2 | S = 2, G = M) = (p[1])/ (p[1] + r[1])
        # P(a = 2 | S = 2, G = F) = (p[2])/ (p[2] + r[2])
        
        # print "Calibration, S = 0 : ", (r[1])*1.0/ (k[1]), (r[2])*1.0/ (k[2])
        # print "Calibration, S = 1 : ", (p[1])*1.0/ (p[1] + r[1]), (p[2])*1.0/ (p[2] + r[2])
        
        # For calibration value of S has to be chosen, I will do it for upto first decimal place, else it will not have a match for any two individuals, as it is upto 8 digits, no chance
        S, Q = { }, { }
        for dep in range(11):      # these are the values of S on which calibration will be calculated
            S[dep] = 0     # initiate a list for each dep
            Q[dep] = 0
            for pl in y_1[1]:
                if dep == int((float("{0:.1f}".format(pl[0][1])))*10):
                    S[dep] += 1
            for ql in y_0[1]:
                if dep == int((float("{0:.1f}".format(ql[0][1])))*10):
                    Q[dep] += 1
            # print int((float("{0:.1f}".format(pl[0][1])))*10), int((float("{0:.1f}".format(ql[0][1])))*10)           

        M, N = { }, { }
        for dep in range(11):      # these are the values of S on which calibration will be calculated
            M[dep] = 0     # initiate a list for each dep
            N[dep] = 0
            for pl in y_1[2]:
                if dep == int((float("{0:.1f}".format(pl[0][1])))*10):
                    M[dep] += 1
            for ql in y_0[2]:
                if dep == int((float("{0:.1f}".format(ql[0][1])))*10):
                    N[dep] += 1
            # print int((float("{0:.1f}".format(pl[0][1])))*10), int((float("{0:.1f}".format(ql[0][1])))*10)
        

        # add1, add2, add3, add4 = 0, 0, 0, 0
        for dep in range(11):
            if not dep in Calibration[0]:
                Calibration[0][dep] = []
            if not dep in Calibration[1]:
                Calibration[1][dep] = []

            if S[dep]+Q[dep] != 0:
                Calibration[0][dep].append((S[dep]*1.0/(S[dep] + Q[dep])))
            if M[dep]+N[dep] != 0:
                Calibration[1][dep].append((M[dep]*1.0/(M[dep] + N[dep])))
        
        # raw_input()

        # Calibration[0] += (p[1])*1.0/ (p[1] + r[1])
        # Calibration[1] += (p[2])*1.0/ (p[2] + r[2])


        # # so now i want to have 2 estimates and 1 inequation
        # # E(S | Y = 0, G = M) = E(S | Y = 0, G = F)
        # # E(S | Y = 0, G = M) = (r[1]*2 + (k[1] - r[1])) / (k[1]) = (k[1] + r[1])/ (k[1])
        # # E(S | Y = 0, G = F) = (r[2]*2 + (k[2] - r[2])) / (k[2]) = (k[2] + r[2])/ (k[2])
        
        # # This is for balance for positive class
        # # E(S | Y = 1, G = M) = E(S | Y = 1, G = F)
        # # E(S | Y = 1, G = M) = (p[1]*2 + q[1]) / (p[1] + q[1])
        # # E(S | Y = 1, G = F) = (p[2]*2 + q[2]) / (p[2] + q[2])

        
        # print y_0
        # print y_1
        # raw_input()

        # # print "Balance for positive class : ", (p[1]*2 + q[1])*1.0/ (p[1] + q[1]), (p[2]*2 + q[2])*1.0/ (p[2] + q[2])
        add1 = 0
        for it in y_1[1]:
            add1 += it[0][1]        # 1 is the score for d = 1, i can choose score for d = 0 as well, they are just 1-x and x basically
        add2 = 0
        for it in y_1[2]:
            add2 += it[0][1]
        
        if len(y_1[1]) != 0: 
            BPC[0].append(add1*1.0/len(y_1[1]))
        if len(y_1[2])!= 0:
            BPC[1].append(add2*1.0/len(y_1[2]))

        # # print "Balance for negative class : ", (k[1] + r[1])*1.0/ (k[1]), (k[2] + r[2])*1.0/ (k[2])
        add1 = 0
        for it in y_0[1]:
            add1 += it[0][0]
        add2 = 0
        for it in y_0[2]:
            add2 += it[0][0]

        if len(y_0[1]) != 0:
            BNC[0].append(add1*1.0/len(y_0[1]))
        if len(y_0[2]) != 0:
            BNC[1].append(add2*1.0/len(y_0[2]))

        # print len(y_1[1]), len(y_0[1]), l[1], len(y_0[2]), len(y_1[2]), l[2], raw_input()

        # This is for equal NPV's and equal FOR's
        # so here it is P(Y = 0| d = 0, G = M) = P(Y = 0| d = 0, G = F)  : equal NPV's
        # P(Y = 0 | d = 0, G = M) = (r[1]) / (k[1] - r[1] + q[1])
        # P(Y = 0 | d = 0, G = F) = (r[2]) / (k[2] - r[2] + q[2])

        # so here is it P(Y = 1| d = 0, G = M) = P(Y = 1| d = 0, G = F)  : equal FOR's
        # P(Y = 1 | d = 0, G = M) = (q[1]) / (k[1] - r[1] + q[1])
        # P(Y = 1 | d = 0, G = F) = (q[2]) / (k[2] - r[2] + q[2])

        # print "NPV's : ", (r[1])*1.0 / (k[1] - r[1] + q[1]), (r[2])*1.0 / (k[2] - r[2] + q[2])
        if (r[1] + q[1]) != 0:
            NPV[0].append((r[1])*1.0 / (r[1] + q[1]))
        if (r[2] + q[2]) != 0:
            NPV[1].append((r[2])*1.0 / (r[2] + q[2]))

        # print "FOR's : ", (q[1])*1.0 / (k[1] - r[1] + q[1]), (q[2])*1.0 / (k[2] - r[2] + q[2])
        if (r[1] + q[1]) != 0:
            FOR[0].append((q[1])*1.0 / (r[1] + q[1]))
        if (r[2] + q[2]) != 0:
            FOR[1].append((q[2])*1.0 / (r[2] + q[2]))



        # equal PPV and equal FDR's
        # so here it is P(Y = 1| d = 1, G = M) = P(Y = 1| d = 1, G = F)  : equal PPV's
        # P(Y = 1| d = 1, G = M) = (p[1]) / (p[1] + r[1])
        # P(Y = 1| d = 1, G = F) = (p[2]) / (p[2] + r[2])

        # so here it is P(Y = 0| d = 1, G = M) = P(Y = 0| d = 1, G = F)  : equal FDR's
        # P(Y = 0| d = 1, G = M) = (r[1]) / (p[1] + r[1])
        # P(Y = 0| d = 1, G = F) = (r[2]) / (p[2] + r[2])

        # print "PPV's : ", (p[1])*1.0 / (p[1] + r[1]), (p[2])*1.0 / (p[2] + r[2])
        if (p[1] + k[1]) != 0:
            PPV[0].append((p[1])*1.0 / (p[1] + k[1]))
        if (p[2] + k[2]) != 0:
            PPV[1].append((p[2])*1.0 / (p[2] + k[2]))

        # print "FDR's : ", (r[1])*1.0 / (p[1] + r[1]), (r[2])*1.0 / (p[2] + r[2])
        if (p[1] + k[1]) != 0:
            FDR[0].append((k[1])*1.0 / (p[1] + k[1]))
        if (p[2] + k[2]) != 0:
            FDR[1].append((k[2])*1.0 / (p[2] + k[2]))

        # This is for error rate balance (equal FPR's)
        # so here it is P(d = 1 | Y = 0, G = M) = P(d = 1 | Y = 0, G = F) : equal FPR
        # P(d = 1 | Y = 0, G = M) = r[1] / k[1]
        # P(d = 1 | Y = 0, G = F) = r[2] / k[2]

        # so here it is P(d = 0 | Y = 0, G = M) = P(d = 0 | Y = 0, G = F) : equal TNR
        # P(d = 0 | Y = 0, G = M) = (k[1] - r[1]) / k[1]
        # P(d = 0 | Y = 0, G = F) = (k[2] - r[2]) / k[2]

        # print "Equal FPR's : ", r[1]*1.0 / k[1], r[2]*1.0 / k[2]
        if (r[1] + k[1]) != 0:
            FPR[0].append(k[1]*1.0 / (k[1] + r[1]))
        if (r[2] + k[2]) != 0:
            FPR[1].append(k[2]*1.0 / (k[2] + r[2]))

        # print "Equal TNR's : ", (k[1] - r[1])*1.0 / k[1], (k[2] - r[2])*1.0 / k[2]
        if (r[1] + k[1]) != 0:
            TNR[0].append(r[1]*1.0 / (k[1] + r[1]))
        if (r[2] + k[2]) != 0:
            TNR[1].append(r[2]*1.0 / (k[2] + r[2]))

        
        # This is for error rate balance (equal FNR's)
        # so here it is P(d = 0 | Y = 1, G = M) = P(d = 0 | Y = 1, G = F) : equal FNR
        # P(d = 0 | Y = 1, G = M) = (q[1]) / (p[1] + q[1])
        # P(d = 0 | Y = 1, G = F) = (q[2]) / (p[2] + q[2])

        # so here it is P(d = 1 | Y = 1, G = M) = P(d = 1 | Y = 1, G = F) : equal TPR
        # P(d = 1 | Y = 1, G = M) = (p[1]) / (p[1] + q[1])
        # P(d = 1 | Y = 1, G = F) = (p[2]) / (p[2] + q[2])

        # print "Equal FNR's : ", (q[1])*1.0 / (p[1] + q[1]), (q[2])*1.0 / (p[2] + q[2])
        if (p[1] + q[1]) != 0:
            FNR[0].append((q[1])*1.0 / (p[1] + q[1]))
        if (p[2] + q[2]) != 0:
            FNR[1].append((q[2])*1.0 / (p[2] + q[2]))

        # print "Equal TPR's : ", (p[1])*1.0 / (p[1] + q[1]), (p[2])*1.0 / (p[2] + q[2])
        if (p[1] + q[1]) != 0:
            TPR[0].append((p[1])*1.0 / (p[1] + q[1]))
        if (p[2] + q[2]) != 0:
            TPR[1].append((p[2])*1.0 / (p[2] + q[2]))

        # This is for Statistical Parity
        # so here it is P(d = 1 | G = M) = P(d = 1 | G = F)
        # P(d = 1 | G = M) = (p[1] + r[1]) / l[1]
        # P(d = 1 | G = F) = (p[2] + r[2]) / l[2]

        # print "Statistical Parity : ", (p[1] + r[1])*1.0 / l[1], (p[2] + r[2])*1.0 / l[2]
        if (l[1]) != 0:
            Statistical[0].append((p[1] + k[1])*1.0 / l[1])
        if (l[2]) != 0:
            Statistical[1].append((p[2] + k[2])*1.0 / l[2])

        # We can decide legitimate factors for conditional statistical parity

        # This is for overall accuracy equality
        # so here it is P(d = Y, G = M) = P(d = Y, G = F)
        # P(d = Y, G = M) = (p[1] + k[1] - r[1]) / l[1]
        # P(d = Y, G = F) = (p[2] + k[2] - r[2]) / l[2]
        # print "Overall accuracy equality : ", (p[1] + k[1] - r[1])*1.0 / l[1], (p[2] + k[2] - r[2])*1.0 / l[2]
        if (l[1]) != 0:
            overall_accuracy[0].append((p[1] + r[1])*1.0 / l[1])
        if (l[2]) != 0:
            overall_accuracy[1].append((p[2] + r[2])*1.0 / l[2])

        # This is for treatment equality(equal FN/FP)
        # P(d=0|Y=1,G=M)/P(d=1,Y=0,G=M) = P(d=0,|Y=1,G=F)/P(d=1|Y=0,G=F)
        # P(d=0|Y=1,G=M) = (q[1]) / (p[1] + q[1])
        # P(d=1,Y=0,G=M) = r[1] / k[1]
        # P(d=0,|Y=1,G=F) = (q[2]) / (p[2] + q[2])
        # P(d=1|Y=0,G=F) = (r[2] / k[2])

        # print "Treatment equality : ", ((q[1])*1.0 / (p[1] + q[1]))*1.0/(r[1]*1.0 / k[1]), ((q[2])*1.0 / (p[2] + q[2]))*1.0/((r[2]*1.0 / k[2]))
        if (k[1]) != 0:
            treatment_equality[0].append(q[1]*1.0 / k[1])
        if (k[2]) != 0:
            treatment_equality[1].append(q[2]*1.0 / k[2])
    '''
    cal_avg(100, headers)



def main():
    """
    Logistic Regression classifier main
    :return:
    """
    # Load the data set for training and testing the logistic regression classifier
    dataset = pd.read_csv(DATA_SET_PATH)
    # data = dataset.dropna()
    # print "Number of Observations :: ", len(dataset)
    # print(dataset.shape)
    # print(list(dataset.columns))
    # sns.countplot(x="default", data=data)
    # plt.show()
    # print data.isnull().sum()

    # Get the first observation
    # print dataset.head()
    # data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)
    # headers = dataset_headers(dataset)
    # print "Data set headers :: {headers}".format(headers=headers)

    # print headers[:-1], raw_input()
    # data2 = pd.get_dummies(dataset, columns = ["Checking-account","Credit-history","Purpose","Savings-account","Present-employment-since","Personal-status-and-sex","Other-debtors","Property","Other-installment-plans","Housing","Job","Telephone","Foreign-worker"])
    
    # dataset = data2

    # print dataset.columns, raw_input()
    # data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)


    # training_features = ['TVnews', 'PID', 'age', 'educ', 'income']
    # training_features = headers[:-2]
    # training_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', 'Female', 'Male', 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    # training_features = ['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
    
    # i am putting job in this list because i don't know if distance between "unskilled-resident" and "skilled" is same as between "skilled" and "super-skilled"
    # dataset.ix[dataset.target > 0, 'target'] = 1
    # dataset.ix[dataset.target < 0, 'target'] = 0
    dataset.target.replace([1, 2], [1, 0], inplace=True)        # this replaces 1 (good credit) with 1 and 2 (bad credit) with 0


    # now i will also measure the fairness through unawareness, generate 1000 instances, one for each 1000 instances and then evaluate the fairness.

    # test_aware(dataset)

    # test_unaware(dataset)   # this will be used for testing for fairness through unawareness
    # create_tests_causal(dataset)


    tar = dataset.target  # target is separated
    # print tar, raw_input()
    dataset.drop('target', axis=1, inplace=True)        # removes target from it, so now dataset just contains the columns on which it has to be trained
    dataset.drop('Personal-status-and-sex', axis=1, inplace=True)
    # dataset = pd.get_dummies(dataset, columns = ["Credit-history", "Purpose", "Other-debtors", "Property", "Other-installment-plans", "Housing", "Job"])
    dataset = pd.get_dummies(dataset, columns = ["Purpose", "Credit-history", "Other-debtors", "Property", "Other-installment-plans", "Housing", "Job", "Present-employment-since"])
    print(len(dataset.columns))
    headers = dataset_headers(dataset)
    
    # trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
    # conditional_parity(dataset)
    # print headers
    # scaler = StandardScaler()         # this was used for normalisation of data
    # scaler.fit(dataset)
    # # # print scaler.mean_
    # dataset = scaler.transform(dataset)
    

    # rkf = RepeatedKFold(n_splits=10, n_repeats = 1, random_state=None)
    # df = dataset            # converts it to a numpy array which is easier to handle
    # z = tar.as_matrix()               # they are changed to numpy matrix in corresondong order, so no problem

    # for train_index, test_index in rkf.split(df):
    #     train_x, test_x = df[train_index], df[test_index]
    #     train_y, test_y = z[train_index], z[test_index]
    #     # create the RFE model and select 3 attributes
    #     rfe = RFE(LogisticRegression(), 10)
    #     rfe = rfe.fit(train_x, train_y)
    #     # summarize the selection of the attributes
    #     for f in range(len(headers)):
    #         if headers[f] in ranks:
    #             ranks[headers[f]].append(rfe.ranking_[f])
    #         else:
    #             ranks[headers[f]] = [rfe.ranking_[f]]
    

        # raw_input()
        # trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
    # train_x, test_x, train_y, test_y = train_test_split(dataset[headers], tar, train_size=0.80)
        # print i, raw_input()
    # print type(dataset), raw_input()
    
        # print rfe.support_
        # print rfe.ranking_
    training(dataset, tar)        # this is the main training method
    
    # print dataset[0][47]
    # np.savetxt("hello.txt", dataset, delimiter=',')
    # fh = open("hello.txt", "w")
    # lines_of_text = ["a line of text", "another line of text", "a third line"]
    # print dataset, raw_input()

    # fh.writelines(dataset)
    # fh.close()

    
    # calculate base rate for male and female
    # m, f = {0:0, 1:0}, {0:0, 1:0}
    # for i in range(len(dataset)):
    #     a = dataset.iloc[i]['Personal-status-and-sex']
    #     if a == 'A91' or a == 'A93' or a == 'A94':
    #         m[0] += 1
    #         if dataset.iloc[i]['target'] == 1:
    #             m[1] += 1
    #     elif a == 'A92':
    #         f[0] += 1
    #         if dataset.iloc[i]['target'] == 1:
    #             f[1] += 1
    #     else:
    #         assert False
    # print "The base rates for males and females :", m, f 


if __name__ == "__main__":
    main()
