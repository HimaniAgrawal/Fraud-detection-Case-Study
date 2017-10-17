# Data cleaning and feature engineering
import numpy as np
import pandas as pd

def quantityTotal(x):
    lst = [i["quantity_total"] for i in x]
    return sum(lst)

def feature_engineering(df):
    '''
    Reads data from a pandas dataframe.
    Adds a column 'is_fraud', which labels if a particular transaction is fraud
    Creates new dataframe with feature selection and feature engineering, while
    dealing with the missing values.
    Returns new dataframe which is ready to be split into features and labels
    data.

    Input:
        df: pandas dataframe
    Returns:
        new_df: pandas dataframe
    '''

    # Adding fraud column to df, to determine if an event is fraud
    df['is_fraud'] = df.acct_type.isin([u'fraudster_event', u'fraudster', \
        u'fraudster_att'])

    # Feature engineering on df
    df= df[df['user_type'].isin([1,3,4,5])]
    df['previous_payouts'] = df['previous_payouts'].apply(lambda x : 1 if len(x)>=1 else 0)
    df["quantity_total"] = df["ticket_types"].apply(quantityTotal)
    df['generic_email']=df['email_domain'].apply(lambda x : 0 if  \
        all(s not in x for s in ['aol', 'gmail', 'live','hotmail','yahoo', \
        'outlook']) else 1)
    df['delivery_method'] = df['delivery_method'].apply(lambda x: 1 if x>=1.0 else 0)
    df = pd.get_dummies(df, columns = ['user_type'])

    # Get sub-selection of columns from df to create new_df
    new_df = df[['generic_email','previous_payouts','delivery_method', \
        'sale_duration2','quantity_total','user_type_1','user_type_3', \
        'user_type_4','user_type_5','is_fraud']]

    return new_df
