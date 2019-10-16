import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# prepare user data
def prepare_user_data():
    user_data = pd.read_table('data/m_user_candidate.tsv')
    columns_arr = ['id',
                   'm_country_code',
                   'm_province_code',
                   'current_salary',
                   'm_current_currency_code',
                   'expected_salary',
                   'm_currency_code',
                   'years_of_experience',
                   'vn_job_category_code',
                   'vn_job_sub_category_code',
                   'vn_job_category_code.1',
                   'vn_job_sub_category_code.1',
                   'm_position_code',
                   ]
    user_data = user_data[columns_arr]
    user_data.rename({
        'vn_job_category_code' : 'resume_job_category_code',
        'vn_job_sub_category_code' : 'resume_job_sub_category_code',
        'vn_job_category_code.1' : 'resume_career_job_category_code',
        'vn_job_sub_category_code.1' : 'resume_career_job_sub_category_code',
    }, axis=1, inplace=True)

    user_data = user_data.fillna(0)
    user_data = user_data.apply(lambda row: set_user_salary(row), axis=1)

    user_data.id = user_data.id.astype(int)
    user_data.current_salary = user_data.current_salary.astype(int)
    user_data.expected_salary = user_data.expected_salary.astype(int)
    user_data.resume_job_category_code = user_data.resume_job_category_code.astype(int)
    user_data.resume_job_sub_category_code = user_data.resume_job_sub_category_code.astype(int)
    user_data.resume_career_job_category_code = user_data.resume_career_job_category_code.astype(int)
    user_data.resume_career_job_sub_category_code = user_data.resume_career_job_sub_category_code.astype(int)
    user_data.m_position_code = user_data.m_position_code.astype(int)

    return user_data


def set_user_salary(row):
    if row.m_current_currency_code == 'USD':
        row.current_salary = row.current_salary * 23000

    if row.m_currency_code == 'USD':
        row.expected_salary = row.expected_salary * 23000

    return row


user_data = prepare_user_data()


# prepare job data
def prepare_job_data():
    job_data = pd.read_table('data/t_job.tsv', low_memory=False)

    job_data = job_data.fillna(0)
    job_data = job_data.apply(lambda row: set_salary(row), axis=1)

    job_data.id = job_data.id.astype(int)
    job_data.vn_job_category_code = job_data.vn_job_category_code.astype(int)
    job_data.vn_job_sub_category_code = job_data.vn_job_sub_category_code.astype(int)
    job_data.salary_from = job_data.salary_from.astype(int)
    job_data.salary_to = job_data.salary_to.astype(int)
    job_data.negotiable_flag = job_data.negotiable_flag.astype(int)

    job_data = job_data.dropna()

    job_data = job_data[job_data['id'].apply(lambda x: str(x).isdigit())]

    return job_data


def set_salary(row):
    if row.m_currency_code == 'USD':
        row.salary_from = row.salary_from * 23000
        row.salary_to = row.salary_to * 23000

    if row.m_currency_code == 'JPY':
        row.salary_from = row.salary_from * 214.023256
        row.salary_to = row.salary_to * 214.023256

    return row


job_data = prepare_job_data()


# prepare application data
def prepare_application_data():
    application_data = pd.read_table('data/t_app.tsv')

    columns_arr = ['m_user_candidate_id','t_job_id']
    application_data = application_data[columns_arr]

    application_data = application_data.dropna()

    application_data = application_data[application_data['m_user_candidate_id'].apply(lambda x: str(x).isdigit())]

    application_data.m_user_candidate_id = application_data.m_user_candidate_id.astype(int)
    application_data.t_job_id = application_data.t_job_id.astype(int)

    application_data['is_matching_job_category_expected'] = 0
    application_data['is_matching_job_category_history'] = 0
    application_data['salary_from_greater_than_current_salary'] = 0
    application_data['salary_to_greater_than_current_salary'] = 0
    application_data['salary_from_greater_than_expected_salary'] = 0
    application_data['salary_to_greater_than_expected_salary'] = 0
    application_data['matching_job_workplace'] = 0
    application_data['job_apply'] = 1

    return application_data


application_data = prepare_application_data()


def set_matching_job_category_expected(row):
    candidate_id = row['m_user_candidate_id']
    job_id = row['t_job_id']
    expected_category = user_data.loc[user_data['id'] == int(candidate_id)].resume_job_category_code.unique().tolist()
    job_category = job_data.loc[job_data['id'] == int(job_id)].vn_job_category_code

    if (len(job_category) and len(expected_category)):
        job_category = int(job_category.values[0])

        if (job_category in expected_category) :
            row['is_matching_job_category_expected'] = 1

    return row


application_data = application_data.apply(lambda row : set_matching_job_category_expected(row), axis=1)


def set_matching_job_category_history(row):
    candidate_id = row['m_user_candidate_id']
    job_id = row['t_job_id']
    expected_category = user_data.loc[user_data['id'] == int(candidate_id)].resume_career_job_category_code.unique().tolist()
    job_category = job_data.loc[job_data['id'] == int(job_id)].vn_job_category_code

    if len(job_category) and len(expected_category):
        job_category = int(job_category.values[0])

        if job_category in expected_category:
            row['is_matching_job_category_history'] = 1

    return row


application_data = application_data.apply(lambda row : set_matching_job_category_history(row), axis=1)


def set_matching_salary(row):
    candidate_id = row['m_user_candidate_id']
    job_id = row['t_job_id']
    current_salary = user_data.loc[user_data['id'] == int(candidate_id)].current_salary.unique()
    expected_salary = user_data.loc[user_data['id'] == int(candidate_id)].expected_salary.unique()
    job_salary_from = job_data.loc[job_data['id'] == int(job_id)].salary_from.unique()
    job_salary_to = job_data.loc[job_data['id'] == int(job_id)].salary_to.unique()
    is_negotiable = job_data.loc[job_data['id'] == int(job_id)].negotiable_flag.unique()

    if int(is_negotiable) != 1:
        row['salary_from_greater_than_current_salary'] = 0
        row['salary_to_greater_than_current_salary'] = 0
        row['salary_from_greater_than_expected_salary'] = 0
        row['salary_to_greater_than_expected_salary'] = 0

    if len(current_salary) and len(job_salary_from) and job_salary_from[0] > current_salary[0]:
        row['salary_from_greater_than_current_salary'] = 1
        row['salary_to_greater_than_current_salary'] = 1

    if len(current_salary) and len(job_salary_to) and job_salary_to[0] > current_salary[0]:
        row['salary_to_greater_than_current_salary'] = 1

    if len(expected_salary) and len(job_salary_from) and job_salary_from[0] > expected_salary[0]:
        row['salary_from_greater_than_expected_salary'] = 1
        row['salary_to_greater_than_expected_salary'] = 1

    if len(current_salary) and len(job_salary_to) and job_salary_to[0] > expected_salary[0]:
        row['salary_to_greater_than_expected_salary'] = 1

    return row


application_data = application_data.apply(lambda row : set_matching_salary(row), axis=1)


def set_matching_workplace(row):
    candidate_id = row['m_user_candidate_id']
    job_id = row['t_job_id']
    expected_workplace = user_data.loc[user_data['id'] == candidate_id].m_province_code.unique().tolist()
    job_workplace = job_data.loc[job_data['id'] == job_id].m_province_code.unique().tolist()

    print(expected_workplace)
    print(job_workplace)

    if len(expected_workplace) and len(job_workplace):
        matching_list = list(set(job_workplace).intersection(expected_workplace))

        if len(matching_list):
            print(matching_list)
            row['matching_job_workplace'] = 1

    return row


application_data = application_data.apply(lambda row : set_matching_workplace(row), axis=1)

X = application_data[['is_matching_job_category_expected',
                     'is_matching_job_category_history',
                     'salary_from_greater_than_expected_salary',
                     'salary_from_greater_than_expected_salary',
                     'salary_to_greater_than_expected_salary',
                     'salary_to_greater_than_current_salary',
                     'matching_job_workplace']].values
Y = application_data['job_apply'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=9)

regressor = LinearRegression(fit_intercept=False)
regressor.fit(X_train, Y_train) #training the algorithm

Y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

corr, _ = pearsonr(data1, data2)

regressor.coef_

