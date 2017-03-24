import pandas as pd
import us
# Mapping for grade assigned by bank
grade_map = {
    'A1': 0,
    'A2': 1,
    'A3': 2,
    'A4': 3,
    'A5': 4,
    'B1': 5,
    'B2': 6,
    'B3': 7,
    'B4': 8,
    'B5': 9,
    'C1': 10,
    'C2': 11,
    'C3': 12,
    'C4': 13,
    'C5': 14,
    'D1': 15,
    'D2': 16,
    'D3': 17,
    'D4': 18,
    'D5': 19,
    'E1': 20,
    'E2': 21,
    'E3': 22,
    'E4': 23,
    'E5': 24,
    'F1': 25,
    'F2': 26,
    'F3': 27,
    'F4': 28,
    'F5': 29,
    'G1': 30,
    'G2': 31,
    'G3': 32,
    'G4': 33,
    'G5': 34
}
ownership_map = {
    'OWN' : 0,
    'RENT' : 1,
    'MORTGAGE' : 2,
    'NONE': 3,
    'OTHER': 4,
    'ANY': 5
}
verification_status_map = {
    'Source Verified': 0,
    'Not Verified': 1,
    'Verified': 2
}
purpose_map = {
    'credit_card': 0,
    'debt_consolidation': 1,
    'educational': 2,
    'home_improvement': 3,
    'house': 4,
    'major_purchase': 5,
    'medical': 6,
    'moving': 7,
    'other': 8,
    'renewable_energy': 9,
    'small_business': 10,
    'vacation': 11,
    'wedding': 12,
    'car': 13
}
list_status_map = {
    'f' : 0,
    'w' : 1
}
application_type_map = {
    'INDIVIDUAL': 0,
    'JOINT': 1
}
paymnt_plan_map = {
    'n': 0,
    'y': 1
}


def read_file(train_file):
    """
    Read csv file
    :param train_file:
    :return:
    """
    dataframe = pd.read_csv(train_file, delimiter=",")
    return dataframe


def remove_redundant_fields(dataframe, *args):
    """
    Remove Fields from Dataframe
    :param dataframe:
    :param args:
    :return:
    """
    for arg in args:
        try:
            del dataframe[arg]
        finally:
            print "Field does not exist or is already deleted {0}".format(arg)

    return dataframe
# Convert text values to numerical categories


def convert_states(state):
    state_object = us.states.lookup(state)
    return int(state_object.fips)


def transform_columns(dataframe):
    dataframe['sub_grade'] = pd.to_numeric(dataframe['sub_grade'].replace(grade_map))
    dataframe['home_ownership'] = pd.to_numeric(dataframe['home_ownership'].replace(ownership_map))
    dataframe['verification_status'] = pd.to_numeric(dataframe['verification_status'].replace(verification_status_map))
    dataframe['pymnt_plan'] = pd.to_numeric(dataframe['pymnt_plan'].replace(paymnt_plan_map))
    dataframe['purpose'] = pd.to_numeric(dataframe['purpose'].replace(purpose_map))
    dataframe['initial_list_status'] = pd.to_numeric(dataframe['initial_list_status'].replace(list_status_map))
    dataframe['application_type'] = pd.to_numeric(dataframe['application_type'].replace(application_type_map))
    return dataframe