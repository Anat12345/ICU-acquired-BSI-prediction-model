'''
The code for building features based on the time series
all time series is calculate for specific window - [3,5]
all the features is split to few functions: prepare_...
'''

import ast
import Handle_promateus_data
import create_daily_data
import create_liquids_file
from dateutil.relativedelta import relativedelta

from datetime import date, timedelta
import CreateFeaturesFile
import math
import Utils
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sqlite3
import Signal_source
import sqlite3

output_dir = 'C://Users//michael//Desktop//predicting BSI//files//Output//'
data_dir = 'C://Users//michael//Desktop//predicting BSI//files//Data//'
#used for windows for blood pressure
def create_short_signal_features (control_file = output_dir+'all_features_20180403211106.csv',
                                        control_df_input = None,
                                        clean_noise_flag = True,
                                        num_days = 3):

    list_signals_param = list()
    # read full file and not partial!!!
    signal_source_temp = Signal_source.Signal_source(signal_name = 'Temperature', signal_file = 'C://Users//michael//Desktop//predicting BSI//files//Data//temperature_12_17.xlsx',min_value = 33, max_value = 42)
    list_signals_param.append(signal_source_temp)
    signal_source_diastolic = Signal_source.Signal_source(signal_name='Arterial Pressure Diastolic', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//diastolic_p_12_17.xlsx',min_value = 40, max_value=220)
    list_signals_param.append(signal_source_diastolic)
    # signal_source = Signal_source.Signal_source(signal_name='Uretheral Catheter', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//urethral_cath_12_17.xlsx')
    # list_signals_param.append(signal_source)
    # signal_source = Signal_source.Signal_source(signal_name='SaO2 (Systemic)', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//O2_STURATION_12_17.xlsx')
    # list_signals_param.append(signal_source)
    # signal_source = Signal_source.Signal_source(signal_name='HR -EKG', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//heart_rate_ecg_12_17.xlsx')
    # list_signals_param.append(signal_source)
    # signal_source = Signal_source.Signal_source(signal_name='Mean Arterial Pressure', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//mean_p_12_17.xlsx')
    # list_signals_param.append(signal_source)
    signal_source = Signal_source.Signal_source(signal_name='Arterial Pressure Systolic', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//systolic_p_12_17.xlsx')
    list_signals_param.append(signal_source)



    dict_commands_param = {}
    if control_df_input is None:
        control_df = pd.read_csv(control_file, encoding="ISO-8859-1")
    else :
        control_df = control_df_input
    df_plus_new_features = control_df[['patient_id','date_sample']]
    for signal_source in list_signals_param:
        print (' start signal - '+signal_source.signal_name)
        short_df = Utils.read_data(signal_source.signal_file)

        features_dict_list = list()
        for index, row in control_df[['date_sample','patient_id']].drop_duplicates().iterrows():
            features_dict = {}
            filter_short_df = short_df.loc[(short_df.patient_id ==row.patient_id)&(short_df.time <= datetime.strptime(str(row.date_sample),"%Y-%m-%d %H:%M:%S")) & (short_df.time>=datetime.strptime(str(row.date_sample),"%Y-%m-%d %H:%M:%S")- timedelta(days = num_days)),:]
            filter_short_df = filter_short_df.loc[(filter_short_df.value >= signal_source.min_value) & (filter_short_df.value <= signal_source.max_value), :]
            # print (' size of filter_short_df = '+ str(filter_short_df.shape))
            if filter_short_df.shape[0]>4:
                features_dict.update(build_hist_gradient_signal(signal_source.signal_name, filter_short_df))
                features_dict['patient_id'] = row.patient_id
            else :
                print (' no '+ signal_source.signal_name + ' found for '+ row.patient_id)
                continue
            features_dict_list.append(features_dict)
        features_df = pd.DataFrame(features_dict_list)
        signal_source.df_source = features_df
        features_df.to_csv('signal_source_'+signal_source.signal_name+'.csv')

        df_plus_new_features = pd.merge(df_plus_new_features, features_df, how='left', on=['patient_id'])

    df_plus_new_features.to_csv(output_dir+'short_signals_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv')
    return df_plus_new_features
# use to splir the values to different ranges
def build_hist_gradient_signal(parameter_name,signal_df):

    features_dict = {}
    signal_df_param = signal_df[signal_df[Utils.signal_parameter_column_name] == parameter_name]
    signal_df_param['time'] = pd.to_datetime(signal_df_param[Utils.signal_time_column_name],format='%d%b%Y:%H:%M:%S.%f')

    signal_df_param['prev_value'] = signal_df_param.groupby('patient_id')['value'].shift(1)
    signal_df_param['prev_time'] = signal_df_param.groupby('patient_id')['time'].shift(1)

    signal_df_param['gradient'] = (signal_df_param['value'] - signal_df_param['prev_value'] )/ round((signal_df_param['time'].subtract(signal_df_param['prev_time'])).dt.total_seconds() / ( 60 *60),1)

    signal_df_param['change'] =  signal_df_param['gradient']/signal_df_param['prev_value']
    # signal_df_param.to_csv('new_short_signal.csv')

    signal_df_no_inf   = signal_df_param[~signal_df_param['change'].isnull()]
    signal_df_no_inf = signal_df_no_inf[~(signal_df_no_inf['change']==np.inf)]
    signal_df_no_inf = signal_df_no_inf[~(signal_df_no_inf['change'] == -np.inf)]

    try:
        count, division = np.histogram(signal_df_no_inf['change'], bins = [-0.2,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.2])
        # signal_df_no_inf['change'].hist(bins=division)
        count = np.nan_to_num(count)
        for i,div in enumerate(division):
            if i>= len(count):
                pass
            else:
                features_dict[parameter_name+'_pct_'+str(div)] = count[i]/len(signal_df_no_inf['change'])
    except Exception as e:
        print('no summary for '+signal_df_param['patient_id'].head(1))
        print (e)
        pass



    group_all_dates = {Utils.signal_value_column_name:{'median_all_' + parameter_name :'median','std_all_' + parameter_name :'std',
                                                       'mean_all_' + parameter_name:'mean', 'min_all_' + parameter_name:'min', 'max_all_' + parameter_name:'max', 'count_all_' + parameter_name: 'count'}}
    try:
        df_group_all_dates = signal_df_param.groupby([Utils.signal_id_column_name]).agg(group_all_dates)
        df_group_all_dates.columns = df_group_all_dates.columns.droplevel(0)
        for index,row in df_group_all_dates.iterrows():
            for key in df_group_all_dates.keys():
                features_dict[key] = row[key]
    except Exception:
        print('no summary for '+key)
        pass

    return features_dict

class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k-2))

# the functions finds the signal avg value in hours without specific medicine (for example, diastoloc pressure without vasoactive medicine
def get_basic_blood_pressure (control_df,blood_pressure_file = data_dir+'diastolic_p_12_17_partial.xlsx',
                              medicine_file = data_dir+'vasoactive_icu_12_17_for_modeling_partial.xlsx',
                              parameter_name = 'Arterial Pressure Diastolic',  min_value = 10,
                              max_value = 250, num_days =30,signals_dir= 'signals//'):

    blood_pressure_df = Utils.read_data(blood_pressure_file)
    medicine_df = Utils.read_data(medicine_file)


    features_dict_list = list()
    for index, row in control_df[['date_sample', 'patient_id']].drop_duplicates().iterrows():

        # for partial data testing
        # if row.patient_id in ['1448850-6','1463525-4','20062233-0','2041284-7']:

        blood_pressure_per_patient = blood_pressure_df.loc[(blood_pressure_df.patient_id == row.patient_id) & (blood_pressure_df.time <= datetime.strptime(str(row.date_sample), "%Y-%m-%d %H:%M:%S")) & (
                blood_pressure_df.time >= datetime.strptime(str(row.date_sample),"%Y-%m-%d %H:%M:%S") - timedelta(days=num_days)), :]
        blood_pressure_per_patient = blood_pressure_per_patient.loc[(blood_pressure_per_patient.value > min_value) & ( blood_pressure_per_patient.value < max_value), :]

        medicine_df_per_patient = medicine_df.loc[(medicine_df.patient_id == row.patient_id) & (medicine_df.start_date <= datetime.strptime(str(row.date_sample), "%Y-%m-%d %H:%M:%S")) & (
                medicine_df.end_date >= datetime.strptime(str(row.date_sample),"%Y-%m-%d %H:%M:%S") - timedelta(days=num_days)), :]

        # Make the db in memory
        with sqlite3.connect(':memory:') as conn:
            # write the tables
            blood_pressure_per_patient.to_sql('blood_pressure_per_patient', conn, index=False)
            medicine_df_per_patient.to_sql('medicine_df_per_patient', conn, index=False)
            conn.create_aggregate("stdev", 1, StdevFunc)

            if medicine_df_per_patient.shape[0]>0:
                qry = ''' select avg (value), stdev(value) from 
                            (select distinct a.* 
                          from
                        blood_pressure_per_patient a left outer join medicine_df_per_patient b where
                          time not between start_date and end_date )
                    '''
            else:
                qry = ''' select avg (value), stdev(value) from 
                            (select distinct a.* 
                          from
                        blood_pressure_per_patient  a)
                        '''
            join_medicine_blood = pd.read_sql_query(qry, conn)

        dict_blood_pressure_without_medicine = dict()
        dict_blood_pressure_without_medicine['date_sample'] = row.date_sample
        dict_blood_pressure_without_medicine['patient_id'] = row.patient_id
        dict_blood_pressure_without_medicine['avg_value'] = join_medicine_blood.iloc[0,0]
        dict_blood_pressure_without_medicine['std_value'] = join_medicine_blood.iloc[0, 1]

        features_dict_list.append(dict_blood_pressure_without_medicine)

    features_df = pd.DataFrame(features_dict_list)
    features_df.to_csv(output_dir + "basic_" + parameter_name.replace(' ', '_') + ".csv")

    return features_df

    # df = pd.DataFrame.from_dict(dict_blood_pressure_without_medicine, orient="index")
    # df = df.reset_index()
    # df.columns = ['patient_id','avg_value','std_value']
    # df.to_csv(output_dir + "basic_"+parameter_name.replace(' ','_')+".csv")

    pass

def summary_signal_windows_with_basic_value (windows_signal_file, basic_values_file, paramater_name):

    short_signal_df = Utils.read_data(windows_signal_file)
    basic_values_df = Utils.read_data(basic_values_file)

    parameters_columns =  [ col for col in list(short_signal_df.keys()) if ((paramater_name in col) & ('median' in col))]
    short_signal_param_df = short_signal_df[['patient_id','date_sample'] + parameters_columns]

    df_plus_basic = pd.merge(short_signal_param_df, basic_values_df, how='left', on=['patient_id','date_sample'])

    list_cols = list()
    for col in parameters_columns:
        df_plus_basic[col+'_minus_basic'] = df_plus_basic[col] -df_plus_basic['avg_value']
        list_cols.append(col+'_minus_basic')
    # # check function of apply ---****************
    res_hist = df_plus_basic[list_cols].apply (lambda x: get_hist_dist(x), axis = 1)
    res_hist_df = pd.DataFrame.from_items(zip(res_hist.index, res_hist.values)).T

    new_features_column = [paramater_name+'_num_wind_50',paramater_name+'_num_wind_40',paramater_name+'_num_wind_30',paramater_name+'_num_wind_20',paramater_name+'_num_wind_10']
    df_plus_basic[new_features_column]=res_hist_df

    df_plus_basic = df_plus_basic.rename(columns={'avg_value': 'avg_value_'+paramater_name.replace(' ','_'),'std_value': 'std_value_'+paramater_name.replace(' ','_')})

    df_features = df_plus_basic[['patient_id','date_sample']+new_features_column + ['avg_value_'+paramater_name.replace(' ','_'),'std_value_'+paramater_name.replace(' ','_')]]

    return df_features
def get_hist_dist (row):

    list_results = list()
    try:
        count, division = np.histogram(row.fillna (-999), bins = [-50,-40,-30,-20,-10,0])
        # signal_df_no_inf['change'].hist(bins=division)
        count = np.nan_to_num(count)

        for i,div in enumerate(count):
            list_results.append(count[i]/len(row.dropna()))
    except Exception as e:
        print('no summary for '+str(row))
        print (e)
        pass
        unknown_val = -999
        list_results = [unknown_val,unknown_val,unknown_val,unknown_val,unknown_val]
    return list_results

# as a preparation need to call create_short_signal_features_fever() for create windows calculation for each signal
def add_blood_pressure_features (control_file,windows_signal_file,output_dir_in_func = output_dir,
                                 signals_dir = 'signals//'):
    print ('start add_blood_pressure_features')

    control_df = pd.read_csv(control_file, encoding="ISO-8859-1")

    list_signals_param = list()
    df_plus_new_features = control_df[['patient_id', 'date_sample']]
    # read full file and not partial!!!
    signal_source_diastolic = Signal_source.Signal_source(signal_name='Arterial Pressure Diastolic', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//diastolic_p_12_17.xlsx',min_value = 10, max_value=220,related_medicine_file=data_dir + 'vasoactive_icu_12_17_for_modeling.xlsx')
    list_signals_param.append(signal_source_diastolic)
    signal_source = Signal_source.Signal_source(signal_name='Mean Arterial Pressure', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//mean_p_12_17.xlsx',min_value = 10, max_value=220,related_medicine_file=data_dir + 'vasoactive_icu_12_17_for_modeling.xlsx')
    list_signals_param.append(signal_source)
    signal_source = Signal_source.Signal_source(signal_name='Arterial Pressure Systolic', signal_file='C://Users//michael//Desktop//predicting BSI//files//Data//systolic_p_12_17.xlsx',min_value = 10, max_value=220,related_medicine_file=data_dir + 'vasoactive_icu_12_17_for_modeling.xlsx')
    list_signals_param.append(signal_source)

    for signal_source in list_signals_param:
        print (' start signal - '+signal_source.signal_name)
        get_basic_blood_pressure(control_df = control_df,
                                 blood_pressure_file=signal_source.signal_file,
                                 medicine_file=signal_source.related_medicine_file,
                                 parameter_name=signal_source.signal_name, min_value=signal_source.min_value,
                                 max_value=signal_source.max_value, num_days=30,
                                 signals_dir=signals_dir)

        signals_features_df = summary_signal_windows_with_basic_value(
            windows_signal_file=windows_signal_file,
            basic_values_file = output_dir_in_func +"basic_" + signal_source.signal_name.replace(' ', '_') + ".csv",
            paramater_name = signal_source.signal_name)
#join all sources together
        df_plus_new_features = pd.merge(df_plus_new_features, signals_features_df, how='left', on=['patient_id','date_sample'])

    #23/12 - deleted - '_' + datetime.now().strftime('%Y%m%d%H%M%S')
    df_plus_new_features.to_csv(output_dir_in_func+signals_dir+'signal_blood_pressurex.csv')


def q1(x):
    return x.quantile(0.25)

def q2(x):
    return x.quantile(0.75)

def subtract_days(row , num_days):
    # print (row[['date_sample']])
    return row['date_sample'] - pd.DateOffset(days=num_days)
# def subtract_days(row , num_days=3):
#     # print (row[['date_sample']])
#     return (1)

# get features :  freq + magnitude1/2 (pct of the energy in the first and second place)
def fft_features(group,val_col_name='value',time_col_name='time'):
    # print (group)
    signal_with_no_nan =   group[val_col_name].dropna()
    if len(signal_with_no_nan)==0:
        return pd.Series([0,0,0])
    magnitude = np.fft.fft(signal_with_no_nan / max(signal_with_no_nan))  # normalize the smooth signal
    freq = np.fft.fftfreq(len(group[time_col_name]) - 4,
                          np.diff(group[time_col_name].dt.to_pydatetime())[0].total_seconds() / (60 * 60))


    # return series
    magnitude1 = 0
    magnitude2 = 0
    freq1 = 0
    try:
        magnitude1 = 100*np.absolute(magnitude)[1]/np.absolute(magnitude)[0]
        magnitude2 = 100 * np.absolute(magnitude)[2] / np.absolute(magnitude)[0]
        freq1 = freq[1]
    except Exception as e:
        pass
    return pd.Series([magnitude1,magnitude2,freq1])


# use_fft_feature only if smooth_signal = True - calculate 3 features based on fft
# inputs: complete_0_values - flag, if True will add 0 value between the values, relevant when no value means 0 value (for example, medicine)
# can complete hours or days :complete_hours= False, complete_days = False
#min_value/max_value - filter out values not in range
# smooth_signal - average window
# fft - get 3 fft features
def create_lab_data (control_file, lab_file = 'C://Users//michael//Desktop//predicting BSI//files//Data//lab_lactat_for_modeling.xlsx',
                       lab_parameter_name ='Lactate ABG', num_days = 3,
                       complete_0_values = False, complete_hours= False, complete_days = False,
                       min_value = -99999, max_value = 99999, calc_std_flag = False,
                       smooth_signal = False, use_fft_feature = False,
                       output_dir_in_func = output_dir, signals_dir = 'signals//',
                       new_parameter_suffix = '',lab_df= pd.DataFrame(columns=['A','B'])):
    print ('------start create_lab_data -' + lab_parameter_name + '(' + str(num_days) +' ['+ str(min_value) +' ,'+ str(max_value) +')---------')
    control_df = pd.read_csv(control_file, encoding="ISO-8859-1")

    if lab_df.shape[0]==0:
        org_lab_df = Utils.read_data(lab_file)
    else :
        org_lab_df = lab_df

    org_lab_df = org_lab_df[org_lab_df.parameter_name == lab_parameter_name]

    org_lab_df.value = pd.to_numeric(org_lab_df.value)
    org_lab_df = org_lab_df.loc[(org_lab_df.value >= min_value) & (org_lab_df.value <= max_value), :]

    df_join  = pd.merge(org_lab_df[['patient_id','time','value','parameter_name']],control_df[['patient_id','date_sample','admission start day icu']],how = 'inner',on = ['patient_id'])
    df_join.date_sample = pd.to_datetime(df_join.date_sample)
    df_join.time = pd.to_datetime(df_join.time)
    df_join['admission start day icu'] = pd.to_datetime(df_join['admission start day icu'])


    filter_df_join = df_join.loc[(df_join.time <= df_join.date_sample) & (
        df_join.date_sample.subtract(df_join.time) <= pd.to_timedelta(num_days, unit='d')), :]

    num_patients_with_param = len(filter_df_join.patient_id.drop_duplicates())
    if num_patients_with_param<100:
        print ('few patients('+str(num_patients_with_param)+') for '+lab_parameter_name)
        return

    # if want to avoid sharp peaks
    if smooth_signal :
        filter_df_join = filter_df_join.sort_values(by=['patient_id', 'time'], ascending=[True, False])
        filter_df_join_group = filter_df_join.groupby(['patient_id','date_sample']).value.rolling(center=True,window=5).mean()

        filter_df_join_group = filter_df_join_group.reset_index()
        filter_df_join_group.index = filter_df_join_group.level_2
        filter_df_join_group['time'] = filter_df_join['time']
        filter_df_join = filter_df_join_group
        filter_df_join = Handle_promateus_data.try_to_remove_columns(filter_df_join,['level_2'])
        filter_df_join = filter_df_join.reset_index()
        if use_fft_feature:
            fft_features_df= \
                filter_df_join_group.groupby (['patient_id','date_sample']).apply( fft_features)

            fft_features_df.columns = [lab_parameter_name +new_parameter_suffix+ '_' + str(num_days) + '_magnitude_1',
                                       lab_parameter_name+new_parameter_suffix + '_' + str(num_days) + '_magnitude_2',
                                       lab_parameter_name +new_parameter_suffix+ '_' + str(num_days) + '_freq']
            fft_features_df = fft_features_df.reset_index()
            # fft_features_df.to_csv('fft_features_df.csv')

    if complete_0_values:
        # df_complete = filter_df_join.groupby(['patient_id','date_sample']).agg(
        #     {'time': {'min_time': 'min', 'max_time': 'max'}})
        # # {'time': {'min_time': 'min', 'max_time': 'max'}, 'date_sample': {'date_sample': 'min'}})
        # df_complete.columns = df_complete.columns.dro plevel(0)
        # df_complete = df_complete.reset_index()

        df_complete = filter_df_join[['date_sample', 'admission start day icu', 'patient_id']].drop_duplicates()
        df_complete['date_sample_minus_days_back'] = df_complete.apply(subtract_days, args=(num_days,), axis=1)
        df_complete['min_time'] = df_complete[['admission start day icu','date_sample_minus_days_back']].max(
            axis=1)  # if wants to complete for each hour 0 value - relevant for vasoactive medicinw


        list_rows = list()
        for index, row in df_complete.iterrows():
            patient_id = row.patient_id

            if complete_hours :
                time_range = pd.date_range(Utils.round_to_hour(row.min_time), Utils.round_to_hour(row.date_sample), freq='H')
            elif complete_days:
                time_range = pd.date_range(Utils.truncate_to_day(row.min_time), Utils.truncate_to_day(row.date_sample),
                                           freq='D')
            else:
                raise Exception ('no resolution for complete 0 values')


            for time1 in time_range:
                list_rows.append([patient_id, time1, lab_parameter_name+new_parameter_suffix, 0, row.date_sample])

            if (index % 100 == 0):
                print(index)

        data = pd.DataFrame(list_rows)
        data.columns = ['patient_id', 'time', 'parameter_name', 'value','date_sample']

        data = data.loc[(data.time <= data.date_sample) & (
                data.date_sample.subtract(data.time) <= pd.to_timedelta(num_days, unit='d')), :]


        join_with_0 = pd.concat([data[['patient_id', 'time', 'parameter_name', 'value','date_sample']],
                  filter_df_join[['patient_id', 'time', 'parameter_name', 'value','date_sample']]])

        # drop duplications - take the value from filter_df_join
        join_with_0.drop_duplicates(subset=['patient_id', 'time', 'parameter_name',
                                                          'date_sample'], inplace=True, keep='last')

        join_with_0.to_csv(output_dir_in_func + lab_parameter_name +new_parameter_suffix+ '_with_0_days_' + str(num_days) + '.csv')
        print ('created ' + output_dir_in_func + lab_parameter_name+new_parameter_suffix + '_with_0_days_' + str(num_days) + '.csv')
        filter_df_join = join_with_0

    # CREATES DIFF FILE AFTER 0 IMPUTATION
    lab_diff_df = filter_df_join.copy(deep=True)
    lab_diff_df['prev_value'] = lab_diff_df.groupby('patient_id')['value'].shift(1)
    lab_diff_df = lab_diff_df.dropna(subset=['prev_value'])
    lab_diff_df['value'] = lab_diff_df['value'] - lab_diff_df['prev_value']
    lab_diff_df['parameter_name'] = 'diff_' + lab_parameter_name+new_parameter_suffix
    lab_diff_df = lab_diff_df[pd.notnull(lab_diff_df['value'])]

    param_name = (lab_parameter_name + new_parameter_suffix).replace(' ', '_').replace('/', '_')
    lab_diff_df.to_csv(output_dir_in_func +  'diff_' + param_name + '_with_0_days_' + str(num_days) + '.csv')
    print('created ' + output_dir_in_func +  'diff_' + param_name + '_with_0_days_' + str(num_days) + '.csv')

    for (lab_parameter_name, df_for_calc) in [(lab_parameter_name+new_parameter_suffix, filter_df_join), ('diff_' + lab_parameter_name+new_parameter_suffix, lab_diff_df)]:

        df_for_calc['sample_date_to_sig_date'] = round(df_for_calc.date_sample.subtract(df_for_calc.time).dt.total_seconds() / (24 * 60 * 60), 5)

        df_for_calc = df_for_calc.sort_values(by=['patient_id', 'time'], ascending=[True, False])

        param_name = (lab_parameter_name+new_parameter_suffix).replace(' ', '_').replace('/','_')
        df_for_calc.value = pd.to_numeric(df_for_calc.value)

        if calc_std_flag:
            df_for_calc_day_back_1 = df_for_calc.loc[(df_for_calc.time <= df_for_calc.date_sample) & (
                    df_for_calc.date_sample.subtract(df_for_calc.time) < pd.to_timedelta(1, unit='d')), :]

            df_for_calc_day_back_num_days = df_for_calc.loc[(df_for_calc.date_sample.subtract(df_for_calc.time) <pd.to_timedelta(num_days, unit='d')) & (
                    df_for_calc.date_sample.subtract(df_for_calc.time) > pd.to_timedelta(num_days-1, unit='d')), :]

            grp_func = {'value':{'std_day_1_' + lab_parameter_name: 'std', 'mean_day_1_' + lab_parameter_name: 'mean'}}
            df_for_calc_day_back_1_grp = df_for_calc_day_back_1.groupby(['patient_id','date_sample']).agg(grp_func)
            df_for_calc_day_back_1_grp.columns = df_for_calc_day_back_1_grp.columns.droplevel(0)
            df_for_calc_day_back_1_grp= df_for_calc_day_back_1_grp.reset_index()

            grp_func = {'value':{'std_day_' + str(num_days) +'_' + lab_parameter_name: 'std', 'mean_day_' + str(num_days) + '_' + lab_parameter_name: 'mean'}}
            df_for_calc_day_back_num_days_grp = df_for_calc_day_back_num_days.groupby(['patient_id','date_sample']).agg(grp_func)
            df_for_calc_day_back_num_days_grp.columns = df_for_calc_day_back_num_days_grp.columns.droplevel(0)
            df_for_calc_day_back_num_days_grp= df_for_calc_day_back_num_days_grp.reset_index()

            df_for_calc_day_back = pd.merge(df_for_calc_day_back_1_grp[['patient_id', 'date_sample','std_day_1' +'_' + lab_parameter_name, 'mean_day_1' + '_' + lab_parameter_name]],
                                            df_for_calc_day_back_num_days_grp[['patient_id', 'date_sample','std_day_' + str(num_days) +'_' + lab_parameter_name, 'mean_day_' + str(num_days) + '_' + lab_parameter_name]],
                                            how='left', on=['patient_id', 'date_sample'])

            df_for_calc_day_back['std_ratio_last_' + str(num_days) +'_' + lab_parameter_name] = df_for_calc_day_back['std_day_1' + '_' + lab_parameter_name] / df_for_calc_day_back['std_day_' + str(num_days) + '_' + lab_parameter_name]
            df_for_calc_day_back['mean_ratio_last_' + str(num_days) +'_' + lab_parameter_name] = df_for_calc_day_back['mean_day_1' + '_' + lab_parameter_name] / \
                                                                                                 df_for_calc_day_back['mean_day_' + str(num_days) +'_' + lab_parameter_name]

        group_functions = {'value' : {'std_'+param_name+'_'+str(num_days):'std','min_'+param_name+'_'+str(num_days): 'min', 'max_'+param_name+'_'+str(num_days) :'max',
                                      'median_'+param_name+'_'+str(num_days):'median','percentile_25_'+param_name+'_'+str(num_days):q1,
                                      'percentile_75_'+param_name+'_'+str(num_days):q2,'mean_'+param_name+'_'+str(num_days):'mean'}}
        filter_df_join_grp = df_for_calc.groupby(['patient_id','date_sample']).agg(group_functions)
        filter_df_join_grp.columns = filter_df_join_grp.columns.droplevel(0)


        # calculate features with last val
        # 25/02/2019 - add date_sample_to_max_ groupby
        last_val = df_for_calc.groupby(['patient_id','date_sample']).first().value
        filter_df_join_grp['last_val_'+param_name+'_'+str(num_days)]  = last_val.values
        filter_df_join_grp[param_name+'_last_minus_median'+'_'+str(num_days)] = filter_df_join_grp['last_val_'+param_name+'_'+str(num_days)] - filter_df_join_grp['median_'+param_name+'_'+str(num_days)]
        filter_df_join_grp[param_name + '_last_minus_median_norm'+'_'+str(num_days)] = filter_df_join_grp[param_name+'_last_minus_median'+'_'+str(num_days)]/filter_df_join_grp['median_'+param_name+'_'+str(num_days)]
        filter_df_join_grp[param_name+'_last_minus_min'+'_'+str(num_days)] = filter_df_join_grp['last_val_'+param_name+'_'+str(num_days)] - filter_df_join_grp['min_'+param_name+'_'+str(num_days)]
        filter_df_join_grp[param_name+'_last_minus_max'+'_'+str(num_days)] = filter_df_join_grp['last_val_'+param_name+'_'+str(num_days)] - filter_df_join_grp['max_'+param_name+'_'+str(num_days)]

        filter_df_join_grp[param_name + '_max_minus_min' + '_' + str(num_days)] = filter_df_join_grp[ 'max_' + param_name + '_' + str(
                                                                                           num_days)] - filter_df_join_grp[
                                                                                       'min_' + param_name + '_' + str(
                                                                                           num_days)]
        filter_df_join_grp['fibonacci_0.36_' + param_name + '_' + str(num_days)] = filter_df_join_grp['last_val_'+param_name+'_'+str(num_days)] - filter_df_join_grp['max_' + param_name + '_' + str(num_days)] -  filter_df_join_grp[param_name + '_max_minus_min' + '_' + str(num_days)]*0.36
        filter_df_join_grp['fibonacci_0.68_' + param_name + '_' + str(num_days)] = filter_df_join_grp['last_val_'+param_name+'_'+str(num_days)] - filter_df_join_grp['max_' + param_name + '_' + str(num_days)] -  filter_df_join_grp[param_name + '_max_minus_min' + '_' + str(num_days)]*0.68


        filter_df_join_grp = filter_df_join_grp.reset_index()

        #17/01/19 - add it to ignore parameter which apear once per patient
        if filter_df_join_grp.shape[0]==0:
            continue

        df = pd.merge(df_for_calc, filter_df_join_grp, how='left', on=['patient_id','date_sample'])
        df_max = df[df['value'] == df['max_'+param_name+'_'+str(num_days)]]
        # df_max_grp = df_max.groupby(['patient_id']).agg({'time':{'min_time':'min','max_time':'max'},'date_sample':{'date_sample':'min'}})
        df_max_grp = df_max.groupby(['patient_id','date_sample']).agg(
            {'time': {'min_time': 'min', 'max_time': 'max'}})

        df_max_grp.columns = df_max_grp.columns.droplevel(0)
        df_max_grp = df_max_grp.reset_index()
        df_max_grp ['sample_to_max_'+param_name+'_first'+'_'+str(num_days)] = df_max_grp['date_sample'].subtract (df_max_grp['min_time'])
        df_max_grp['sample_to_max_' + param_name + '_last'+'_'+str(num_days)] = df_max_grp['date_sample'].subtract(df_max_grp['max_time'])

        df_min = df[df['value'] == df['min_' + param_name+'_'+str(num_days)]]
        # df_min_grp = df_min.groupby(['patient_id']).agg(
        #     {'time': {'min_time': 'min', 'max_time': 'max'}, 'date_sample': {'date_sample': 'min'}})
        df_min_grp = df_min.groupby(['patient_id','date_sample']).agg(
            {'time': {'min_time': 'min', 'max_time': 'max'}})

        df_min_grp.columns = df_min_grp.columns.droplevel(0)
        df_min_grp = df_min_grp.reset_index()
        df_min_grp['sample_to_min_' + param_name + '_first'+'_'+str(num_days)] = df_min_grp['date_sample'].subtract(df_min_grp['min_time'])
        df_min_grp['sample_to_min_' + param_name + '_last'+'_'+str(num_days)] = df_min_grp['date_sample'].subtract(df_min_grp['max_time'])

        # 28/10 drop na
        df_for_calc = df_for_calc[np.isfinite(df_for_calc['value'])]

        for patient in df_for_calc.patient_id.unique():

            filter_df_join_patient = df_for_calc[df_for_calc.patient_id == patient]
            z = np.polyfit(filter_df_join_patient.sample_date_to_sig_date.values.flatten(),
                           filter_df_join_patient.value.values.flatten(), 1)

            filter_df_join_grp.loc[filter_df_join_grp.patient_id==patient, 'a_' + param_name+'_'+str(num_days)] = z[0]
            filter_df_join_grp.loc[filter_df_join_grp.patient_id == patient, 'b_' + param_name+'_'+str(num_days)] = z[1]


        # calculate
        out_df = pd.merge(filter_df_join_grp, df_max_grp[['patient_id','date_sample','sample_to_max_'+param_name+'_first'+'_'+str(num_days),'sample_to_max_'+param_name+'_last'+'_'+str(num_days)]],
                          how='left', on=['patient_id','date_sample'])
        out_df = pd.merge(out_df, df_min_grp[['patient_id','date_sample', 'sample_to_min_'+param_name+'_first'+'_'+str(num_days),'sample_to_min_'+param_name+'_last'+'_'+str(num_days)]], how='left',
                          on=['patient_id','date_sample'])

        if calc_std_flag:
            out_df = pd.merge(out_df, df_for_calc_day_back, how='left',
                              on=['patient_id', 'date_sample'])

        if smooth_signal and use_fft_feature and 'diff' not in param_name:
            out_df = pd.merge(out_df, fft_features_df, how='left',
                              on=['patient_id', 'date_sample'])


        out_df.fillna(0, inplace=True)
        # 23/12 delet '_' + datetime.now().strftime('%Y%m%d%H%M%S') - to simplfy rerun
        out_df.to_csv(output_dir_in_func+signals_dir+'lab_'+param_name+'_'+str(num_days)+ '.csv')
        print ('created '+ output_dir_in_func + signals_dir+'lab_' + param_name + '_' + str(num_days) + '.csv')
    # return out_df

#join data of pio2 anf fio2
# fio2 was summarized per hour - the value in round hour
# here calculate fio2 in the same hours as pio2 so later I can calculate the fraction
# changes 14/1 add - getting data also from first day
def create_fio2_pio2_ratio (control_file,pio2_file, fio2_file,num_days = 5,new_param_name = 'FIO2',default_val=0,num_hours_to_search=2):

    print('start create_fio2_pio2_ratio - prepare '+new_param_name)
    control_df = pd.read_csv(control_file, encoding="ISO-8859-1")
    pio2_df = Utils.read_data( pio2_file)
    fio2_df = Utils.read_data(fio2_file)
    print ('finish reading files')

    # every patient has pio2
    df_join  = pd.merge(pio2_df,control_df[['patient_id','date_sample','admission start day icu']],how = 'inner',on = ['patient_id'])
    df_join.date_sample = pd.to_datetime(df_join.date_sample)
    df_join.time = pd.to_datetime(df_join.time)
    control_df.date_sample = pd.to_datetime(control_df.date_sample)

    # fio2_df.date_sample = pd.to_datetime(fio2_df.date_sample)

    # without first day
    # control_pio2_join = df_join.loc[(df_join.time <= df_join.date_sample) & (
    #     df_join.date_sample.subtract(df_join.time) < pd.to_timedelta(num_days, unit='d')), :]
    #takes data from first day for apache
    df_join['admission start day icu'] = pd.to_datetime(df_join['admission start day icu'])
    df_join['time_minus_start_icu'] = df_join.time.subtract(df_join['admission start day icu'])
    df_join['date_sample_minus_time'] = df_join.date_sample.subtract(df_join.time)
    control_pio2_join = df_join[(((df_join.date_sample_minus_time <= pd.Timedelta(num_days, unit='d'))&
                       (df_join.date_sample_minus_time >= pd.Timedelta(0, unit='d'))))
                    | ((df_join['time_minus_start_icu'] <= pd.Timedelta(1.5, unit='d')) &
                        (df_join['time_minus_start_icu'] >= pd.Timedelta(-0.5, unit='d')))]



    control_pio2_join = control_pio2_join.rename(columns={'time': 'time_pio2'})
    control_pio2_join = control_pio2_join.rename(columns={'value': 'value_pio2'})

    df_join  = pd.merge(control_df[['patient_id','date_sample','admission start day icu']],fio2_df,how = 'inner',on = ['patient_id'])
    df_join.date_sample = pd.to_datetime(df_join.date_sample)
    df_join.time = pd.to_datetime(df_join.time)

    # without first day
    # control_fio2_join = df_join.loc[(df_join.time <= df_join.date_sample) & (
    #     df_join.date_sample.subtract(df_join.time) < pd.to_timedelta(num_days, unit='d')), :]

    #takes data from first day for apache
    control_fio2_join = df_join
    control_fio2_join['admission start day icu'] = pd.to_datetime(control_fio2_join['admission start day icu'])
    control_fio2_join['time_minus_start_icu'] = control_fio2_join.time.subtract(
        control_fio2_join['admission start day icu'])

    control_fio2_join['date_sample_minus_time'] = control_fio2_join.date_sample.subtract(control_fio2_join.time)
    control_fio2_join = control_fio2_join[(((control_fio2_join.date_sample_minus_time <= pd.Timedelta(num_days, unit='d'))&
                       (control_fio2_join.date_sample_minus_time >= pd.Timedelta(0, unit='d'))))
                    | ((control_fio2_join['time_minus_start_icu'] <= pd.Timedelta(1.5, unit='d')) &
                        (control_fio2_join['time_minus_start_icu'] >= pd.Timedelta(-0.5, unit='d')))]

    control_fio2_join = control_fio2_join.rename(columns={'time': 'time_fio2'})
    control_fio2_join = control_fio2_join.rename(columns={'value': 'value_fio2'})
    control_fio2_join = control_fio2_join.rename(columns={'parameter_name': 'parameter_name_fio2'})

    #add dummy with default so if there is no value in the last 8 hours will take the defult
    fio2_fio2_df_dummy = control_pio2_join[['patient_id','date_sample']]
    fio2_fio2_df_dummy['time_fio2'] = control_pio2_join['time_pio2'].subtract(pd.to_timedelta(num_hours_to_search,unit = 'h'))
    fio2_fio2_df_dummy['value_fio2'] =default_val
    fio2_fio2_df_dummy['parameter_name_fio2'] = new_param_name


    control_fio2_join = pd.concat([control_fio2_join[['patient_id','time_fio2','value_fio2','parameter_name_fio2','date_sample']],
                                   fio2_fio2_df_dummy[
                                       [ 'patient_id', 'time_fio2', 'value_fio2', 'parameter_name_fio2',
                                        'date_sample']]])

    #11/12 - change to left , so no fio2 will be replaced by default val
    fio2_pio2_df = pd.merge (control_pio2_join,control_fio2_join, on =['patient_id','date_sample'],how = 'left')
    fio2_pio2_df['value_fio2'] = fio2_pio2_df['value_fio2'].fillna(default_val)

    # fio2_pio2_df.to_csv('tmp.csv')
    #take value from last round hour
    fio2_pio2_df_filter = fio2_pio2_df[((fio2_pio2_df.time_pio2.subtract(fio2_pio2_df.time_fio2)<pd.to_timedelta(num_hours_to_search,unit = 'h'))&
                                          (fio2_pio2_df.time_pio2 > fio2_pio2_df.time_fio2))
                                       | (np.isnat(fio2_pio2_df.time_fio2))]

    fio2_pio2_df_filter.sort_values(inplace=True,by=['patient_id', 'time_pio2','time_fio2'], ascending=[True, True, True])
    fio2_pio2_df_filter = fio2_pio2_df_filter.drop_duplicates(subset = ['patient_id','date_sample','time_pio2','parameter_name_fio2'],keep = 'last')
    fio2_pio2_df_filter['value_fio2'] = fio2_pio2_df_filter['value_fio2'].fillna(default_val)

    relevant_fio2 = fio2_pio2_df_filter[['patient_id','date_sample','time_pio2','parameter_name_fio2','value_fio2']]
    relevant_fio2.columns = ['patient_id','date_sample','time','parameter_name','value']

    relevant_fio2.to_csv(output_dir+new_param_name+'_calculated.csv')

# much more slowly function - reas the original commandas again
# def create_fio2_pio2_ratio (control_file,pio2_file, fio2_dir,num_days = 5,fio2_file_prefix = 'fio2',new_param_name = 'FIO2'):
#     control_df = pd.read_csv(control_file, encoding="ISO-8859-1")
#     pio2_df = Utils.read_data( pio2_file)
#
#     df_join  = pd.merge(pio2_df,control_df[['patient_id','date_sample']],how = 'inner',on = ['patient_id'])
#     df_join.date_sample = pd.to_datetime(df_join.date_sample)
#     df_join.time = pd.to_datetime(df_join.time)
#
#     filter_df_join = df_join.loc[(df_join.time <= df_join.date_sample) & (
#         df_join.date_sample.subtract(df_join.time) < pd.to_timedelta(num_days, unit='d')), :]
#
#     filter_df_join = filter_df_join.sort_values(by=['patient_id', 'time','value'], ascending=[True, False,False])
#
#     list_rows = list()
#
#     d1 = date(2013, 1, 1)  # start date
#     d2 = date(2017, 12, 1)  # end date -- need to be 18
#
#     delta = d2 - d1  # timedelta
#
#     for i in range((d2.year - d1.year) * 12 + 1):
#
#         min_date_to_search = d1 + relativedelta(months=+i)
#         max_date_to_search = d1 + relativedelta(months=i+1)
#
#         year_file = str(min_date_to_search.year)[2:4]
#         month_file = str(min_date_to_search)[5:7]
#
#         fio2_df = Utils.read_data(fio2_dir + '//' + year_file + '//'+fio2_file_prefix+'_' + month_file + year_file + '.xlsx',skip_rows=6)
#         print (fio2_dir + '//' + year_file + '//'+fio2_file_prefix+'_' + month_file + year_file + '.xlsx')
#         filter_df_join_month = filter_df_join[(filter_df_join.time> min_date_to_search) &(filter_df_join.time< max_date_to_search)]
#
#         print (filter_df_join_month.shape)
#         for index, row in filter_df_join_month.iterrows():
#             # print (row)
#             if (index%100==0):
#                 print (index)
#             fio2_df_row = fio2_df[(fio2_df[Utils.signal_id_column_name_heb] == row.patient_id) & (fio2_df[Utils.signal_time_column_name_heb]>(row.time - pd.to_timedelta(1, unit='h')))
#                             & (fio2_df[Utils.signal_time_column_name_heb]<row.time)]
#
#             avg_fio2 = np.mean(fio2_df_row[Utils.signal_value_column_name_heb])
#             new_row = [row.patient_id, row.time,avg_fio2,new_param_name]
#             list_rows.append(new_row)
#         output_df = pd.DataFrame(list_rows)
#         output_df.columns = ['patient_id', 'time', 'value', 'parameter_name']
#         output_df.to_csv(output_dir + new_param_name + '_' + str(i) + '_calculated.csv')
#         list_rows = list()
#
#     # output_df = pd.DataFrame(list_rows)
#     # output_df.columns = ['patient_id','time','value','parameter_name']
#     # output_df.to_csv(output_dir+new_param_name+'_calculated.csv')

# support different function between 2 files
# notice: the time must be exactly thhe same!
def create_signal_combination (input_file1, feature1, feature2, func, input_file2 = '',num_days = '',
                               min_val_feature1 = -9999, max_val_feature1 = 9999,
                               min_val_feature2=-9999, max_val_feature2=9999):

    input_df1 = Utils.read_data(input_file1)

    if input_file2 == '':
        input_df2 = input_df1
    else:
        input_df2 = Utils.read_data(input_file2)

    feature1_df = input_df1[input_df1.parameter_name==feature1]
    feature2_df = input_df2[input_df2.parameter_name == feature2]

    feature1_df = feature1_df[(feature1_df.value >= min_val_feature1) & (feature1_df.value <= max_val_feature1)]
    feature2_df = feature2_df[(feature2_df.value >= min_val_feature2) & (feature2_df.value <= max_val_feature2)]


    feature1_df.time = pd.to_datetime(feature1_df.time)
    feature2_df.time = pd.to_datetime(feature2_df.time)
    df_join = pd.merge(feature1_df, feature2_df, how='inner', on=['patient_id','time'])


    parameter_name = feature1.replace('-','_').replace(' ','_') + '_' + func + '_' + feature2.replace('-','_').replace(' ','_')
    if func =='multiply':
        df_join['parameter_name'] = parameter_name
        df_join ['value'] = df_join['value_x']*df_join['value_y']
    elif func == 'divide':
        df_join['parameter_name'] = parameter_name
        df_join['value'] = df_join['value_x'] / df_join['value_y']
    elif func == 'minus':
        df_join['parameter_name'] = parameter_name
        df_join['value'] = df_join['value_x'] - df_join['value_y']
    elif func == 'plus':
        df_join['parameter_name'] = parameter_name
        df_join['value'] = df_join['value_x'] + df_join['value_y']

    df_join_columns = list(df_join.keys())
    output_file = output_dir + parameter_name + str(num_days) + '.csv'
    print (output_file)
    if 'date_sample' in df_join_columns:
        df_join[['value', 'time', 'parameter_name', 'patient_id','date_sample']].to_csv(output_file)
    else :
        df_join[['value', 'time', 'parameter_name', 'patient_id']].to_csv(output_file)
    return

#parse vector to data - Micahel got data in different format - need to convert it the usual mode
def convert_new_control_to_new_source (file_to_convert = 'C://Users//michael//Desktop//predicting BSI//files//Data//large_bcu.xlsx'):

    df = pd.read_csv(file_to_convert, encoding="ISO-8859-1")

    columns_to_parse = ['immature granulocytes','bands',
                        'immature granulocytes - abs','mean platelets volume',
                        'platelt destribution width (pdw)','plateletcrit (PCt)',
                        'CRP','RBC destribution width (RDW)']


    for column in columns_to_parse:
        list_rows_for_output = list()
        print (' parse '+ column)
        for index, row in df[[column, 'Patient ID']].drop_duplicates().iterrows():
            try:
                list_dates_values = row[column].split('#')
                for date_value in list_dates_values:
                    date, value = date_value.split('|')

                    list_rows_for_output.append([row['Patient ID'],date,value,column])
            except Exception:
                # print(row[column])
                pass

        data = pd.DataFrame(list_rows_for_output)
        data.columns = ['patient_id','time','value','parameter_name']

        data = data[~data.value.isin(['Cancelled','----','בעבודה'])]

        data.to_csv(output_dir+column+'_for_modeling.csv')


def prepare_blood_old(control_file, num_days_list):
    print('start prepare_blood')

    df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                                          'WBC', 'Neutrophils', 'multiply')

    df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                                          'WBC', 'Lymphocytes', 'multiply')

    df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                                          'WBC', 'Basophils', 'multiply')

    df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                                          'WBC', 'Eosinphils', 'multiply')

    df_to_add = create_signal_combination(
        'C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
        'BUN', 'Creat', 'divide')

    for num_days in num_days_list:
        pass
        # # ---------create lab features -------------------------
        #

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_lactat_for_modeling.xlsx',
                        lab_parameter_name='Lactate ABG', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_glucose_for_modeling.xlsx',
                        lab_parameter_name='Glucose-ABG', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_ca_for_modeling.xlsx',
                        lab_parameter_name='Ca', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_potassium_for_modeling.xlsx',
                        lab_parameter_name='Potassium-ABG', num_days=3, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='Platelets', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='Neutrophils', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='Eosinphils', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='WBC', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='Basophils', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='Hemoglobin CBC', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='Lymphocytes', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
                        lab_parameter_name='Reticulocytes', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Albumin', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Bilirubin (Total)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Creat', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Uric Acid', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Urinary Creatinine', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='SGOT (AST)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='SGPT (ALT)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Phosphorus', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Magnesium', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Gamma GT', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='CCT', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Calcium (Total)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='BUN', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
                        lab_parameter_name='Alkaline Phosphatase', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//mcv_13_18.xlsx',
                        lab_parameter_name='Mean RBC Volume', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//mch_13_18.xlsx',
                        lab_parameter_name='MCH', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//mchc_hct_13_18.xlsx',
                        lab_parameter_name='Haematocrit', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//mchc_hct_13_18.xlsx',
                        lab_parameter_name='MCHC', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//direct_bilirubin_12_18.xlsx',
                        lab_parameter_name='Bilirubin (Direct)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//WBC_multiply_Neutrophils.csv',
                        lab_parameter_name='WBC_multiply_Neutrophils', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//WBC_multiply_Lymphocytes.csv',
                        lab_parameter_name='WBC_multiply_Lymphocytes', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//WBC_multiply_Basophils.csv',
                        lab_parameter_name='WBC_multiply_Basophils', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//WBC_multiply_Eosinphils.csv',
                        lab_parameter_name='WBC_multiply_Eosinphils', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//BUN_divide_Creat.csv',
                        lab_parameter_name='BUN_divide_Creat', num_days=num_days, output_dir_in_func=output_dir)

        # -------------------------------------------
        # delete - '-Results Vector'
        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//immature granulocytes_for_modeling.csv',
                        lab_parameter_name='immature granulocytes', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//bands_for_modeling.csv',
                        lab_parameter_name='bands', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//immature granulocytes - abs_for_modeling.csv',
                        lab_parameter_name='immature granulocytes - abs', num_days=num_days,
                        output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//mean platelets volume_for_modeling.csv',
                        lab_parameter_name='mean platelets volume', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//platelt destribution width (pdw)_for_modeling.csv',
                        lab_parameter_name='platelt destribution width (pdw)', num_days=num_days,
                        output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//plateletcrit (PCt)_for_modeling.csv',
                        lab_parameter_name='plateletcrit (PCt)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//CRP_for_modeling.csv',
                        lab_parameter_name='CRP', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//RBC destribution width (RDW)_for_modeling.csv',
                        lab_parameter_name='RBC destribution width (RDW)', num_days=num_days,
                        output_dir_in_func=output_dir)

        #
# all features related to blood
def prepare_blood (control_file, num_days_list):
    print('start prepare_blood')

    #todo - delet comment
    # df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
    #                                       'WBC', 'Neutrophils', 'multiply')
    #
    # df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
    #                                       'WBC', 'Lymphocytes', 'multiply')
    #
    # df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
    #                                       'WBC', 'Basophils', 'multiply')
    #
    # df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//cbc_13_18.xlsx',
    #                                       'WBC', 'Eosinphils', 'multiply')
    #
    # df_to_add = create_signal_combination('C://Users//michael//Desktop//predicting BSI//files//Data//chemistry_13_18.xlsx',
    #                                       'BUN', 'Creat', 'divide')

    # -----once
    # file1 = data_dir+'labs//'+'lab_gas_13_16.xlsx'
    # file2 = data_dir+'labs//'+'lab_gas_16_17.xlsx'
    # df1 = Utils.read_data(file1)
    # df2 = Utils.read_data(file2)
    # df3 = pd.concat([df1,df2]).drop_duplicates()
    # df3.to_csv(data_dir+'lab_gas_13_17.csv')
    #---end once

    df_to_add = create_signal_combination(
        'C://Users//michael//Desktop//predicting BSI//files//Data//lab_gas_13_17.csv',
        'Sodium-ABG', 'Chloride-ABG', 'minus')
    df_to_add = create_signal_combination(
        'C://Users//michael//Desktop//predicting BSI//files//Data//lab_gas_13_17.csv',
        'Sodium_ABG_minus_Chloride_ABG', 'Bicarbonate (ABG) s', 'minus')
    # end anion gap creation (as in mimic)

    files_list = [
                  # tmp
                  data_dir+'lab_gas_13_17.csv',
                  data_dir+'cbc_13_18.xlsx',
                  data_dir+'chemistry_13_18.xlsx',
                  data_dir+'mcv_13_18.xlsx',
                  data_dir+'mchc_hct_13_18.xlsx',
                  data_dir+'direct_bilirubin_12_18.xlsx',
                  data_dir+'mch_13_18.xlsx',
                  output_dir+'Sodium_ABG_minus_Chloride_ABG_minus_Bicarbonate_(ABG)_s.csv',
                  output_dir+'WBC_multiply_Neutrophils.csv',
                  output_dir+'WBC_multiply_Lymphocytes.csv',
                  output_dir+'WBC_multiply_Basophils.csv',
                  output_dir+'WBC_multiply_Eosinphils.csv',
                  output_dir+'BUN_divide_Creat.csv',
                  data_dir+'comp_lab_12_18.xlsx'
                  ]

    for file_name in files_list :
        print('handle file :' + file_name)
        df = Utils.read_data(file_name)
        parameters_list = df.parameter_name.drop_duplicates()

        for param_name in parameters_list:
            print ('handle file :' + file_name+ ', parameter :'+param_name)
            for num_days in num_days_list:
                create_lab_data(control_file,
                               lab_df=df,
                               lab_parameter_name=param_name,
                               num_days=num_days, output_dir_in_func=output_dir)

    for num_days in num_days_list:
        pass


        # -------------------------------------------
        # delete - '-Results Vector'
        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//immature granulocytes_for_modeling.csv',
                        lab_parameter_name='immature granulocytes', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//bands_for_modeling.csv',
                        lab_parameter_name='bands', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//immature granulocytes - abs_for_modeling.csv',
                        lab_parameter_name='immature granulocytes - abs', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//mean platelets volume_for_modeling.csv',
                        lab_parameter_name='mean platelets volume', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//platelt destribution width (pdw)_for_modeling.csv',
                        lab_parameter_name='platelt destribution width (pdw)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//plateletcrit (PCt)_for_modeling.csv',
                        lab_parameter_name='plateletcrit (PCt)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//CRP_for_modeling.csv',
                        lab_parameter_name='CRP', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//RBC destribution width (RDW)_for_modeling.csv',
                        lab_parameter_name='RBC destribution width (RDW)', num_days=num_days, output_dir_in_func=output_dir)

        #
def create_A_aPO2 (paco2_file='C://Users//michael//Desktop//predicting BSI//files//Data//paco2_for_modeling.xlsx',
                   paco2_param_name = 'PaCO2',
                  fio2_file = output_dir + 'FIO2_calculated.csv',
                   fio2_param_name = 'FIO2',
                   pao2_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_pao2_for_modeling.xlsx',
                   pao2_param_name='PaO2'):

    print ('start create_A_aPO2')
    paco2_df1 = Utils.read_data(paco2_file)
    paco2_df1= paco2_df1[paco2_df1.parameter_name==paco2_param_name]
    paco2_df1 = paco2_df1.rename(columns = {'value':'value_paco2'})
    fio2_df1 = Utils.read_data(fio2_file)
    fio2_df1 = fio2_df1.rename(columns={'value': 'value_fio2'})
    fio2_df1=fio2_df1[fio2_df1.parameter_name == fio2_param_name]
    pao2_df1 = Utils.read_data(pao2_file)
    pao2_df1 = pao2_df1.rename(columns={'value': 'value_pao2'})
    pao2_df1 = pao2_df1[pao2_df1.parameter_name == pao2_param_name]

    paco2_df1.time = pd.to_datetime(paco2_df1.time)
    fio2_df1.time = pd.to_datetime(fio2_df1.time)
    pao2_df1.time = pd.to_datetime(pao2_df1.time)

    paco2_df1[paco2_df1.patient_id == 36].to_csv('paco2_df1.csv')
    pao2_df1[pao2_df1.patient_id == 36].to_csv('pao2_df1.csv')
    fio2_df1[fio2_df1.patient_id == 36].to_csv('fio2_df1.csv')

    df_join = pd.merge(paco2_df1, fio2_df1, how='inner', on=['patient_id','time'])
    df_join = pd.merge(df_join, pao2_df1, how='inner', on=['patient_id', 'time'])

    df_join.to_csv(output_dir+'df_join.csv')
    parameter_name = 'A_aPO2'

    df_join = df_join[~pd.to_numeric(df_join['value_fio2'], errors='coerce').isnull()]
    df_join = df_join[~pd.to_numeric(df_join['value_paco2'], errors='coerce').isnull()]
    df_join = df_join[~pd.to_numeric(df_join['value_pao2'], errors='coerce').isnull()]

    df_join['parameter_name'] = parameter_name
    df_join ['value'] = pd.to_numeric(df_join['value_fio2'])*7.13 \
                        - pd.to_numeric(df_join['value_paco2'])/0.8-\
                        pd.to_numeric(df_join['value_pao2'])

    # df_join.to_csv('A_apo2_calc.csv')
    output_file = output_dir + parameter_name +  '.csv'
    print (output_file)
    if 'date_sample' in list(df_join.keys()):
        df_join[['value', 'time', 'parameter_name', 'patient_id','date_sample']].to_csv(output_file)
    else :
        df_join[['value', 'time', 'parameter_name', 'patient_id']].to_csv(output_file)
    return

# all features related to breathing
def prepare_breathing (control_file, num_days_list):
    print ('start prepare_breathing')
    num_days = 100 # this is used also for apache - first day in icu so we need all the data
    create_fio2_pio2_ratio(control_file,
                           pio2_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_pao2_for_modeling.xlsx',
                           fio2_file=output_dir + 'FIO2_round_to_hour.csv', num_days=num_days,
                           new_param_name='FIO2',default_val=30)

    create_fio2_pio2_ratio(control_file,
                           pio2_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_pao2_for_modeling.xlsx',
                           fio2_file=output_dir + 'PEEP_round_to_hour.csv', num_days=num_days,
                           new_param_name='PEEP',default_val=0)

    df_to_add = create_signal_combination(data_dir + 'lab_pao2_for_modeling.xlsx',
                                          'PaO2', 'FIO2', 'divide', output_dir + 'FIO2_calculated.csv')

    df_to_add = create_signal_combination(data_dir + 'lab_pao2_for_modeling.xlsx',
                                          'PaO2', 'PEEP', 'divide', output_dir + 'PEEP_calculated.csv')

    create_A_aPO2(paco2_file='C://Users//michael//Desktop//predicting BSI//files//Data//paco2_for_modeling.xlsx',paco2_param_name = 'PaCO2',
                  fio2_file = output_dir + 'FIO2_calculated.csv', fio2_param_name = 'FIO2',
                  pao2_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_pao2_for_modeling.xlsx',pao2_param_name = 'PaO2')


    for num_days in num_days_list:
        create_lab_data(control_file,
                        lab_file=output_dir + 'PaO2_divide_FIO2.csv',
                        lab_parameter_name='PaO2_divide_FIO2', num_days=num_days,
                        output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=output_dir + 'PaO2_divide_PEEP.csv',
                        lab_parameter_name='PaO2_divide_PEEP', num_days=num_days,
                        output_dir_in_func=output_dir)

        #16/12/18 -delete 0 complete_0_values = True, use the calculated file instead of the hourly data
        create_lab_data(control_file,
                        lab_file=output_dir + 'PEEP_calculated.csv', #'PEEP_round_to_hour.csv',
                        lab_parameter_name='PEEP', num_days=num_days, output_dir_in_func=output_dir
                        )

        create_lab_data(control_file,
                        lab_file=output_dir + 'FIO2_calculated.csv',# 'FIO2_round_to_hour.csv',
                        lab_parameter_name='FIO2', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=output_dir +  'A_aPO2.csv',
                        lab_parameter_name='A_aPO2', num_days=num_days, output_dir_in_func=output_dir,
                        complete_0_values=True, complete_hours=True)


# all features related to liquid
def prepare_liquid (control_file, num_days_list):
    print ('start prepare_liquid')

    # # need to run create_input_liquid_data and rename column time in read C://Users//michael//Desktop//predicting BSI//files//Data//input_liquids_per_hour_for_modeling.csv
    input_liquid_calculated_file = create_liquids_file.create_input_liquid_data(control_file, num_days=40)
    # input_liquid_calculated_file = output_dir + 'input_liquids_per_hour_for_modeling.csv'

    updatd_input_liquid_calculated_file =  add_dummy_rows_for_control_set(control_file,
                                           input_liquid_calculated_file,'input_liquid',
                                            output_file=output_dir + 'input_liquids_per_hour_for_modeling_with_dummy.csv')
    #
    # # for total liquid calculate 30 day - used for calculated feature - total_liquid_30-700 * days_in_icu
    # prepare input, total, urine, output - sum files which be later used in create_lab_data
    create_liquids_file.create_liquid_balance(control_file,
                                              num_days_list+[30], min_value=-3000,
                                              max_value=3000,
                                              input_liquids_per_hour_for_modeling_file=updatd_input_liquid_calculated_file)

    for num_days in num_days_list:
        pass
        create_lab_data(control_file,
                        lab_file=output_dir + 'sum_daily_urine.csv',
                        lab_parameter_name='urine', num_days=num_days, complete_0_values=True,
                        complete_days=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=output_dir + 'sum_daily_output_liquid.csv',
                        lab_parameter_name='output_liquid', num_days=num_days, complete_0_values=True,
                        complete_days=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=output_dir + 'sum_daily_input_liquid.csv',
                        lab_parameter_name='input_liquid', num_days=num_days, complete_0_values=True,
                        complete_days=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=output_dir + 'sum_daily_total_liquid.csv',
                        lab_parameter_name='total_liquid', num_days=num_days, complete_0_values=True,
                        complete_days=True, output_dir_in_func=output_dir,
                        new_parameter_suffix='_daily')

        create_lab_data(control_file,
                        lab_file=output_dir + 'sum_hour_total_liquid.csv',
                        lab_parameter_name='total_liquid', num_days=num_days, complete_0_values=True,
                        complete_hours=True, output_dir_in_func=output_dir,
                        new_parameter_suffix='_hourly')

# all features related to heart
def prepare_heart (control_file, num_days_list):
    print ('start prepare_heart')
    for num_days in num_days_list:

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//heart_rate_ecg_12_17.xlsx',
                        lab_parameter_name='HR -EKG', num_days=num_days, min_value=35, max_value=210, calc_std_flag=True,
                        smooth_signal = True, use_fft_feature=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//temperature_12_17.xlsx',
                        lab_parameter_name='Temperature', num_days=num_days, min_value=34.5, max_value=45, calc_std_flag=True,
                        smooth_signal=True, use_fft_feature = True, output_dir_in_func=output_dir)


        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//systolic_p_12_17.xlsx',
                        lab_parameter_name='Arterial Pressure Systolic', num_days=num_days, min_value=30, max_value=350, calc_std_flag=True, smooth_signal=True, use_fft_feature=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//diastolic_p_12_17.xlsx',
                        lab_parameter_name='Arterial Pressure Diastolic', num_days=num_days, min_value=20, max_value=310, calc_std_flag=True, smooth_signal=True, use_fft_feature=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//mean_p_12_17.xlsx',
                        lab_parameter_name='Mean Arterial Pressure', num_days=num_days, min_value=25, max_value=330, calc_std_flag=True, smooth_signal=True, use_fft_feature=True, output_dir_in_func=output_dir)

    # #---------creating blood pressure features-----------------
    #---if need to create windows again

    # comment - because it is not selected in feature selection 26/02/19
    # res = CreateFeaturesFile.create_short_signal_features_windows(control_file=control_file,output_dir_param=output_dir)
    # windows_file = res['output_filename']
    #
    # # windows_file = out   put_dir+'short_signals_windows_20180914114934.csv'
    # add_blood_pressure_features(control_file = control_file,windows_signal_file=windows_file,
    #                             output_dir_in_func=output_dir)

# all features related to heart
def prepare_gaz (control_file, num_days_list):
    print ('start prepare_gaz')


    for num_days in num_days_list:
        pass

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//RR_total_for_modeling.xlsx',
                        lab_parameter_name='.RR total', num_days=num_days, min_value=4, max_value=60, calc_std_flag=True,
                        smooth_signal = True, use_fft_feature=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//endtidal_for_modeling.xlsx',
                        lab_parameter_name='General - End Tidal CO2', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//paco2_for_modeling.xlsx',
                        lab_parameter_name='PaCO2', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_ph_for_modeling.xlsx',
                        lab_parameter_name='PH (ABG)', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_Bicarbonate_for_modeling.xlsx',
                        lab_parameter_name='Bicarbonate (ABG) s', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_pao2_for_modeling.xlsx',
                        lab_parameter_name='PaO2', num_days=num_days, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_sodium_for_modeling.xlsx',
                        lab_parameter_name='Sodium-ABG', num_days=num_days, output_dir_in_func=output_dir)


        create_lab_data(control_file,
                        lab_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_base_exess_for_modeling.xlsx',
                        lab_parameter_name='Base.Exess', num_days=num_days, output_dir_in_func=output_dir)

# used for add rows for patients if no value - add 1 row in time of sample date
def add_dummy_rows_for_control_set(control_file, org_file,parameter_name,output_file):
    control_df =pd.read_csv(control_file, encoding="ISO-8859-1")
    org_df = pd.read_csv(org_file, encoding="ISO-8859-1")

    org_df.time = pd.to_datetime(org_df.time)
    # ------add dummy for date_sample
    dummy_df =  control_df[['patient_id','date_sample']]
    dummy_df.date_sample = pd.to_datetime(dummy_df.date_sample).dt.floor("H")
    dummy_df['value'] = 0
    dummy_df['parameter_name'] =parameter_name
    dummy_df.columns = ['patient_id','time','value','parameter_name']

    #add 0 for missing patients
    org_df_concat = pd.concat([dummy_df[['patient_id', 'time', 'parameter_name','value']],org_df[['patient_id', 'time', 'parameter_name','value']]])
    org_df_concat.drop_duplicates(subset=['patient_id', 'time', 'parameter_name'], inplace=True, keep='last')


    org_df_concat.to_csv (output_file)
    return output_file

def prepare_medicine (control_file, num_days_list):
    print ('start prepare_medicine')

    #---vasoactive-------------
    org_file = data_dir+'vasoactive_rate_per_hour_for_modeling.csv'
    new_file_name = output_dir+'vasoactive_rate_per_hour_for_modeling_with_dummy.csv'

    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             org_file,
                                                             'Noradrenaline (Norepinephrine)',
                                                             output_file=new_file_name)
    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Vassopressin (Pitressin)',
                                                             output_file=new_file_name)
    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Adrenaline (Epinephrine)',
                                                             output_file=new_file_name)
    for num_days in num_days_list:

        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Noradrenaline (Norepinephrine)', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)


        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Vassopressin (Pitressin)', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)


        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Adrenaline (Epinephrine)', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)


    #------------Morphine/Fentanyl/Propofol 2% ( Diprivan )' (only 1 patient with 1%), 'Dormicum'

    org_file = data_dir+'analgesia_medicine_rate_per_hour_for_modeling.csv'
    new_file_name = output_dir+'analgesia_medicine_rate_per_hour_for_modeling_with_dummy.csv'

    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             org_file,
                                                             'Fentanyl',
                                                             output_file=new_file_name)
    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             org_file,
                                                             'Remifentanil (Ultiva)',
                                                             output_file=new_file_name)
    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Morphine',
                                                             output_file=new_file_name)
    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Propofol 2% ( Diprivan )',
                                                             output_file=new_file_name)
    updated_vasoactive_file = add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Dormicum',
                                                             output_file=new_file_name)


    for num_days in num_days_list:
        # todo - delete  comment
        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Fentanyl', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Remifentanil (Ultiva)', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Morphine', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Propofol 2% ( Diprivan )', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)


        create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Dormicum', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)


# all things we need to do once at the begining
def data_prepration ():#once

    import minimize_min_signals
    import create_vasoactive_data

    # todo - delete comment
    # all to run as data preparation
    # convert commands to hourly rate
    # create_vasoactive_data.create_vasoactive_data(data_dir + 'vasoactive_icu_12_17_for_modeling.xlsx',
    #                        'vasoactive_rate_per_hour_for_modeling.csv')
    # create_vasoactive_data.create_vasoactive_data(data_dir + 'fusid_for_modeling_12_17.xlsx',
    #                        'fusid_rate_per_hour_for_modeling.csv')

    # convert data per min to value in every round hour(take the average value around each hoour)
    # minimize_min_signals.minimize_min_signal('C://Users//michael//Desktop//predicting BSI//files//Data//peep', 'peep', 'PEEP')
    # minimize_min_signals.minimize_min_signal('C://Users//michael//Desktop//predicting BSI//files//Data//fio2', 'fio2', 'FIO2')

    # medicine in push
    create_daily_data.create_diarrhea_medicine_data(output_ldiarrhea_calculated_file=output_dir + 'diarrhea_medicine_for_modeling.csv')

    # todo - delete comment
    # medicine with period
    create_daily_data.create_analgesia_medicine_data(
        output_analgesia_calculated_file=data_dir + 'analgesia_medicine_for_modeling.csv')
    #
    create_vasoactive_data.create_vasoactive_data(data_dir + 'analgesia_medicine_for_modeling.csv',
                           'analgesia_medicine_rate_per_hour_for_modeling.csv')


def prepare_fusid(control_file, num_days_list):

    print ('start prepare_fusid')
    fusid_file = data_dir + 'fusid_rate_per_hour_for_modeling.csv'
    updated_fusid_file = output_dir + 'fusid_rate_per_hour_for_modeling_with_dummy.csv'
    add_dummy_rows_for_control_set(control_file, fusid_file, 'Fusid', updated_fusid_file)
    #convert commands to daily summary which will be used later
    create_daily_data.create_sum_fusid_daily(control_file,
                                             fusid_file=updated_fusid_file,
                                             num_days=max(num_days_list) + 1)
    for num_days in num_days_list:


        create_lab_data(control_file,
                        lab_file=output_dir + 'fusid_daily_' + str(max(num_days_list)+1) + '.csv',
                        lab_parameter_name='fusid_daily', num_days=num_days,
                        complete_0_values=True,
                        complete_days=True, output_dir_in_func=output_dir)

        create_lab_data(control_file,
                        lab_file=updated_fusid_file,
                        lab_parameter_name='Fusid', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

def prepare_stools_admission (control_file,stool_file=data_dir + 'wounds_stools_13_18_for_modeling.xlsx'):
    stool_df_filtered = Utils.read_data(stool_file,nrows = 1000)
    stool_df_filtered1 = stool_df_filtered[(stool_df_filtered.parameter_name == 'יציאה') &
                                           (~stool_df_filtered.value.isin(['ללא', "ZASSI"])) | (
                                                   stool_df_filtered.parameter_name == 'stool') | (
                                                   stool_df_filtered.parameter_name == '-1')]

    control_df  = Utils.read_data(control_file)
    join_df = pd.merge (control_df[['admission start day icu','admission end day ICU','patient_id','deceaseddate']],
                        stool_df_filtered1, on=['patient_id'], how = 'left')

    join_df.time =pd.to_datetime(join_df.time)
    join_df['admission start day icu'] = pd.to_datetime(join_df['admission start day icu'])
    join_df['admission end day ICU'] = pd.to_datetime(join_df['admission end day ICU'])
    control_df['admission start day icu'] = pd.to_datetime(control_df['admission start day icu'])
    control_df['admission end day ICU'] = pd.to_datetime(control_df['admission end day ICU'])
    join_df = join_df[(join_df.time.isnull())|
                      (join_df.time>=join_df['admission start day icu']) &(join_df.time<=join_df['admission end day ICU'])]
    join_df.to_csv('join_df.csv')

    # if there is no stool, replace with end admission
    mask = np.isnat(join_df['time'])
    join_df.loc[mask,'time'] = join_df['admission end day ICU']


    group_functions = {'time': {'first_stool_after_admission': 'min'}}
    stool_df_grp = join_df.groupby(['patient_id','admission start day icu','admission end day ICU']).agg(group_functions)
    stool_df_grp.columns = stool_df_grp.columns.droplevel(0)
    stool_df_grp.reset_index(level=0, inplace=True)
    stool_df_grp.reset_index(level=0, inplace=True)
    stool_df_grp.reset_index(level=0, inplace=True)


    stool_df_grp['first_stool_date_minus_admission_date'] = stool_df_grp['first_stool_after_admission'].subtract(stool_df_grp['admission start day icu'])

    join_stool_control = pd.merge(control_df[['admission start day icu','admission end day ICU','patient_id','LOS ICU','deceaseddate']],
                                  stool_df_grp[['patient_id','admission start day icu','admission end day ICU','first_stool_date_minus_admission_date']],
                                  on = ['patient_id','admission start day icu','admission end day ICU'],how = 'inner')

    join_stool_control[ 'death_in_a_month_from_admission'] = 0
    mask = pd.to_datetime(join_stool_control.deceaseddate).subtract(pd.to_datetime(join_stool_control['admission start day icu']))<=pd.to_timedelta(30,'d')
    join_stool_control.loc[mask, 'death_in_a_month_from_admission'] = 1

    join_stool_control['first_stool_date_minus_admission_date_num'] = round(join_stool_control['first_stool_date_minus_admission_date'].dt.total_seconds() / (24 * 60 * 60), 1)


    '''
    # ----------- count number of szzy and ignore those patients-------------------
    # support Zassi and colostomia - if there is less than 4 suzzi records in num_days(7) - delete those rows and calculate regular way
    # thaa summarize per day and join to valid stool
    stool_df_filtered2 = stool_df_filtered[(stool_df_filtered.parameter_name=='יציאה') &
                                   (stool_df_filtered.value.isin([ "ZASSI"])) | (stool_df_filtered.parameter_name=='קולוסטומיה / אילאוסטומיה')]
    stool_df_filtered2['value_for_sum'] = -999

    #group for all days - get patients_with_colostomea
    group_functions = {'value_for_sum': {'value_for_sum': 'count'}}
    stool_df_filtered2_grp = stool_df_filtered2.groupby(['patient_id']).agg(group_functions)
    stool_df_filtered2_grp.columns = stool_df_filtered2_grp.columns.droplevel(0)
    stool_df_filtered2_grp.reset_index(level=0, inplace=True)
    threshols_for_valid = 4

    patients_with_colostomea = stool_df_filtered2_grp[stool_df_filtered2_grp.value_for_sum >= threshols_for_valid].patient_id

    stool_df_filtered2= stool_df_filtered2[stool_df_filtered2.patient_id.isin(patients_with_colostomea)]
    stool_df_grp2 = group_by_day(stool_df_filtered2, grp_function='min')


    join_types = pd.concat ([stool_df_grp1,stool_df_grp2])
    group_functions = {'value': {'value': 'min'}}
    stool_df_grp = join_types.groupby(['patient_id', 'time','parameter_name']).agg(group_functions)
    stool_df_grp.columns = stool_df_grp.columns.droplevel(0)
    stool_df_grp.reset_index(level=0, inplace=True)
    stool_df_grp.reset_index(level=0, inplace=True)
    stool_df_grp.reset_index(level=0, inplace=True) 
    '''
    join_stool_control.to_csv(output_dir+'signals//'+'stool_in_admission.csv')


def prepare_stool (control_file, num_days_list,
                   stool_file=data_dir + 'wounds_stools_13_18_for_modeling.xlsx',use_also_diarrhea = True):
    print ('start handle_signal_data_new.prepare_stool')

    basic_num_days_stool = 100
    max_num_days = max(num_days_list + [basic_num_days_stool])
    #convert commands to daily summary which will be used later
    # todo - delete comment
    create_daily_data.create_stool_data(control_file, stool_file=stool_file,
                                        num_days=max_num_days)

    for num_days in num_days_list:
        # stool sum per day

        create_lab_data(control_file,
                        lab_file=output_dir + 'stool_count_' + str(max_num_days) + '.csv',
                        lab_parameter_name='Num_stools',
                        num_days=num_days, complete_0_values=True,
                        complete_days=True, output_dir_in_func=output_dir)

        if use_also_diarrhea:
            create_lab_data(control_file,
                            lab_file=output_dir + 'stool_count_diarrhea_' + str(max_num_days) + '.csv',
                            lab_parameter_name='Num_stools_diarrhea', num_days=num_days,
                            complete_0_values=True,
                            complete_days=True, output_dir_in_func=output_dir)


    # #one calculation of basic_num_days_stool days with 0 of stool
    create_lab_data(control_file,
                    lab_file=output_dir + 'stool_count_' + str(max_num_days) + '.csv',
                    lab_parameter_name='Num_stools',
                    num_days= basic_num_days_stool, complete_0_values=True,
                    complete_days=True, output_dir_in_func=output_dir)

def prepare_zonda (control_file, num_days_list):
    print ('start prepare_zonda')
    #convert commands to daily summary which will be used later
    create_daily_data.create_sum_zonda_daily(control_file, zonda_file=data_dir + 'zonda_iliostomy_loss.xlsx',
                                             num_days=max(num_days_list)+1)

    for num_days in num_days_list:

        create_lab_data(control_file,
                        lab_file=output_dir + 'zonda_daily_' + str(max(num_days_list)+1) + '.csv',
                        lab_parameter_name='zonda_daily', num_days=num_days, complete_0_values=True,
                        complete_days=True, output_dir_in_func=output_dir,signals_dir='signals//')

#calculate the apaci in the first day in icu
def prepare_apache (control_file):

        create_daily_data.prepare_apache(control_file)

    # take the values in the first day in icu

def prepare_sofa(control_file, num_days_list):
    print('start prepare_sofa')
    print('start prepare_sofa')
    max_num_days = max(num_days_list)


    create_daily_data.prepare_sofa(control_file,num_days=max_num_days,vasoactive_file=data_dir+'vasoactive_rate_per_hour_for_modeling.csv',pao2_divide_fio2_file= output_dir+'PaO2_divide_FIO2.csv',
                 platelets_file= data_dir+'cbc_13_18.xlsx',bilirubin_file = data_dir+'chemistry_13_18.xlsx',mean_arterial_pressure_file = data_dir+'mean_p_12_17.xlsx',
                 Creatinine_file = data_dir+'chemistry_13_18.xlsx', output_dir_in_func = output_dir)

    for num_days in num_days_list:
        # stool sum per day

        create_lab_data(control_file,
                        lab_file=output_dir + 'sofa_daily_'+str(max_num_days)+'.csv',
                        lab_parameter_name='sofa_score',
                        num_days=num_days, output_dir_in_func=output_dir)

def prepare_rass (control_file, num_days_list):
    
    lab_file = data_dir + 'RASS_for_modeling.xlsx'
    for num_days in num_days_list:
        # stool sum per day

        create_lab_data(control_file,
                        lab_file=lab_file,
                        lab_parameter_name='RASS',
                        num_days=num_days, output_dir_in_func=output_dir)


def prepare_pressor_in_admission(control_file):
    control_df = Utils.read_data(control_file)
    med_file = data_dir+'vasoactive_rate_per_hour_for_modeling.csv'
    med_df = Utils.read_data(med_file)

    commands_df = med_df[
        med_df.parameter_name.isin(['Noradrenaline (Norepinephrine)', 'Vassopressin (Pitressin)','Adrenaline (Epinephrine)'])]

    control_df['admission start day icu'] = pd.to_datetime(control_df['admission start day icu'])
    commands_df.time = pd.to_datetime(commands_df.time)

    join_control_commands = pd.merge(commands_df, control_df[
        ['patient_id', 'date_sample', 'admission start day icu', 'admission end day ICU']], how='inner',
                                     on=['patient_id'])
    # take only command in the first day

    mask_is_active = (
            (join_control_commands.time.subtract(join_control_commands['admission start day icu']) > pd.to_timedelta(
                -0.5, 'd'))
            & (join_control_commands.time.subtract(join_control_commands['admission start day icu']) < pd.to_timedelta(
        1, 'd'))
    )
    col_name = 'is_active_pressor_in_admission'

    join_control_commands[col_name] = 0

    dummy_df = control_df[['patient_id', 'date_sample', 'admission start day icu']]
    dummy_df[col_name] = 0

    join_control_commands.loc[mask_is_active, col_name] = 1

    group_functions = {
        col_name: {col_name: 'max'}
    }

    join_control_commands = pd.concat(
        [dummy_df, join_control_commands[['patient_id', 'date_sample', 'admission start day icu', col_name]]])
    features_df = join_control_commands.groupby(['patient_id', 'date_sample', 'admission start day icu']).agg(
        group_functions)
    features_df.columns = features_df.columns.droplevel(0)
    features_df = features_df.reset_index()

    features_df.to_csv(output_dir + 'signals//' + 'is_pressor_in_admission.csv')


def prepare_intubation_at_admission (control_file):

    control_df = Utils.read_data(control_file)
    catheter_file = 'catheter_data_new.xlsx'
    values_list = ['Tracheostomy - Gantt','Intubation - Gantt']
    commands_df = Utils.read_data(Utils.signal_data_dir + 'Data//' + catheter_file)
    commands_df = commands_df[commands_df.order_name.isin(values_list)]

    control_df['admission start day icu'] = pd.to_datetime(control_df['admission start day icu'] )
    commands_df.start_date = pd.to_datetime(commands_df.start_date)
    commands_df.end_date = pd.to_datetime(commands_df.end_date)

    join_control_commands = pd.merge(commands_df, control_df[
        ['patient_id', 'date_sample', 'admission start day icu', 'admission end day ICU']], how='inner',
                                         on=['patient_id'])
    # take only command in the first day


    mask_is_active = (
            (join_control_commands.start_date.subtract(join_control_commands['admission start day icu'])> pd.to_timedelta(-0.5,'d'))
                      & (join_control_commands.start_date.subtract(join_control_commands['admission start day icu'])< pd.to_timedelta(1,'d'))
    )
    col_name  = 'is_active_intubation_in_admission'

    join_control_commands[col_name] = 0

    dummy_df = control_df[['patient_id', 'date_sample','admission start day icu']]
    dummy_df[col_name] = 0

    join_control_commands.loc[mask_is_active, col_name] =1

    group_functions = {
        col_name:{col_name: 'max'}
    }

    join_control_commands = pd.concat([dummy_df,join_control_commands[['patient_id', 'date_sample','admission start day icu',col_name]]])
    features_df = join_control_commands.groupby(['patient_id','date_sample','admission start day icu']).agg(group_functions)
    features_df.columns = features_df.columns.droplevel(0)
    features_df = features_df.reset_index()

    features_df.to_csv(output_dir+'signals//'+'is_intubated_in_admission.csv')

# join all the prepartion process
#all fileas will be under output_dir+//signals'
def create_signal_data(control_file = output_dir + 'all_features_20180813143540.csv',
                       num_days_list=[3, 5]  #[5,30]
                       ):
    print ('start create_signal_data')

    # ----once------

    # convert_new_control_to_new_source (file_to_convert = data_dir+ 'vector_labs.csv')
    #todo -delete comment
    # data_prepration()
    # # #------------------
    # # #

    # ------relevant for stool:
    prepare_pressor_in_admission (control_file)
    prepare_apache(control_file)
    prepare_rass (control_file, num_days_list)
    prepare_sofa(control_file, num_days_list)
    prepare_medicine (control_file, num_days_list)
    prepare_stool(control_file, num_days_list)

    ## add stool features on admission date : first stool + 30 days mortality
    # prepare_stools_admission (control_file)
    prepare_blood(control_file, num_days_list)
    # ------end relevant for stool

    prepare_intubation_at_admission(control_file)
    prepare_liquid(control_file, num_days_list)
    prepare_heart (control_file, num_days_list)
    prepare_gaz(control_file, num_days_list)
    prepare_zonda(control_file, num_days_list)
    prepare_fusid(control_file, num_days_list)


    print('end create_signal_data')








    # # ---start comment ------create pio2 fio2 ration - long process - summarize one hour before pao2 - no use
    # # create_fio2_pio2_ratio(control_file, pio2_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_pao2_for_modeling.xlsx', fio2_dir='C://Users//michael//Desktop//predicting BSI//files//Data//fio2')
    # # create_fio2_pio2_ratio(control_file,
    # #                        pio2_file='C://Users//michael//Desktop//predicting BSI//files//Data//lab_pao2_for_modeling.xlsx',
    # #                        fio2_dir='C://Users//michael//Desktop//predicting BSI//files//Data//peep', num_days=5,
    # #                        fio2_file_prefix='peep',new_param_name='PEEP')
    # # -end long process----------
    ## --end comment------------



#----------------
#-----------------end data preparation ------------------

if __name__ == "__main__":

    # -------------trials--------
    # on partial data
    # get_basic_blood_pressure(control_file=output_dir+'all_features_20180420120525.csv', blood_pressure_file=data_dir + 'diastolic_p_12_17_partial.xlsx',
    #                          medicine_file=data_dir + 'vasoactive_icu_12_17_for_modeling_partial.xlsx',
    #                          parameter_name='Arterial Pressure Diastolic', min_value=10, max_value=250, num_days=30)


    # get_basic_blood_pressure(control_file=output_dir + 'all_features_20180420120525.csv',
    #                          blood_pressure_file=data_dir + 'diastolic_p_12_17.xlsx',
    #                          medicine_file=data_dir + 'vasoactive_icu_12_17_for_modeling.xlsx',
    #                          parameter_name='Arterial Pressure Diastolic', min_value=10, max_value=250, num_days=30)

    # create_short_signal_features()

    # summary_signal_windows_with_basic_value(short_signal_file = output_dir+'short_signals_20180404000538.csv',
    #                                         basic_values_file = output_dir+'basic_Arterial_Pressure_Diastolic.csv',
    #                                         paramater_name = 'Arterial Pressure Diastolic')
    pass
