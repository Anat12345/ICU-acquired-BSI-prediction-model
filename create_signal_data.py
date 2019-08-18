import numpy as np
import create_daily_data
import Utils
import handle_signal_data_new
import mimic.create_control_set
import mimic.prepare_data_files
from os import path
import sys
import pandas as pd
data_dir = 'C://Users//michael//Desktop//predicting BSI//files//mimic//Data//'
output_dir = "C://Users//michael//Desktop//predicting BSI//files//mimic//Output//"

def prepare_liquid(control_file, num_days_list):

    # create total liquid per day for every patient
    mimic.prepare_data_files.create_input_liquid_data(control_file,
                         mimic.create_control_set.inputevents_mv_file_name,
                          mimic.create_control_set.d_items_file_name,
                          output_file=data_dir + 'input_liquids_per_hour_for_modeling.csv',num_days=max(num_days_list))

    mimic.prepare_data_files.create_output_liquid_data(control_file,
                                                      mimic.create_dataset.outputevents_file_name,
                                                      mimic.create_dataset.d_items_file_name,
                                                      output_file = data_dir+'output_liquid_urine.csv')



    mimic.prepare_data_files.create_liquid_balance(control_file, num_days_list, min_value=0, max_value=3000,
                          input_liquids_per_hour_for_modeling_file=data_dir + 'input_liquids_per_hour_for_modeling.csv',
                          urine_filename=data_dir + 'output_liquid_urine.csv')


def prepare_signals(control_file, data_file = data_dir+'Respiratory_data_for_modeling.csv',num_days_list=[5]):

    print ('start prepare_signals - '+data_file)
    data_df = pd.read_csv(data_file, encoding="ISO-8859-1")
    data_df = data_df.rename(columns = {'SUBJECT_ID':'patient_id','VALUE':'value','CHARTTIME':'time','LABEL':'parameter_name'})
    # data_df.to_csv(data_file)

    group_functions = {'patient_id' : {'patient_id_nunique': 'nunique', 'rows_cnt' :'count'}}
    data_df_grp = data_df.groupby(['parameter_name']).agg(group_functions)
    data_df_grp.columns = data_df_grp.columns.droplevel(0)
    data_df_grp.reset_index(level=0, inplace=True)
    print (data_df_grp)

    blood_df_grp_filter = data_df_grp[data_df_grp.patient_id_nunique>200]
    #take only numeric values

    for param_name in blood_df_grp_filter.parameter_name :

        calc_std_flag = True
        smooth_signal = True
        use_fft_feature = True
        if param_name=='Heart Rate':
            min_value = 35
            max_value = 210
        elif param_name == 'Temperature Fahrenheit':
            min_value = 93.2
            max_value = 113
        elif param_name=='Temperature Celsius':
            min_value = 34.5
            max_value = 45
        elif param_name in ['Arterial Blood Pressure mean','ART BP mean','Non Invasive Blood Pressure mean']:
            min_value = 25
            max_value = 330
        elif param_name in ['Arterial Blood Pressure diastolic','ART BP Diastolic','Non Invasive Blood Pressure diastolic',
                            'Manual Blood Pressure Diastolic Right']:
            min_value = 20
            max_value = 310
        elif param_name in  ['ART BP Systolic','Arterial Blood Pressure systolic','Non Invasive Blood Pressure systolic',
                             'Manual Blood Pressure Systolic Right','Manual Blood Pressure Systolic Left']:
            min_value = 30
            max_value = 350
        elif param_name=='Respiratory Rate':
            min_value = 4
            max_value = 80


        else:
            min_value = -99999
            max_value = 99999
            calc_std_flag = False
            smooth_signal = False
            use_fft_feature = False
        for num_days in num_days_list:

            try:
                handle_signal_data_new.create_lab_data(control_file,
                                                       lab_df=data_df,
                                                       lab_parameter_name=param_name,min_value=min_value,
                                                       max_value=max_value,
                                                       smooth_signal=smooth_signal,use_fft_feature=use_fft_feature
                                                       ,calc_std_flag=calc_std_flag,
                                                       num_days=num_days, output_dir_in_func=output_dir)
            except Exception:
                print(param_name + ' - create_lab_data failed')
                pass




# take all numeric values as features (exists for a t least 100 patients)
def prepare_blood(control_file, blood_lab_file = data_dir+'blood_data_for_modeling.csv',num_days_list=[5]):

    blood_lab_file_clean = data_dir+'blood_data_for_modeling_clean.csv'

    blood_df = pd.read_csv(blood_lab_file, encoding="ISO-8859-1")
    blood_df = blood_df[blood_df['value'].str.contains('>|<|GREATER|ERROR|UNABLE|LESS|VOIDED|-') == False]
    blood_df['value'] = blood_df['value'].str.replace('+','')
    blood_df = blood_df[blood_df['value'].apply(lambda x: x[0].isdigit())]
    blood_df.to_csv(blood_lab_file_clean)


    group_functions = {'patient_id' : {'patient_id_nunique': 'nunique', 'rows_cnt' :'count'}}
    blood_df_grp = blood_df.groupby(['parameter_name']).agg(group_functions)
    blood_df_grp.columns = blood_df_grp.columns.droplevel(0)
    blood_df_grp.reset_index(level=0, inplace=True)
    # print (blood_df_grp)

    blood_df_grp_filter = blood_df_grp[blood_df_grp.patient_id_nunique>200]
    #take only numeric values

    for param_name in blood_df_grp_filter.parameter_name :
        if param_name in ['Ventilation Rate','Anisocytosis','Fibrin Degradation Products']:
            continue
        # if param_name < 'Hypochromia':
        #     print ('skip '+param_name)
        #     continue
        for num_days in num_days_list:
            handle_signal_data_new.create_lab_data(control_file,
                                                   lab_df=blood_df,
                                                   lab_parameter_name=param_name,
                                                   num_days=num_days, output_dir_in_func=output_dir)


def prepare_stool(control_file, stool_file,num_days_list=[5]):

    print (' start mimic - prepare_stool')
    handle_signal_data_new.prepare_stool(control_file,num_days_list,stool_file=stool_file,use_also_diarrhea=False)


def prepare_medicine (control_file, num_days_list):
    print ('start prepare_medicine')

    #---vasoactive-------------
    org_file = data_dir+'mimic_medications_data_for_modeling.csv'
    new_file_name = output_dir+'mimic_medications_data_with_dummy.csv'

    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             org_file,
                                                             'Norepinephrine',
                                                             output_file=new_file_name)
    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Phenylephrine',
                                                             output_file=new_file_name)
    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Epinephrine',
                                                             output_file=new_file_name)
    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Dopamine',
                                                             output_file=new_file_name)
    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Vasopressin',
                                                             output_file=new_file_name)
    for num_days in num_days_list:
        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Norepinephrine', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Phenylephrine', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Epinephrine', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Dopamine', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)
        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Vasopressin', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)


    #------------Morphine/Fentanyl/Propofol 2% ( Diprivan )' (only 1 patient with 1%), 'Dormicum'

    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             org_file,
                                                             'Midazolam(Versed)',
                                                             output_file=new_file_name)
    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                            updated_vasoactive_file,
                                                             'Fentanyl',
                                                             output_file=new_file_name)
    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Morphine Sulfate',
                                                             output_file=new_file_name)
    updated_vasoactive_file = handle_signal_data_new.add_dummy_rows_for_control_set(control_file,
                                                             updated_vasoactive_file,
                                                             'Propofol',
                                                             output_file=new_file_name)


    for num_days in num_days_list:
        # todo - delete  comment
        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Fentanyl', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Morphine Sulfate', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Propofol', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=updated_vasoactive_file,
                        lab_parameter_name='Midazolam(Versed)', num_days=num_days, complete_0_values=True, complete_hours=True, output_dir_in_func=output_dir)

#different logic from rambam because there is fio2 every hour and gaz 3 times a day
def create_fio2_pio2_ratio (control_file,pio2_file, fio2_file,num_days = 5,
                            new_param_name = 'FIO2',default_val=0,
                            num_hours_to_search = 8):

    print('start create_fio2_pio2_ratio - prepare '+new_param_name)
    fio2_parameter_name = 'FIO2'
    control_df = Utils.read_data(control_file)#to do delete nrows
    pio2_df = Utils.read_data( pio2_file)

    pio2_df = pio2_df[pio2_df.parameter_name=='pO2']

    fio2_df = Utils.read_data(fio2_file)
    # old - when we used chartevents table
    # fio2_df = fio2_df.rename(columns = {'SUBJECT_ID':'patient_id','VALUE':'value','CHARTTIME':'time','LABEL':'parameter_name'})

    fio2_df = fio2_df[fio2_df.parameter_name ==fio2_parameter_name ]
    print ('finish reading files')

    control_df.date_sample = pd.to_datetime(control_df.date_sample)
    pio2_df.date_sample = pd.to_datetime(pio2_df.date_sample)
    fio2_df.date_sample = pd.to_datetime(fio2_df.date_sample)
    # every patient has pio2
    df_join  = pd.merge(pio2_df,control_df[['patient_id','date_sample','admission start day icu']],
                        how = 'inner',on = ['patient_id','date_sample'])

    df_join.date_sample = pd.to_datetime(df_join.date_sample)
    df_join.time = pd.to_datetime(df_join.time)
    df_join['admission start day icu'] = pd.to_datetime(df_join['admission start day icu'])

    df_join['time_minus_start_icu'] = df_join.time.subtract(df_join['admission start day icu'])
    df_join['date_sample_minus_time'] = df_join.date_sample.subtract(df_join.time)

    #takes data from first day for apache
    control_pio2_join = df_join[(((df_join.date_sample_minus_time <= pd.Timedelta(num_days, unit='d'))&
                       (df_join.date_sample_minus_time >= pd.Timedelta(0, unit='d'))))
                    | ((df_join['time_minus_start_icu'] <= pd.Timedelta(1.5, unit='d')) &
                        (df_join['time_minus_start_icu'] >= pd.Timedelta(-0.5, unit='d')))]



    control_pio2_join = control_pio2_join.rename(columns={'time': 'time_pio2'})
    control_pio2_join = control_pio2_join.rename(columns={'value': 'value_pio2'})



    control_fio2_join  = pd.merge(control_df[['patient_id','date_sample']],
                                  fio2_df,how = 'inner',on = ['patient_id','date_sample'])
    control_fio2_join.date_sample = pd.to_datetime(control_fio2_join.date_sample)
    control_fio2_join.time = pd.to_datetime(control_fio2_join.time)
    control_fio2_join['admission start day icu'] = pd.to_datetime(control_fio2_join['admission start day icu'])

    control_fio2_join['time_minus_start_icu'] = control_fio2_join.time.subtract(control_fio2_join['admission start day icu'])
    control_fio2_join['date_sample_minus_time'] = control_fio2_join.date_sample.subtract(control_fio2_join.time)

    #takes data from first day for apache
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
    fio2_fio2_df_dummy['parameter_name_fio2'] = fio2_parameter_name
    fio2_fio2_df_dummy['source'] = 'dummy'

    control_fio2_join = pd.concat([control_fio2_join[['source','patient_id','time_fio2','value_fio2','parameter_name_fio2','date_sample']],
                                   fio2_fio2_df_dummy[
                                       ['source', 'patient_id', 'time_fio2', 'value_fio2', 'parameter_name_fio2',
                                        'date_sample']]])

    #11/12 - change to left , so no fio2 will be replaced by default val
    fio2_pio2_df = pd.merge (control_pio2_join[['patient_id','time_pio2','value_pio2','date_sample']],
                             control_fio2_join[['source','patient_id','time_fio2','value_fio2','parameter_name_fio2','date_sample']],
                             on =['patient_id','date_sample'],how = 'left')


    # 10/01/2019 - not need
    # # when there is no fio2 or no fio2 in the hours before
    # mask = ((fio2_pio2_df.time_pio2.subtract(fio2_pio2_df.time_fio2)>=pd.to_timedelta(num_hours_to_search,unit = 'h'))
    #         |
    #         (np.isnat(fio2_pio2_df.time_fio2))
    #         |
    #         (fio2_pio2_df.time_pio2<fio2_pio2_df.time_fio2))
    # fio2_pio2_df.loc[mask,'value_fio2'] = default_val
    # fio2_pio2_df.loc[mask, 'time_fio2'] = fio2_pio2_df.loc[mask, 'time_pio2']
    # fio2_pio2_df.to_csv('tmp.csv')
    #take value from last round hour



    fio2_pio2_df_filter = fio2_pio2_df[((fio2_pio2_df.time_pio2.subtract(fio2_pio2_df.time_fio2)<=pd.to_timedelta(num_hours_to_search,unit = 'h'))&
                                          (fio2_pio2_df.time_pio2 >= fio2_pio2_df.time_fio2))
                                        | (np.isnat(fio2_pio2_df.time_fio2))]


    print ('--------------')
    # print(fio2_pio2_df_filter.keys())

    fio2_pio2_df_filter.sort_values(inplace=True,by=['patient_id', 'time_pio2','time_fio2'], ascending=[True, True, True])
    # fio2_pio2_df_filter.to_csv(output_dir+'fio2_pio2_df_filter.csv')
    fio2_pio2_df_filter = fio2_pio2_df_filter.drop_duplicates(subset = ['patient_id','date_sample','time_pio2','parameter_name_fio2'],keep = 'last')
    # fio2_pio2_df_filter.to_csv(output_dir + 'fio2_pio2_df_filter_after_drop_duplications.csv')
    fio2_pio2_df_filter['value_fio2'] = fio2_pio2_df_filter['value_fio2'].fillna(default_val)

    relevant_fio2 = fio2_pio2_df_filter[['patient_id','date_sample','time_pio2','parameter_name_fio2','value_fio2','source']]
    relevant_fio2.columns = ['patient_id','date_sample','time','parameter_name','value','source']

    relevant_fio2.parameter_name = fio2_parameter_name

    relevant_fio2.to_csv(output_dir+new_param_name+'_calculated.csv')

def join_fio2_sources(control_file,blood_file=data_dir+ 'blood_data_for_modeling_clean.csv',
                      respiratory_file=data_dir + 'Respiratory_data_for_modeling.csv',num_days = 5):
    control_df = Utils.read_data(control_file)
    #-------------------------
    fio2_df = Utils.read_data(respiratory_file)
    fio2_df = fio2_df.rename(columns = {'SUBJECT_ID':'patient_id','VALUE':'value','CHARTTIME':'time','LABEL':'parameter_name'})
    fio2_parameter_name = 'Inspired O2 Fraction'
    fio2_df = fio2_df[fio2_df.parameter_name ==fio2_parameter_name ]
    control_fio2_join  = pd.merge(control_df[['patient_id','date_sample','admission start day icu']],fio2_df,how = 'inner',on = ['patient_id'])
    control_fio2_join.date_sample = pd.to_datetime(control_fio2_join.date_sample)
    control_fio2_join.time = pd.to_datetime(control_fio2_join.time)
    control_fio2_join['admission start day icu'] = pd.to_datetime(control_fio2_join['admission start day icu'])

    control_fio2_join['time_minus_start_icu'] = control_fio2_join.time.subtract(control_fio2_join['admission start day icu'])
    control_fio2_join['date_sample_minus_time'] = control_fio2_join.date_sample.subtract(control_fio2_join.time)

    #takes data from first day for apache
    control_fio2_join_a = control_fio2_join[(((control_fio2_join.date_sample_minus_time <= pd.Timedelta(num_days, unit='d'))&
                       (control_fio2_join.date_sample_minus_time >= pd.Timedelta(0, unit='d'))))
                    | ((control_fio2_join['time_minus_start_icu'] <= pd.Timedelta(1.5, unit='d')) &
                        (control_fio2_join['time_minus_start_icu'] >= pd.Timedelta(-0.5, unit='d')))]

    control_fio2_join_a['source'] = 'chartevents'
    #--------------------------------------
    fio2_df = Utils.read_data(blood_file)
    fio2_parameter_name = 'Oxygen'
    fio2_df = fio2_df[fio2_df.parameter_name ==fio2_parameter_name ]
    control_fio2_join  = pd.merge(control_df[['patient_id','date_sample','admission start day icu']],
                                  fio2_df[['patient_id','date_sample','time','value']],how = 'inner',on = ['patient_id','date_sample'])
    control_fio2_join.date_sample = pd.to_datetime(control_fio2_join.date_sample)
    control_fio2_join.time = pd.to_datetime(control_fio2_join.time)
    control_fio2_join['admission start day icu'] = pd.to_datetime(control_fio2_join['admission start day icu'])

    control_fio2_join['time_minus_start_icu'] = control_fio2_join.time.subtract(control_fio2_join['admission start day icu'])
    control_fio2_join['date_sample_minus_time'] = control_fio2_join.date_sample.subtract(control_fio2_join.time)

    #takes data from first day for apache
    control_fio2_join_b = control_fio2_join[(((control_fio2_join.date_sample_minus_time <= pd.Timedelta(num_days, unit='d'))&
                       (control_fio2_join.date_sample_minus_time >= pd.Timedelta(0, unit='d'))))
                    | ((control_fio2_join['time_minus_start_icu'] <= pd.Timedelta(1.5, unit='d')) &
                        (control_fio2_join['time_minus_start_icu'] >= pd.Timedelta(-0.5, unit='d')))]

    control_fio2_join_b['source'] = 'blood_gaz'
    #---------------------------
    control_fio2_join_all = pd.concat([control_fio2_join_a,control_fio2_join_b])
    control_fio2_join_all['parameter_name'] = 'FIO2'

    output_file = output_dir+'fio2_both_sources.csv'
    control_fio2_join_all.to_csv(output_file)
    return output_file


def prepare_breathing (control_file,num_days_list):

    blood_file = data_dir+ 'blood_data_for_modeling_clean.csv'
    print ('start prepare_breathing')
    num_days = 100 # this is used also for apache - first day in icu so we need all the data
    # -------10/01/2019 - no need - data in mimic is in the same hours
    #todo- delete comment
    fio2_joined_file = join_fio2_sources(control_file,blood_file=blood_file,respiratory_file=data_dir + 'Respiratory_data_for_modeling.csv')
    fio2_joined_file = output_dir + 'fio2_both_sources.csv'
    create_fio2_pio2_ratio(control_file,
                           pio2_file=blood_file,
                           fio2_file=fio2_joined_file,
                           num_days=num_days,
                           new_param_name='FIO2',default_val=30)

    handle_signal_data_new.create_A_aPO2(paco2_file=blood_file,
                                         paco2_param_name = 'pCO2',
                  fio2_file = output_dir+'FIO2_calculated.csv', fio2_param_name = 'FIO2',
                  pao2_file=blood_file,pao2_param_name = 'pO2')
    # ---------end no need
    #todo - delete comment
    df_to_add = handle_signal_data_new.create_signal_combination(blood_file,
                                          'pO2', 'FIO2', 'divide', fio2_joined_file,min_val_feature2=21)

    for num_days in num_days_list:
        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=output_dir + 'pO2_divide_FIO2.csv',
                        lab_parameter_name='pO2_divide_FIO2', num_days=num_days,
                        output_dir_in_func=output_dir)

        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=output_dir + 'A_aPO2.csv',
                        lab_parameter_name='A_aPO2', num_days=num_days, output_dir_in_func=output_dir,
                        complete_0_values=True, complete_hours=True)


def create_signal_data(control_file,num_days_list=[5]):

    print ('start mimic - create_signal_data')
    blood_lab_file = data_dir + 'blood_data_for_modeling.csv'
    stool_file = data_dir+'mimic_stools.csv'
    antib_file = data_dir+'mimic_anti_data.csv'

    #prepare stool + all signals data from chartevents - once
    # data_preparation_for_all_patients()

    # todo - delete comment
    data_preparation(control_file,max(num_days_list))
    #
    # # ---checked
    prepare_blood(control_file, blood_lab_file=blood_lab_file, num_days_list=num_days_list)
    prepare_medicine(control_file, num_days_list)
    prepare_fusid(control_file, num_days_list)
    prepare_signals(control_file,data_dir+'Routine_Vital_Signs_data_for_modeling.csv',num_days_list = num_days_list)
    prepare_signals(control_file, data_dir + 'Respiratory_data_for_modeling.csv',num_days_list = num_days_list)
    prepare_signals(control_file, data_dir + 'Hemodynamics_data_for_modeling.csv',num_days_list = num_days_list)
    prepare_signals(control_file, data_dir + 'Labs_data_for_modeling.csv',num_days_list = num_days_list)
    prepare_signals(control_file, data_dir + 'general_data_for_modeling.csv',num_days_list = num_days_list)
    prepare_GSC(control_file,num_days_list)
    # stool in mimic is not good
    prepare_stool(control_file, stool_file, num_days_list=num_days_list)
    prepare_breathing (control_file,num_days_list)

    # prepare_intubation_at_admission(control_file)
    prepare_apache(control_file)
    prepare_pressor_in_admission(control_file)
    # #--- end checked



    # for married - demographic + apachi: data_preparation,prepare_breathing,prepare_apache
    #
    #



    #--- to check]
    # todo - need to check - didn't run it till now
    # prepare_liquid(control_file, num_days_list)


    # prepare_zonda(control_file, num_days_list)
    #
    # prepare_sofa(control_file, num_days_list)

def prepare_pressor_in_admission (control_file):
    control_df = Utils.read_data(control_file)
    med_file = data_dir + 'mimic_medications_data_for_modeling.csv'
    med_df = Utils.read_data(med_file)

    commands_df = med_df[med_df.parameter_name.isin(['Vasopressin','Dopamine','Epinephrine','Phenylephrine','Norepinephrine'])]

    control_df['admission start day icu'] = pd.to_datetime(control_df['admission start day icu'] )
    commands_df.time = pd.to_datetime(commands_df.time)

    join_control_commands = pd.merge(commands_df, control_df[
        ['patient_id', 'date_sample', 'admission start day icu', 'admission end day ICU']], how='inner',
                                         on=['patient_id'])
    # take only command in the first day


    mask_is_active = (
            (join_control_commands.time.subtract(join_control_commands['admission start day icu'])> pd.to_timedelta(-0.5,'d'))
                      & (join_control_commands.time.subtract(join_control_commands['admission start day icu'])< pd.to_timedelta(1,'d'))
    )
    col_name  = 'is_active_pressor_in_admission'

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

    features_df.to_csv(output_dir+'signals//'+'is_pressor_in_admission.csv')

def prepare_intubation_at_admission (control_file):

    control_df = Utils.read_data(control_file)
    catheter_file = data_dir+'ventsettings.csv'

    commands_df = Utils.read_data(catheter_file)
    commands_df = commands_df[commands_df.MechVent ==1]

    commands_df = commands_df.rename(
        columns={'subject_id': 'patient_id',  'charttime': 'time', 'LABEL': 'parameter_name'})

    control_df['admission start day icu'] = pd.to_datetime(control_df['admission start day icu'] )
    commands_df.time = pd.to_datetime(commands_df.time)

    #indication to intubation
    commands_df = commands_df[commands_df.MechVent==1]

    join_control_commands = pd.merge(commands_df, control_df[
        ['patient_id', 'date_sample', 'admission start day icu', 'admission end day ICU']], how='inner',
                                         on=['patient_id'])
    # take only command in the first day


    mask_is_active = (
            (join_control_commands.time.subtract(join_control_commands['admission start day icu'])> pd.to_timedelta(-0.5,'d'))
                      & (join_control_commands.time.subtract(join_control_commands['admission start day icu'])< pd.to_timedelta(1,'d'))
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

def prepare_diarrhea_med (control_file):
    pass

def prepare_fusid (control_file,num_days_list):

    handle_signal_data_new.prepare_fusid(control_file,num_days_list=num_days_list)


def prepare_apache(control_file, signals_dir='signals//'):

    blood_data_file = data_dir + 'blood_data_for_modeling_clean.csv'
    diagnosis_file = output_dir+signals_dir+'diagnoses.csv'
    diagnosis_df = Utils.read_data(diagnosis_file)


    control_df = Utils.read_data(control_file)
    control_df = pd.merge(control_df[['patient_id', 'age','admission start day icu']],diagnosis_df,how = 'left',on = ['patient_id'])
    # -------- age

    age_df = control_df[
        ['patient_id', 'age','date_sample','admission start day icu','diagnoses_is_chf', 'diagnoses_is_copd', 'diagnoses_is_cirrhosis']]
    age_df = age_df.rename(columns={'Reference Event age at event': 'age'})
    age_df = create_daily_data.add_age_apachi_score(age_df)

    #
    # # #-------- WBC
    parameter_name = 'WBC'
    wbc_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                               data_file=blood_data_file,
                                                               parameter_name_list=['White Blood Cells', 'WBC Count'])
    wbc_df = create_daily_data.add_wbc_apachi_score(wbc_df)

    # #--------Hematocrit
    hematocrit_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                    data_file=blood_data_file,
                                                    parameter_name_list=['Hematocrit','Hematocrit, Calculated'])
    hematocrit_df = create_daily_data.add_hematocrit_apachi_score(hematocrit_df)

    # #--------Creatinine
    Creat_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                                 data_file=blood_data_file,
                                                                 parameter_name_list=['Creatinine'])
    Creat_df = create_daily_data.add_creat_apachi_score(Creat_df)

    # #--------Potassium
    potassium_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                                     data_file=blood_data_file,
                                                                     parameter_name_list=['Potassium','Potassium, Whole Blood'])
    potassium_df = create_daily_data.add_potassium_apachi_score(potassium_df)

    # #-------Sodium
    sodium_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                                  data_file=blood_data_file,
                                                                  parameter_name_list=['Sodium','Sodium, Whole Blood'])
    sodium_df = create_daily_data.add_sodium_apachi_score(sodium_df)

    # #-------PH
    ph_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                              data_file=blood_data_file,
                                                              parameter_name_list=['pH'])
    ph_df = create_daily_data.add_ph_apachi_score(ph_df)

    #-------PaO2 - todo
    fio2_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                data_file= output_dir + 'FIO2_calculated.csv',
                                                                parameter_name_list=['FIO2'])
    fio2_df = fio2_df.rename(columns={'min_val': 'min_val_fio2', 'max_val': 'max_val_fio2'})
    A_apao2_df = create_daily_data.get_min_max_value_first_day_icu(control_file, data_file=output_dir+'A_aPO2.csv',
                                                                   parameter_name_list=['A_aPO2'])
    A_apao2_df = A_apao2_df.rename(columns={'min_val': 'min_val_A_apao2', 'max_val': 'max_val_A_apao2'})
    pao2_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                    data_file=blood_data_file,
                                                                parameter_name_list=['pO2'])
    pao2_df = pao2_df.rename(columns={'min_val': 'min_val_pao2', 'max_val': 'max_val_pao2'})

    breath_df = pd.merge(pao2_df, A_apao2_df, on=['patient_id', 'date_sample'], how='left')
    breath_df = pd.merge(breath_df, fio2_df, on=['patient_id', 'date_sample'], how='left')

    # breath_df = pd.merge(fio2_df, A_apao2_df, on=['patient_id', 'date_sample'], how='inner')
    # breath_df = pd.merge(breath_df, pao2_df, on=['patient_id', 'date_sample'], how='inner')
    breath_df = create_daily_data.add_breath_apachi_score(breath_df)

    # -----Respiratory Rate
    rr_df = create_daily_data.get_min_max_value_first_day_icu(control_file, data_file=data_dir + 'Respiratory_data_for_modeling.csv',
                                                              parameter_name_list=['Respiratory Rate','Respiratory Rate (Set)','Respiratory Rate (spontaneous)','Respiratory Rate (Total)']
                                            , min_value=4, max_value=60)
    rr_df['min_val'] = np.round(rr_df['min_val'])
    rr_df['max_val'] = np.round(rr_df['max_val'])
    rr_df = create_daily_data.add_rr_apachi_score(rr_df)

    # # HR
    hr_df = create_daily_data.get_min_max_value_first_day_icu(control_file, data_file=data_dir + 'Routine_Vital_Signs_data_for_modeling.csv',
                                                              parameter_name_list=['Heart Rate'], min_value=35, max_value=210)
    hr_df = create_daily_data.add_hr_apachi_score(hr_df)

    # Mean Arterial Pressure
    map_df = create_daily_data.get_min_max_value_first_day_icu(control_file, data_file=data_dir + 'Routine_Vital_Signs_data_for_modeling.csv',
                                                               parameter_name_list=['Arterial Blood Pressure mean','ART BP mean','Non Invasive Blood Pressure mean'], min_value=25, max_value=330)
    map_df = create_daily_data.add_map_apachi_score(map_df)

    # temperature
    tmp_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                               data_file=data_dir + 'Routine_Vital_Signs_data_for_modeling.csv',
                                                               parameter_name_list=['Temperature Fahrenheit'],
                                                               min_value=94.1, max_value=113)
    tmp_df = create_daily_data.add_tmp_apachi_score(tmp_df,'F')

    #GSC

    gsc_df = create_daily_data.get_min_max_value_first_day_icu(control_file,
                                                               data_file=data_dir + 'mimic_gsc_score.csv',
                                                               parameter_name_list=['gsc'])
    gsc_df = create_daily_data.add_gsc_apachi_score(gsc_df)

    # map_df = Utils.read_data(output_dir + 'apache_map.csv')
    # tmp_df = Utils.read_data(output_dir + 'apache_tmp.csv')
    # age_df = Utils.read_data(output_dir + 'apache_age.csv')
    # wbc_df = Utils.read_data(output_dir + 'apache_wbc.csv')
    # hematocrit_df = Utils.read_data(output_dir + 'apache_hematocrit.csv')
    # Creat_df = Utils.read_data(output_dir + 'apache_creat.csv')
    # potassium_df = Utils.read_data(output_dir + 'apache_potassium.csv')
    # sodium_df= Utils.read_data(output_dir + 'apache_sodium.csv')
    # ph_df= Utils.read_data(output_dir + 'apache_ph.csv')
    # hr_df = Utils.read_data(output_dir + 'apache_hr.csv')
    # rr_df = Utils.read_data(output_dir + 'apache_rr.csv')
    # breath_df= Utils.read_data(output_dir + 'apache_breath.csv')
    # gsc_df = Utils.read_data(output_dir + 'apache_gsc.csv')

    mylist = [age_df, wbc_df, hematocrit_df, Creat_df, potassium_df, sodium_df, ph_df, rr_df, hr_df, map_df, tmp_df,
              breath_df,gsc_df]

    age_df.date_sample = pd.to_datetime(age_df.date_sample)
    wbc_df.date_sample = pd.to_datetime(wbc_df.date_sample)
    hematocrit_df.date_sample = pd.to_datetime(hematocrit_df.date_sample)
    Creat_df.date_sample = pd.to_datetime(Creat_df.date_sample)
    potassium_df.date_sample = pd.to_datetime(potassium_df.date_sample)
    sodium_df.date_sample = pd.to_datetime(sodium_df.date_sample)
    ph_df.date_sample = pd.to_datetime(ph_df.date_sample)
    rr_df.date_sample = pd.to_datetime(rr_df.date_sample)
    hr_df.date_sample = pd.to_datetime(hr_df.date_sample)
    map_df.date_sample = pd.to_datetime(map_df.date_sample)
    tmp_df.date_sample = pd.to_datetime(tmp_df.date_sample)
    breath_df.date_sample = pd.to_datetime(breath_df.date_sample)
    gsc_df.date_sample = pd.to_datetime(gsc_df.date_sample)

    for i in range(len(mylist)):
        if i == 0:
            result = mylist[i]
        else:
            result = pd.merge(
                result,
                mylist[i],
                how='left',
                on=['patient_id', 'date_sample']
            )

    # count missing scores
    result['num_scores_exist'] = result[
        ['age_score', 'wbc_score', 'Haematocrit_score', 'creat_score', 'potassium_score', 'sodium_score', 'ph_score',
         'rr_score', 'hr_score', \
         'map_score', 'tmp_score','gsc_score']].apply(lambda x: x.count(), axis=1)

    result.to_csv(output_dir + 'apache_parameters.csv')
    # result = Utils.read_data(output_dir+'apache_parameters.csv')

    result = result.fillna(0)
    print(result.keys())
    result['apache_score'] = result['age_score'] + result['wbc_score'] + result['Haematocrit_score'] + result['creat_score'] + \
                             result['potassium_score'] + result['sodium_score'] + result['ph_score'] + result[
                                 'rr_score'] + result['hr_score'] + result['map_score'] + \
                             result['tmp_score'] +result['gsc_score']+     result['breath_score']\
                            +5 * result[
                                 ['diagnoses_is_chf', 'diagnoses_is_copd', 'diagnoses_is_cirrhosis']].max(axis=1) \

    result.loc[result.num_scores_exist < 9, 'apache_score'] = -999
    result[['patient_id', 'date_sample', 'apache_score']].to_csv(
        output_dir + signals_dir + 'signals_apache_parameters.csv')


def data_preparation_for_all_patients ():

    # checked - for all patients - stool is partial
    # mimic.prepare_data_files.create_stool_data(mimic.create_control_set.outputevents_file_name)

    # checked
    # mimic.prepare_data_files.create_anti_data(  mimic.create_control_set.inputevents_mv_file_name,
    #                                             mimic.create_control_set.d_items_file_name)



    # checked
    # mimic.prepare_data_files.create_medications_data(  mimic.create_control_set.inputevents_mv_file_name,
    #                                             mimic.create_control_set.d_items_file_name)

    # checked
    # mimic.prepare_data_files.create_catheter_data( mimic.create_control_set.procedureevents_mv_file_name,
    #                                             mimic.create_control_set.d_items_file_name)

    # checked
    # mimic.prepare_data_files.prepare_chartevents_files(d_items_file_name=mimic.create_control_set.d_items_file_name)

    #checked
    # mimic.prepare_data_files.prepare_GSC_data(gsc_file=data_dir + 'GSC_score_data_for_modeling.csv')

    mimic.prepare_data_files.prepare_intubation_data()
    pass

def prepare_GSC (control_file, num_days_list):
    print ('start prepare_GSC')
    for num_days in num_days_list:
        handle_signal_data_new.create_lab_data(control_file,
                        lab_file=data_dir + 'mimic_gsc_score.csv',
                        lab_parameter_name='gsc', num_days=num_days,
                        output_dir_in_func=output_dir)

def data_preparation (control_file, max_days_for_signals = 5):

    control_df = pd.read_csv(control_file, encoding="ISO-8859-1")
    #prepare lab file for control_file
    mimic.prepare_data_files.create_labs_files (control_df,
                                                mimic.create_control_set.labs_file_name,
                                                mimic.create_control_set.d_labitems_file_name,
                                                num_days=mimic.create_control_set.num_days_to_take_from_lab)

    mimic.prepare_data_files.prepare_diagnosis_data(control_file)






if __name__ == "__main__":
    # data_set_for_modeling_file = output_dir + 'all_dataset_clean.csv'
    # create_signal_data(data_set_for_modeling_file,num_days_list=[5])

    data = Utils.read_data(data_dir+'Routine_Vital_Signs_data_for_modeling.csv')
    print (data.shape)
    # print (data.keys())
    data[data.HADM_ID==138376].to_csv(output_dir+'vital_hadm_138376.csv')
    data[data.ICUSTAY_ID == 256064].to_csv(output_dir + 'vital_hadm_138376_a.csv')
    data[data.ITEMID == 225312].to_csv(output_dir + 'vital_hadm_138376_b.csv')