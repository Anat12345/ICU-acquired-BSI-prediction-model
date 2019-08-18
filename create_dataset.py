
import create_vasoactive_data
import numpy as np
import pandas as pd
import  my_modeling
from sklearn.model_selection import train_test_split
import datetime
import Handle_promateus_data
import os
from datetime import datetime, timedelta
import handle_signal_data_new
import create_daily_data
import create_liquids_file
import mimic.create_control_set
import mimic.create_signal_data
import mimic.prepare_data_files
import Utils
'''
after running :
1. create_control_set
2. create_signal_data
need to join all features to one files
'''

data_dir = 'C://Users//michael//Desktop//predicting BSI//files//mimic//Data//'
output_dir = "C://Users//michael//Desktop//predicting BSI//files//mimic//Output//"


#------------------------------------


# creates the dataset : control set from samples + lab files (data for control group)+ signal features
def add_antib_features (df, antib_file,num_days_back):

    print ('start add_antib_features ')
    df.date_sample = pd.to_datetime(df.date_sample)

    antib_df  =  Utils.read_data( data_dir +antib_file)

    dict_groups = dict()
    dict_groups ['highly broad spectrum'] = ['Ceftazidime','Piperacillin / Tazobactam(Zosyn)','Aztreonam', 'Imipenem / Cilastatin','Meropenem', 'Cefepime']
    dict_groups['broad spectrum antibiotics'] = ['Levofloxacin','Ciprofloxacin','Moxifloxacin','Ampicillin / Sulbactam(Unasyn)', 'Ceftriaxone',  'Bactrim(SMX / TMP)' ]
    dict_groups ['not broad spectrum antibiotics'] = ['Doxycycline','Metronidazole','Cefazolin','Azithromycin','Amikacin','Ethambutol','Vancomycin','Clindamycin','Linezolid','Gentamicin','Oxacillin','Erythromycin','Rifampin','Penicillin G potassium','Nafcillin','Ampicillin','Keflex','Colistin']
    dict_groups ['all antibiotics'] =dict_groups ['highly broad spectrum']+dict_groups['broad spectrum antibiotics']+dict_groups ['not broad spectrum antibiotics']
    dict_groups['antifubgals'] = ['Micafungin','Ambisome','Voriconazole','Caspofungin','Fluconazole']

    df_plus_new_features = df.copy()

    for key, values_list in dict_groups.items():
        filtered_antib_df = antib_df[antib_df.order_name.isin(values_list)]

        print ('num unique orders:'+ str(filtered_antib_df.order_name.nunique()))
        print('num values in list :' + str(len(values_list)))

        df_plus_new_features= Handle_promateus_data.add_summary_data_from_commands(df_plus_new_features, command_df_in=filtered_antib_df,
                                              num_days_back=num_days_back,
                                              parameter_name=key)
    return df_plus_new_features


# from icu admission time
def add_catheter_features (df, catheter_file,num_days_back):
    print ('start add_catheter_features ')

    catheter_df  =  Utils.read_data(data_dir +catheter_file)
    df.date_sample = pd.to_datetime(df.date_sample)

    dict_groups = dict()


    dict_groups['AL'] = ['Arterial Line']
    dict_groups['PA'] = ['AVA Line','CCO PAC','PA Catheter']
    dict_groups['iabp_impella'] = ['IABP line','Impella Line']
    dict_groups['other_cath'] = ['ICP Catheter','Multi Lumen','Presep Catheter','Sheath','Trauma line','Triple Introducer','Tunneled (Hickman) Line']
    dict_groups['picc'] = ['PICC Line','Midline','Indwelling Port (PortaCath)']
    dict_groups['dial_line'] = ['Dialysis Catheter','Pheresis Catheter']

    df_plus_new_features = df.copy()

    for key, values_list in dict_groups.items():
        filtered_catheter_df = catheter_df[catheter_df.order_name.isin(values_list)]
        df_plus_new_features= Handle_promateus_data.add_summary_data_from_commands(df_plus_new_features, command_df_in=filtered_catheter_df,
                                              num_days_back=num_days_back,
                                              parameter_name=key)
    return df_plus_new_features

# add only indication of is or is not bedsores in num_days_back
def add_bedsores_features (control_df,
                           bedsores_file=data_dir + 'Skin_data_for_modeling.csv',
                                num_days_back=5):

    print ('start add_bedsores_features')
    bedsores_df = Utils.read_data( bedsores_file)

    bedsores_df = bedsores_df.rename(columns = {'SUBJECT_ID':'patient_id', 'CHARTTIME':'time','VALUE':'value' })
    bedsores_df = bedsores_df[(bedsores_df.value >0)]


    df_join  = pd.merge(bedsores_df,control_df[['patient_id','date_sample']],how = 'inner',on = ['patient_id'])
    df_join.date_sample = pd.to_datetime(df_join.date_sample)
    df_join.time = pd.to_datetime(df_join.time)
    control_df.date_sample = pd.to_datetime(control_df.date_sample)

    filter_df_join = df_join.loc[(df_join.time <= df_join.date_sample) & (
            df_join.date_sample.subtract(df_join.time) < pd.to_timedelta(num_days_back+1, unit='d')), :]


    group_functions = {'value': {'value':'count'}}
    bedsores_df_grp = filter_df_join.groupby(['patient_id','date_sample']).agg(group_functions)
    bedsores_df_grp.columns = bedsores_df_grp.columns.droplevel(0)
    bedsores_df_grp.reset_index(level=0, inplace=True)
    bedsores_df_grp.reset_index(level=0, inplace=True)

    bedsores_df_grp['is_bedsores_'+str(num_days_back)] = 0
    bedsores_df_grp.loc[
        bedsores_df_grp['value'] >0, 'is_bedsores_'+str(num_days_back)] = 1

    merge_with_control_df =  pd.merge(control_df, bedsores_df_grp[['patient_id','date_sample','is_bedsores_'+str(num_days_back)]], how='left', on=['patient_id', 'date_sample'])
    merge_with_control_df['is_bedsores_'+str(num_days_back)] = merge_with_control_df['is_bedsores_'+str(num_days_back)].fillna(0)
    return merge_with_control_df

def  build_dataset (control_df,num_days_from_last_pos,num_days_in_icu):

    print ('start mimic - build_dataset')
    stool_file = output_dir+'Num_stools_with_0_days_100.csv'
    antib_file = 'mimic_anti_data.csv'
    catheter_file = 'mimic_catheter_data.csv'
    tpn_file = 'mimic_tpn_data.csv'
    bedsores_file = data_dir+'Skin_data_for_modeling.csv'


    join_df =control_df
    join_df.date_sample = pd.to_datetime(join_df.date_sample)

    join_df_clean = join_df
    #for married - only take relevant from signal dir, no anti...

    # # --- add catheter features
    join_df_clean = add_catheter_features(join_df_clean, catheter_file, num_days_back=14)
    join_df_clean = add_catheter_features(join_df_clean, catheter_file, num_days_back=7)
    # join_df_clean.to_csv(output_dir + 'tmp_catheter_plus.csv')

    join_df_clean = add_antib_features(join_df_clean, antib_file=antib_file, num_days_back=14)
    # join_df_clean.to_csv(output_dir+'tmp_anti_plus.csv')
    #


    # add tpn features
    join_df_clean = Handle_promateus_data.add_tpn_features(join_df_clean, tpn_file=tpn_file,num_days_back=7)
    # join_df_clean.to_csv(output_dir + 'tmp_tpn_plus.csv')

    join_df_clean = add_bedsores_features(join_df_clean, bedsores_file=bedsores_file, num_days_back=3)
    #
    #  no stool , count stool... bad data
    join_df_clean = Handle_promateus_data.add_stool_featurs(join_df_clean, stool_file=stool_file, num_days_back=5)
    join_df_clean = Handle_promateus_data.add_stool_featurs(join_df_clean, stool_file=stool_file, num_days_back=3)
    join_df_clean = Handle_promateus_data.add_stool_featurs(join_df_clean, stool_file=stool_file, num_days_back=7)

    # join_df_clean.to_csv(output_dir + 'tmp_stool_plus.csv')

    signal_path = output_dir + 'signals//'
    print ('searrch signals in  '+ signal_path)
    files = os.listdir(signal_path)
    # tmp todo
    for name in files:
        # print(path+name)
        join_df_clean = Handle_promateus_data.add_signal_features(join_df_clean,
                                            signal_file=signal_path + name)

        # print (list(join_df_clean.keys()))

    # ---todo -------------



    join_df_clean.to_csv(output_dir + 'all_features_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv')

    # join_df_clean = pd.read_csv(output_dir+'all_features_20181219163929.csv', encoding="ISO-8859-1")

    #add indication of death in next 30 days from sample
    join_df_clean[ 'death_in_a_month'] = 0
    mask = pd.to_datetime(join_df_clean.deceaseddate).subtract(pd.to_datetime(join_df_clean.date_sample))<=pd.to_timedelta(30,'d')
    join_df_clean.loc[mask, 'death_in_a_month'] = 1

    join_df_clean= Handle_promateus_data.try_to_remove_columns(join_df_clean)

    join_df_clean = Handle_promateus_data.clean_columns(join_df_clean, Handle_promateus_data.label_col_name)
    # print(list(join_df_clean.keys()))
    # -------todo-------
    # add calculated features - change it to relevant features in mimic
    # join_df_clean = add_calculated_features(join_df_clean)
    #---------------------

    join_df_clean = join_df_clean.sample(frac=1).reset_index(drop=True)
    join_df_clean['label']= join_df_clean['label'].astype(np.int)

    join_df_clean.to_csv(output_dir + 'all_features_cleaned_'+str(num_days_in_icu)+'_days_in_icu_'+str(num_days_from_last_pos)+'_days_after_pos_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv')

    train, test = train_test_split(join_df_clean, test_size=0.1)
    train_file = output_dir + 'all_features_cleaned_train_' +str(num_days_in_icu)+'_days_in_icu_'+str(num_days_from_last_pos)+'_days_after_pos_'+ datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
    test_file = output_dir + 'all_features_cleaned_test_' +str(num_days_in_icu)+'_days_in_icu_'+str(num_days_from_last_pos)+'_days_after_pos_'+ datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
    train.to_csv(train_file)
    test.to_csv(test_file)

    print ('----finish dataset creation-----')
    print(train_file)
    print (test_file)
    return join_df_clean

    return all_features_cleaned_file
    pass

def create_data_set(num_days_from_last_pos=4,num_days_in_icu=3,
                    num_days_list_for_signal_features=[3, 5],type ='bactaremia'):

    print ('start mimic - create_data_set ')

    #for bactaremia
    handle_signal_data_new.output_dir =  'C://Users//michael//Desktop//predicting BSI//files//mimic//Output_'+str(num_days_in_icu)+'//'
    # for married
    # handle_signal_data_new.output_dir = 'C://Users//michael//Desktop//predicting BSI//files//mimic//Output_married_' + str(num_days_in_icu) + '//'

    global output_dir
    output_dir= handle_signal_data_new.output_dir
    create_daily_data.output_dir = handle_signal_data_new.output_dir
    create_liquids_file.output_dir = handle_signal_data_new.output_dir
    mimic.create_control_set.output_dir =handle_signal_data_new.output_dir
    mimic.create_signal_data.output_dir = handle_signal_data_new.output_dir
    mimic.prepare_data_files.output_dir = handle_signal_data_new.output_dir
    create_vasoactive_data.output_dir = handle_signal_data_new.output_dir
    Handle_promateus_data.output_dir = handle_signal_data_new.output_dir

    global data_dir
    create_vasoactive_data.data_dir = data_dir
    handle_signal_data_new.data_dir = data_dir
    create_liquids_file.data_dir = data_dir
    Handle_promateus_data.data_dir = data_dir




    control_file = output_dir+type+'_'+'control_file_'+str(num_days_in_icu)+'_days_in_icu_'+str(num_days_from_last_pos)+'_days_after_pos.csv'

    if type =='bactaremia':
        control_df,control_file = mimic.create_control_set.create_control_set_bactaremia( control_file,num_days_from_last_pos=num_days_from_last_pos,
                                                 num_days_in_icu=num_days_in_icu)

    elif type == 'bactaremia_first_pos_neg':
        control_df, control_file = mimic.create_control_set.create_control_set_bactaremia_first_pos_neg(control_file,
                                                                                              num_days_from_last_pos=num_days_from_last_pos,
                                                                                              num_days_in_icu=num_days_in_icu)

    # elif type == 'married':
    #     control_df,control_file = mimic.create_control_set.create_control_set_married( control_file,num_days_in_icu=num_days_in_icu)
    # else:
    #     print ('insert type is not valid ' + type)


    mimic.create_signal_data.create_signal_data(control_file=control_file, num_days_list=num_days_list_for_signal_features)

    control_df = pd.read_csv(control_file, encoding="ISO-8859-1")
    join_df_clean = build_dataset(control_df,num_days_from_last_pos=num_days_from_last_pos,num_days_in_icu=num_days_in_icu)


def run_modeling (all_features_cleaned_file):


    my_modeling.run_cv(data_train_file=all_features_cleaned_file,
           calculate_features=False, num_features_per_model=20)

if __name__ == "__main__":

    #for bactaremia - the paper data 18.8.19
    # create_data_set(num_days_from_last_pos=4, num_days_in_icu=2, num_days_list_for_signal_features=[3,5])

    #for bactaremia - first pos + first neg
    create_data_set(num_days_from_last_pos=4, num_days_in_icu=2, num_days_list_for_signal_features=[3,5],type='bactaremia_first_pos_neg')


    # for married
    # create_data_set(num_days_from_last_pos=-1, #not in use
    #                 num_days_in_icu=2,
    #                 num_days_list_for_signal_features=[3],type = 'married')

    #for stool
    # create_data_set(num_days_from_last_pos=4, num_days_in_icu=5, num_days_list_for_signal_features=[3])

    # all_features_cleaned_file = build_dataset()
    #
    # all_features_cleaned_file =output_dir +'all_features_cleaned_20180926202605.csv'
    # print ('features file :' +all_features_cleaned_file)
    # run_modeling(all_features_cleaned_file)