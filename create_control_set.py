import Handle_promateus_data
import Utils
import pandas as pd
import sqlite3



output_dir = "C://Users//michael//Desktop//predicting BSI//files//mimic//Output//"
data_dir = 'C://Users//michael//Desktop//predicting BSI//files//mimic//Data//'
sample_file_name = data_dir +'MICROBIOLOGYEVENTS.csv.gz'
admission_file_name  = data_dir +'ADMISSIONS.csv.gz'
d_items_file_name  = data_dir +'D_ITEMS.csv.gz'
icustays_file_name  = data_dir +'ICUSTAYS.csv.gz'
patients_file_name = data_dir + 'PATIENTS.csv.gz'
labs_file_name =  data_dir + 'LABEVENTS.csv.gz'
outputevents_file_name = data_dir + 'OUTPUTEVENTS.csv.gz'
inputevents_mv_file_name = data_dir + 'INPUTEVENTS_MV.csv.gz'
d_labitems_file_name = data_dir + 'D_LABITEMS.csv.gz'
procedureevents_mv_file_name = data_dir+'PROCEDUREEVENTS_MV.csv.gz'
diagnoses_icd_file_name = data_dir+'DIAGNOSES_ICD.csv.gz'
d_icd_diagnoses_file_name = data_dir+ 'D_ICD_DIAGNOSES.csv.gz'

# ---------declartions for modeling------------


num_days_to_take_from_lab = 5
num_days_to_take_from_anti = 14
#---------------------------------------------

#for reviewers include cons

def prepare_sample_file_including_cons (sample_file_name ,outputfile,
                         bactaremia_types_file = data_dir+ 'mimic_bactaremia_types_for_modeling.csv' ):

    bact_df = pd.read_csv(bactaremia_types_file)
    MICROBIOLOGYEVENTS = pd.read_csv(sample_file_name
                                     # , nrows=100
                                     , compression='gzip',
                                     error_bad_lines=False)


    print (MICROBIOLOGYEVENTS.shape)

    # Make the db in memory
    conn = sqlite3.connect(':memory:')
    # write the tables
    MICROBIOLOGYEVENTS.to_sql('MICROBIOLOGYEVENTS', conn, index=False)
    bact_df.to_sql('bact_df', conn, index=False)

    # todo - need to update battaremia types - michael
    # get relevant bactaremia from blood culture- types selected ( in 'blood' no records)
    # for review count no need to group by
    # qry = ''' select SUBJECT_ID,HADM_ID,case when CHARTTIME is null then CHARTDATE else CHARTTIME end as CHARTTIME ,SPEC_ITEMID,SPEC_TYPE_DESC,
    #         max(case when org_name is null  then 0 else 1 end) as label,group_concat(org_name, ', ') org_name,
    #         case when ORG_NAME is null then '0' when ORG_NAME in (select org_name from bact_df) then '1' else 'cons' end label_details
    #           from
    #         (select distinct SUBJECT_ID,HADM_ID,CHARTDATE,CHARTTIME,SPEC_ITEMID,org_name,SPEC_TYPE_DESC from MICROBIOLOGYEVENTS) a
    #         where a.spec_type_desc in ( 'BLOOD CULTURE','BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)')
    #         group by SUBJECT_ID,HADM_ID,
    #         case when CHARTTIME is null then CHARTDATE else CHARTTIME end ,
    #         SPEC_ITEMID,SPEC_TYPE_DESC
    #     '''

    qry = ''' select SUBJECT_ID,HADM_ID,
            case when CHARTTIME is null then CHARTDATE else CHARTTIME end as CHARTTIME ,
            SPEC_ITEMID,SPEC_TYPE_DESC,org_name,
            case when ORG_NAME is null then '0' when ORG_NAME in (select org_name from bact_df) then '1' else 'cons' end label_details
              from
            (select distinct SUBJECT_ID,HADM_ID,CHARTDATE,CHARTTIME,SPEC_ITEMID,org_name,SPEC_TYPE_DESC from MICROBIOLOGYEVENTS) a
            where a.spec_type_desc in ( 'BLOOD CULTURE','BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)')
        '''

    print (qry)
    MICROBIOLOGYEVENTS_with_type = pd.read_sql_query(qry, conn)

    print(MICROBIOLOGYEVENTS_with_type.head())
    MICROBIOLOGYEVENTS_with_type.to_csv(outputfile)
    return MICROBIOLOGYEVENTS_with_type

# gets relevant samples of bactaremia in bactaremia_types_file
def prepare_sample_file (sample_file_name ,outputfile,
                         bactaremia_types_file = data_dir+ 'mimic_bactaremia_types_for_modeling.csv' ):

    bact_df = pd.read_csv(bactaremia_types_file)
    MICROBIOLOGYEVENTS = pd.read_csv(sample_file_name
                                     # , nrows=100
                                     , compression='gzip',
                                     error_bad_lines=False)


    print (MICROBIOLOGYEVENTS.shape)

    # Make the db in memory
    conn = sqlite3.connect(':memory:')
    # write the tables
    MICROBIOLOGYEVENTS.to_sql('MICROBIOLOGYEVENTS', conn, index=False)
    bact_df.to_sql('bact_df', conn, index=False)

    # todo - need to update battaremia types - michael
    # get relevant bactaremia from blood culture- types selected ( in 'blood' no records)
    qry = ''' select SUBJECT_ID,HADM_ID,case when CHARTTIME is null then CHARTDATE else CHARTTIME end as CHARTTIME ,SPEC_ITEMID,SPEC_TYPE_DESC,
            max(case when org_name is null  then 0 else 1 end) as label,group_concat(org_name, ', ') org_name
              from
            MICROBIOLOGYEVENTS a
            where a.spec_type_desc in ( 'BLOOD CULTURE','BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)')
            and ( ORG_NAME is null or ORG_NAME in (
           select org_name from bact_df)) 
            group by SUBJECT_ID,HADM_ID,
            case when CHARTTIME is null then CHARTDATE else CHARTTIME end ,
            SPEC_ITEMID,SPEC_TYPE_DESC
        '''

    print (qry)
    MICROBIOLOGYEVENTS_with_type = pd.read_sql_query(qry, conn)

    print(MICROBIOLOGYEVENTS_with_type.head())
    MICROBIOLOGYEVENTS_with_type.to_csv(outputfile)
    return MICROBIOLOGYEVENTS_with_type

#takes admissions in the icu (micu,tsicu,sicu) of 2 days or more
def prepare_admissions_file(admission_file_name ,icustays_file_name ,outputfile='icustays_for_modeling.csv'):

    ADMISSIONS = pd.read_csv(admission_file_name
                             , compression='gzip',
                             error_bad_lines=False)

    icustays = pd.read_csv(icustays_file_name
                           , compression='gzip',
                           error_bad_lines=False)

    print (ADMISSIONS.shape)
    print(icustays.shape)

    # Make the db in memory
    conn = sqlite3.connect(':memory:')
    # write the tables
    ADMISSIONS.to_sql('ADMISSIONS', conn, index=False)
    icustays.to_sql('icustays', conn, index=False)

    # Notice : filter out short than 2 days in ICU!
    qry = '''
    select intime,outtime, icustay_id,first_careunit,los,b.*  
    from icustays a, admissions b
    where first_careunit in ('MICU','SICU','TSICU')
    and a.hadm_id = b.hadm_id
    and dbsource = 'metavision'
    and a.los>2
    '''

    print (qry)
    icustays_df = pd.read_sql_query(qry, conn)

    # print(icustays_df.head())
    icustays_df.to_csv(outputfile)
    return icustays_df

def join_icustays_sample (sample_df ,icustays_df, outputfile ='sample_icustays_for_modeling1.csv'
                         ):  # Make the db in memory
    conn = sqlite3.connect(':memory:')
    # write the tables
    sample_df.to_sql('sample_df', conn, index=False)

    icustays_df = icustays_df.loc[:, ~icustays_df.columns.isin(['ROW_ID','SUBJECT_ID','Unnamed: 0'])]
    icustays_df.to_sql('icustays_df', conn, index=False)

    print (list(icustays_df.keys()))
    print(list(sample_df.keys()))
    qry = '''
    select a.*,admittime,dischtime,deathtime ,admission_type,admission_location,language, religion,marital_status,ethnicity,edregtime,edouttime,diagnosis,HAS_CHARTEVENTS_DATA,intime,outtime,first_careunit
    from sample_df a, icustays_df b  
    where a.hadm_id = b.hadm_id
    '''

    print(qry)
    join_df = pd.read_sql_query(qry, conn)

    join_df =join_df.loc[:, join_df.columns != 'ROW_ID']

    # print(join_df.head())
    join_df.to_csv(outputfile)
    return join_df


def join_icustays_patients(join_icustays_sample_df, patients_file_name, outputfile):
    PATIENTS = pd.read_csv(patients_file_name, compression='gzip',
                           error_bad_lines=False)

    print(PATIENTS.shape)

    conn = sqlite3.connect(':memory:')
    # write the tables
    join_icustays_sample_df = join_icustays_sample_df.loc[:, ~join_icustays_sample_df.columns.isin(['ROW_ID', 'Unnamed: 0'])]
    join_icustays_sample_df.to_sql('join_icustays_sample_df', conn, index=False)
    PATIENTS.to_sql('PATIENTS', conn, index=False)


    # 5/6/2019 - was charttime-dob as age
    qry = '''
    select a.*,ADMITTIME-dob as age,b.gender, b.dod
     from join_icustays_sample_df a, PATIENTS b
    where a.subject_id = b.subject_id
    '''

    print(qry)
    join_df = pd.read_sql_query(qry, conn)
    join_df.to_csv(outputfile)
    return join_df

def create_control_file_2_days_married (outputfile):

    icustays_outputfile = data_dir + 'married_icustays_for_modeling.csv'

    # take icu admissions + los > 2 days
    icustays_df = prepare_admissions_file(admission_file_name, icustays_file_name, outputfile=icustays_outputfile)
    icustays_df = pd.read_csv(icustays_outputfile)

    # get sample after more than 2 days in icu


    # add patient data - age + gender

    join_icustays_sample_patients_df = join_icustays_patients(join_icustays_sample_df=icustays_df,
                                                              patients_file_name = patients_file_name,
                                                              outputfile=outputfile)
    # join_icustays_sample_patients_df = pd.read_csv(join_icustays_patients_outputfile)

    return join_icustays_sample_patients_df


def create_control_file_bactaremia(join_icustays_patients_outputfile):

    # output files files
    sample_outputfile = output_dir+'MICROBIOLOGYEVENTS_for_modeling1.csv'
    icustays_outputfile = data_dir+'icustays_for_modeling.csv'
    join_icustays_sample_outputfile = output_dir+'sample_icustays_for_modeling1.csv'


    # take only blood with bactermia or null
    sample_df = prepare_sample_file(sample_file_name, outputfile=sample_outputfile);
    # sample_df = pd.read_csv(sample_outputfile)

    # take icu admissions + los > 2 days

    icustays_df = prepare_admissions_file(admission_file_name, icustays_file_name, outputfile=icustays_outputfile)
    # icustays_df = pd.read_csv(icustays_outputfile)

    # get sample after more than 2 days in icu

    join_icustays_sample_df = join_icustays_sample(sample_df, icustays_df, outputfile=join_icustays_sample_outputfile)
    # join_icustays_sample_df = pd.read_csv(join_icustays_sample_outputfile)

    # add patient data - age + gender

    join_icustays_sample_patients_df = join_icustays_patients(join_icustays_sample_df, patients_file_name,
                                                              outputfile=join_icustays_patients_outputfile)
    # join_icustays_sample_patients_df = pd.read_csv(join_icustays_patients_outputfile)

    return join_icustays_sample_patients_df


def prepare_table(join_df, num_days=2):
    join_df.CHARTTIME = pd.to_datetime(join_df.CHARTTIME)
    join_df.INTIME = pd.to_datetime(join_df.INTIME)
    join_df.OUTTIME = pd.to_datetime(join_df.OUTTIME)
    join_df.ADMITTIME = pd.to_datetime(join_df.ADMITTIME)

    # filter out less than 2 days in icu
    join_df['sample_minus_start_icu'] = join_df['CHARTTIME'].subtract(join_df.INTIME)
    join_df = join_df[join_df['sample_minus_start_icu'] > pd.Timedelta(num_days, unit='d')]
    join_df['sampe_start_hospital'] = join_df['CHARTTIME'].subtract(join_df.ADMITTIME)
    join_df = join_df[join_df['sampe_start_hospital'] > pd.Timedelta(num_days, unit='d')]
    join_df['end_icu_sampe'] = join_df['OUTTIME'].subtract(join_df['CHARTTIME'])

    join_df = join_df.rename(columns={'INTIME': 'admission start day icu',
                                      'ADMITTIME': 'ADMISSION START DAY HOSPITAL',
                                      'CHARTTIME': 'date_sample', 'GENDER': 'gender',
                                      'SUBJECT_ID': 'patient_id','DOD':'deceaseddate',
                                      'OUTTIME':'admission end day ICU'})
    print(join_df.keys())
    return join_df
    # [
    #     ['HADM_ID', 'patient_id', 'date_sample', 'SPEC_ITEMID', 'SPEC_TYPE_DESC', 'label', 'ICUSTAY_ID', 'DBSOURCE',\
    #      'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID',\
    #      'admission start day icu', 'LOS',\
    #      'ADMISSION START DAY HOSPITAL', 'DISCHTIME', 'DEATHTIME',\
    #      'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION',\
    #      'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY',\
    #      'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS',\
    #      'HAS_CHARTEVENTS_DATA', 'age', 'gender', 'sample_minus_start_icu',\
    #      'sampe_start_hospital', 'end_icu_sampe']]


def save_data_file_to_csv(file_name,nrows = 'all'):
    data = pd.read_csv(file_name, compression='gzip',
                       nrows=nrows,
                       error_bad_lines=False)

    print(data.shape)
    print ('saved data to '+ file_name + '.csv')
    data.to_csv(file_name + '_'+str(nrows)+'.csv')

def create_control_set_married(control_file_name,num_days_in_icu = 2 ):

    join_icustays_patients_outputfile = output_dir+'icustays_patients_for_married_modeling.csv'

    # ---join all tables to create all information needed to create control file
    # join_icustays_patients_df = create_control_file_2_days_married(join_icustays_patients_outputfile)
    join_icustays_patients_df = Utils.read_data( join_icustays_patients_outputfile)

    join_icustays_patients_df['ADMITTIME'] = pd.to_datetime(join_icustays_patients_df['ADMITTIME'])
    join_icustays_patients_df['DEATHTIME'] = pd.to_datetime(join_icustays_patients_df['DEATHTIME'])

    # I add 1 so the prepare_table will not filter out all rows
    join_icustays_patients_df['CHARTTIME'] = join_icustays_patients_df['ADMITTIME']+pd.Timedelta(num_days_in_icu+1, unit='d')
    # add 1 so data will be saved for the first day
    join_icustays_patients_df['date_sample'] = join_icustays_patients_df['ADMITTIME'] +pd.Timedelta(num_days_in_icu+1, unit='d')
    join_icustays_patients_df ['days_death_admit'] = join_icustays_patients_df['DEATHTIME'].subtract (join_icustays_patients_df['ADMITTIME'])

    # add calculated features of dates - filter out less than num_days in icu
    join_icustays_patients_df = prepare_table(join_icustays_patients_df,
                                                     num_days = num_days_in_icu)

    join_icustays_patients_df[ 'death_in_a_month_from_admit'] = 0
    mask = pd.to_datetime(join_icustays_patients_df.deceaseddate).subtract(pd.to_datetime(join_icustays_patients_df['admission start day icu']))<=pd.to_timedelta(30,'d')
    join_icustays_patients_df.loc[mask, 'death_in_a_month_from_admit'] = 1


    join_icustays_patients_df.to_csv(control_file_name)
    print ('created '+control_file_name)

    # model_dataset_clean_df = pd.read_csv(data_set_for_modeling_file)

    # to create csv from zip file
    # save_data_file_to_csv(d_items_file_name)

    return join_icustays_patients_df,control_file_name


def create_control_set_bactaremia_first_pos_neg(control_file_name,num_days_in_icu = 2,num_days_from_last_pos = 3 ):

    join_icustays_patients_outputfile = output_dir+'sample_icustays_patients_for_modeling.csv'

    # ---join all tables to create all information needed to create control file
    join_icustays_sample_patients_df = create_control_file_bactaremia(join_icustays_patients_outputfile)
    join_icustays_sample_patients_df = Utils.read_data( join_icustays_patients_outputfile)

    # add calculated features of dates - filter out less than num_days in icu
    join_icustays_sample_patients_df = prepare_table(join_icustays_sample_patients_df,
                                                     num_days = num_days_in_icu)
    # print (list(join_icustays_sample_patients_df.keys()))
    #

    model_dataset_df = Handle_promateus_data.clean_rows_first_pos_neg(df=join_icustays_sample_patients_df, label_col_name='label',
                                                        positive_value=1, negative_value=0,
                                                        num_days_from_last_pos = num_days_from_last_pos, num_days_in_icu = num_days_in_icu)



    model_dataset_df.to_csv(control_file_name)
    print ('created '+control_file_name)


    return model_dataset_df,control_file_name

def create_control_set_bactaremia(control_file_name,num_days_in_icu = 2,num_days_from_last_pos = 3 ):

    join_icustays_patients_outputfile = output_dir+'sample_icustays_patients_for_modeling.csv'

    # ---join all tables to create all information needed to create control file
    join_icustays_sample_patients_df = create_control_file_bactaremia(join_icustays_patients_outputfile)
    join_icustays_sample_patients_df = Utils.read_data( join_icustays_patients_outputfile)

    # add calculated features of dates - filter out less than num_days in icu
    join_icustays_sample_patients_df = prepare_table(join_icustays_sample_patients_df,
                                                     num_days = num_days_in_icu)
    # print (list(join_icustays_sample_patients_df.keys()))
    #

    model_dataset_df = Handle_promateus_data.clean_rows(df=join_icustays_sample_patients_df, label_col_name='label',
                                                        positive_value=1, negative_value=0,
                                                        num_days_from_last_pos = num_days_from_last_pos, num_days_in_icu = num_days_in_icu)

    # model_dataset_df.to_csv(output_dir + 'dataset.csv')
    # model_dataset_df = pd.read_csv(output_dir + 'dataset.csv')
    # model_dataset_clean_df = model_dataset_df[
    #     ['HADM_ID', 'ADMISSION_TYPE', 'ETHNICITY', 'FIRST_CAREUNIT', 'FIRST_WARDID', 'LAST_CAREUNIT',
    #      'MARITAL_STATUS', 'RELIGION', 'age', 'date_sample', 'delta_from_prev_pos', 'gender', 'label',
    #      'patient_id', 'sampe_start_hospital', 'admission start day icu', 'sample_minus_start_icu']]


    model_dataset_df.to_csv(control_file_name)
    print ('created '+control_file_name)

    # model_dataset_clean_df = pd.read_csv(data_set_for_modeling_file)

    # to create csv from zip file
    # save_data_file_to_csv(d_items_file_name)

    return model_dataset_df,control_file_name


    # remove all unnamed features
    # prepare signal
    # prepare stool
    # prepare liquid
    # prepare medicine - antibutic
    # prpare catheter
    # prepare tpn
    # prepare labs
    # prepare FIOS
    # previous illness

if __name__ == '__main__':
    # to create csv from zip file
    save_data_file_to_csv(d_items_file_name)

    # create_control_set()