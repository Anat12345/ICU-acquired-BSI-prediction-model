import Handle_promateus_data
import Utils
import create_liquids_file
import create_vasoactive_data
import handle_signal_data_new
import mimic.Chartevent_param_data as ChartEventParam
import os
import pandas as pd
import sqlite3
import mimic.create_control_set

output_dir = "C://Users//michael//Desktop//predicting BSI//files//mimic//Output//"
data_dir = 'C://Users//michael//Desktop//predicting BSI//files//mimic//Data//'


#prepare blood labs data
def create_labs_files(model_dataset_df, labs_file_name, d_labitems_file_name, num_days=5,
                      outputfile=data_dir+'blood_data_for_modeling.csv'):
    print('started create_labs_files')

    # reads labs data

    d_labitems = pd.read_csv(d_labitems_file_name, compression='gzip',
                             # nrows=100000,
                             error_bad_lines=False)

    print(d_labitems.shape)

    LABEVENTS = pd.read_csv(labs_file_name, compression='gzip',
                            # nrows=100000, #todo
                            error_bad_lines=False)

    print(LABEVENTS.shape)


    # get data of only control file hadm_id
    lab_df1 = LABEVENTS[
        LABEVENTS[['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE']].HADM_ID.isin(model_dataset_df.HADM_ID.unique())]
    print('lab_df1 - only hadm_id in control ' + str(lab_df1.shape))

    # add data of sample date to cut over few days back
    lab_df2 = pd.merge(lab_df1, model_dataset_df, how='inner', on=['HADM_ID'])
    # print('lab_df2 - join control data : ' + str(lab_df2.shape))
    lab_df2.CHARTTIME = pd.to_datetime(lab_df2.CHARTTIME)
    lab_df2.date_sample = pd.to_datetime(lab_df2.date_sample)
    lab_df2['admission start day icu'] = pd.to_datetime(lab_df2['admission start day icu'])

    lab_df2['date_sample_minus_charttime'] = lab_df2.date_sample.subtract(lab_df2.CHARTTIME)
    lab_df2['charttime_minus_start_icu'] = lab_df2.CHARTTIME.subtract(lab_df2['admission start day icu'])

    lab_df2 = lab_df2[(((lab_df2.date_sample_minus_charttime <= pd.Timedelta(num_days, unit='d'))&
                       (lab_df2.date_sample_minus_charttime >= pd.Timedelta(0, unit='d'))))
                    | ((lab_df2['charttime_minus_start_icu'] <= pd.Timedelta(1.5, unit='d')) &
                        (lab_df2['charttime_minus_start_icu'] >= pd.Timedelta(-0.5, unit='d')))]
    lab_df3 = lab_df2[
        ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'date_sample', 'date_sample_minus_charttime']]
    print('lab_df3  - filter out charttime: ' + str(lab_df3.shape))

    # add parametr name for itemid code
    lab_df4 = pd.merge(lab_df3, d_labitems[d_labitems.FLUID == 'Blood'][['LABEL', 'ITEMID', 'FLUID', 'CATEGORY']],
                       how='inner', on=['ITEMID'])
    print('lab_df4 - add parameter_name ' + str(lab_df4.shape))

    # rename columns
    lab_df4 = lab_df4.rename(
        columns={'LABEL': 'parameter_name', 'CHARTTIME': 'time', 'VALUE': 'value', 'SUBJECT_ID': 'patient_id'})
    lab_df4.to_csv(outputfile)

    print(' created ' + outputfile)

    # -----for calculating stat about frequency of types in control file
    # group_functions = {'HADM_ID':{'num_hadm_id':'nunique'},'SUBJECT_ID':{'num_subject_id':'nunique'}}
    # lab_df4 = lab_df3.groupby (['ITEMID','LABEL','FLUID','CATEGORY']).agg(group_functions)

def  create_stool_data(outputevents_file_name):

    OUTPUTEVENTS = pd.read_csv(outputevents_file_name, compression='gzip',
                             # nrows=100000,
                             error_bad_lines=False)

    print(OUTPUTEVENTS.shape)

    #convert file to rambam stool format
    # Make the db in memory
    conn = sqlite3.connect(':memory:')
    # write the tables
    OUTPUTEVENTS.to_sql('OUTPUTEVENTS', conn, index=False)


    '''
    item_id to be included:
    226574 -     Jejunostomy
    226579 -     Stool
    226580-     Fecal Bag
    226582-     Ostomy(output)
    226583-     Rectal Tube
    226586 -     Stool stimate
    '''

    qry = ''' select hadm_id ,icustay_id, subject_id as patient_id, 'stool' as parameter_name, charttime as time, case when value>20 then 1 else 0 end as value from outputevents a
              where itemid in (226574,226579,226580,226582,226583,226586)
        '''
    print (qry)
    stools_df = pd.read_sql_query(qry, conn)

    stools_df.to_csv(data_dir+'mimic_stools.csv')

def create_anti_data (inputevents_mv_file_name,d_items_file_name,
                      outputfile=data_dir + 'mimic_anti_data.csv'):
    
    
    # Make the db in memory
    conn = sqlite3.connect(':memory:')

    d_items = pd.read_csv(d_items_file_name, compression='gzip',
                             # nrows=100000,
                             error_bad_lines=False)
    d_items.to_sql('D_ITEMS', conn, index=False)
    print(d_items.shape)

    INPUTEVENTS_MV = pd.read_csv(inputevents_mv_file_name, compression='gzip',
                            # nrows=1000000, #todo
                            error_bad_lines=False)

    print(INPUTEVENTS_MV.shape)
    # write the tables
    INPUTEVENTS_MV.to_sql('INPUTEVENTS_MV', conn, index=False)

    # sometimes end_time< start_time and amount is negative, it mix the amount - changed it to originalamount
    qry = '''select subject_id  as patient_id, hadm_id,icustay_id, starttime as start_date, endtime as end_date,
                    label as order_name, originalamount as amount, amountuom,rate,patientweight 
            from INPUTEVENTS_MV a, D_ITEMS b
            where lower(ordercategoryname) like '%anti%' 
            and a.ITEMID = b.ITEMID
            and originalamount> 0'''
    print (qry)
    anti_df = pd.read_sql_query(qry, conn)
    anti_df = fix_start_date(anti_df)
    anti_df.to_csv(outputfile)

    print(' created ' + outputfile)

def fix_start_date(df):
        mask = (df.start_date > df.end_date)
        df['tmp_start_date'] = df.start_date
        df.loc[mask,'start_date'] = df.loc[mask,'end_date']
        df.loc[mask, 'end_date'] = df.loc[mask,'tmp_start_date']
        return df

def create_medications_data(inputevents_mv_file_name, d_items_file_name,
                            outputfile_pressor_sedatives=data_dir + 'mimic_medications_data.csv',
                            outputfile_fusid = data_dir + 'mimic_fusid_data.csv',
                            outputfile_tpn=data_dir + 'mimic_tpn_data.csv'):
    # todo - delete comment
    # Make the db in memory
    conn = sqlite3.connect(':memory:')

    d_items = pd.read_csv(d_items_file_name, compression='gzip',
                          # nrows=100000,
                          error_bad_lines=False)
    d_items.to_sql('d_items', conn, index=False)
    print(d_items.shape)

    print (' try read ' + inputevents_mv_file_name)
    inputevents_mv = pd.read_csv(inputevents_mv_file_name, compression='gzip',
                                     # nrows=100000,
                                     error_bad_lines=False)

    print(inputevents_mv.shape)
    # write the tables
    inputevents_mv.to_sql('inputevents_mv', conn, index=False)



    # ------------pressor_sedatives------
    qry = '''select subject_id as patient_id , hadm_id ,icustay_id, starttime as start_date,
            endtime as end_date,
            label as order_name,abs(COALESCE(COALESCE(rate,originalrate),amount)) as rate,amount
            from inputevents_mv a, D_ITEMS b
            where b.category  in ('Medications')
            and b.label in ('Epinephrine','Dopamine','Midazolam (Versed)','Fentanyl','Phenylephrine','Norepinephrine','Propofol','Vasopressin','Morphine Sulfate')
            and statusdescription not in ('Rewritten')
            and a.itemid = b.itemid'''
    print(qry)
    medications_df = pd.read_sql_query(qry, conn)
    medications_df = medications_df[medications_df.rate>0]
    medications_df = fix_start_date(medications_df)
    medications_df.to_csv(outputfile_pressor_sedatives)

    print(' created ' + outputfile_pressor_sedatives)
    create_vasoactive_data.create_vasoactive_data(outputfile_pressor_sedatives,
                           'mimic_medications_data_for_modeling.csv')

    #------------fusid
    qry = '''select subject_id as patient_id, hadm_id ,icustay_id, starttime as start_date, endtime as end_date,
            label , 'Fusid' as order_name,abs(COALESCE(COALESCE(rate,originalrate),amount)) as rate,amount
            from inputevents_mv a, D_ITEMS b
            where b.category  in ('Medications')
            and b.label in ('Furosemide (Lasix) 500/100','Furosemide (Lasix)')
            and statusdescription not in ('Rewritten')
            and a.itemid = b.itemid'''
    print(qry)
    fusid_df = pd.read_sql_query(qry, conn)
    fusid_df = fusid_df[fusid_df.rate>0]
    fusid_df = fix_start_date(fusid_df)
    fusid_df.to_csv(outputfile_fusid)

    create_vasoactive_data.create_vasoactive_data(outputfile_fusid,
                                                  'fusid_rate_per_hour_for_modeling.csv')



    print(' created ' + outputfile_fusid)

    # # ---------------tpn --------------
    qry = '''select subject_id as patient_id, hadm_id ,icustay_id, starttime as start_date, endtime as end_date,
            label as order_name,abs(COALESCE(COALESCE(rate,originalrate),amount)) as rate,amount
            from inputevents_mv a, D_ITEMS b
            where b.category  in ('Nutrition - Parenteral')
            and statusdescription not in ('Rewritten')
            and a.itemid = b.itemid'''
    print(qry)
    tpn_df = pd.read_sql_query(qry, conn)

    tpn_df = fix_start_date(tpn_df)
    tpn_df.to_csv(outputfile_tpn)


def create_catheter_data(procedureevents_mv_file_name, d_items_file_name,
                     outputfile=data_dir + 'mimic_catheter_data.csv'):
    # Make the db in memory
    conn = sqlite3.connect(':memory:')

    d_items = pd.read_csv(d_items_file_name, compression='gzip',
                          # nrows=100000,
                          error_bad_lines=False)
    d_items.to_sql('d_items', conn, index=False)
    print(d_items.shape)

    print (' try read '+procedureevents_mv_file_name)
    procedureevents_mv = pd.read_csv(procedureevents_mv_file_name, compression='gzip',
                                 # nrows=100000,
                                 error_bad_lines=False)

    print(procedureevents_mv.shape)
    # write the tables
    procedureevents_mv.to_sql('procedureevents_mv', conn, index=False)

    qry = '''select subject_id as patient_id, hadm_id ,icustay_id, starttime as start_date, endtime as end_date, 
            label as order_name
            from procedureevents_mv a, D_ITEMS b
            where b.category  in ('Access Lines - Invasive') 
            and a.itemid = b.itemid'''
    print(qry)
    catheter_df = pd.read_sql_query(qry, conn)
    catheter_df.to_csv(outputfile)

    print(' created ' + outputfile)


def create_output_liquid_data (control_file,
                              outputevents_file_name,
                              d_items_file_name,
                               output_file = data_dir+'output_liquid_urine.csv'):

    control_df = pd.read_csv(control_file, encoding="ISO-8859-1")
    # Make the db in memory
    conn = sqlite3.connect(':memory:')

    d_items = pd.read_csv(d_items_file_name, compression='gzip',
                             # nrows=100000,
                             error_bad_lines=False)
    d_items.to_sql('D_ITEMS', conn, index=False)
    print(d_items.shape)

    OUTPUTENENTS = pd.read_csv(outputevents_file_name, compression='gzip',
                            # nrows=1000000, #todo
                            error_bad_lines=False)

    print(OUTPUTENENTS.shape)
    # write the tables
    OUTPUTENENTS.to_sql('OUTPUTEVENTS ', conn, index=False)


    # sometimes end_time< start_time and amount is negative, it mix the amount - changed it to originalamount
    qry = '''select subject_id as patient_id,charttime as time,-1*value as value, label as parameter_name
            from OUTPUTEVENTS a, D_ITEMS b
            where a.ITEMID = b.ITEMID
            and UNITNAME = 'mL'
            and LABEL in ('R Ureteral Stent','L Ureteral Stent','Foley','Void','R Nephrostomy',
            'L Nephrostomy','Straight Cath')'''

    print (qry)
    all_liquid_df = pd.read_sql_query(qry, conn)
    all_liquid_df = all_liquid_df.patient_id.isin(control_df.patient_id)
    all_liquid_df.to_csv(output_file)

    return


#the main function to create liquid file - call all the others
def create_liquid_balance (control_file,num_days_list, min_value=0,max_value=3000,
                           input_liquids_per_hour_for_modeling_file=data_dir+'input_liquids_per_hour_for_modeling.csv',
                           urine_filename=data_dir + 'output_liquid_urine.csv'):

    print ('start create_liquid_balance')

    # input
    create_liquids_file.create_output_liquid_data(control_file,
                              min_value, max_value, 'value','input_liquid',
                              liquid_source_file=input_liquids_per_hour_for_modeling_file
                              ,num_days_list=num_days_list)

    # all urine
    create_liquids_file.create_output_liquid_data(control_file,   min_value, max_value, 'value',
                              'urine',
                              liquid_source_file_list=[urine_filename],num_days_list=num_days_list)




    # # # total_liquid
    create_liquids_file.create_output_liquid_data(control_file,
                              min_value, max_value, 'value','total_liquid',
                              liquid_source_file_list=[input_liquids_per_hour_for_modeling_file,
                                                       urine_filename],
                                                        num_days_list = num_days_list)


def convert_commans_to_min_amount (all_liquid_df,output_file = output_dir+'input_liquids_per_hour_for_modeling.csv'):
    list_rows = list()
    num_rows = all_liquid_df.shape[0]
    print ('num_rows : '+str(num_rows))
    # all_liquid_df = pd.read_csv(data_dir +'input_liquids_for_modeling.csv', encoding="ISO-8859-1")
    for index, row in all_liquid_df.iterrows():
        parameter_name = row.order_name
        patient_id = row.patient_id

        time_range = pd.date_range(row.start_date, pd.to_datetime(row.end_date)-pd.to_timedelta(1,'m'), freq='1min')
        # print (' len(time_range) = '+ str(len(time_range)))

        for time1 in time_range:
            #15/10/18 - replace calculation of rate by rate column
            # list_rows.append([patient_id,time1,parameter_name,row.val_given/len(time_range)] )
            list_rows.append([patient_id, time1, parameter_name, row.rate / 60])

        if (index%1000 ==0):
            print (index)
    data = pd.DataFrame(list_rows)
    data.columns = ['patient_id','time','parameter_name','value']


    # group by hour
    data['time_hour'] = data['time'].dt.floor('h')
    group_functions = {'value': {'value':'sum'}}
    data_group = data.groupby(['patient_id','time_hour']).agg(group_functions)
    data_group.columns = data_group.columns.droplevel(0)
    data_group.reset_index(level=0, inplace=True)

    # data_group.to_csv('input_liquid_to_check.csv')
    if data_group.shape[0]>0:
        data_group['parameter_name'] = 'input_liquid'

    data_group = data_group.rename(columns={'time_hour': 'time'})
    data_group.index.names = ['time']


    data_group.to_csv(output_file)
    return output_file

def create_input_liquid_data( control_file,
                              inputevents_mv_file_name,
                              d_items_file_name,
                              output_file = output_dir+'input_liquids_per_hour_for_modeling.csv',
                              num_days = 5):
    control_df = pd.read_csv(control_file, encoding="ISO-8859-1")
    # Make the db in memory
    conn = sqlite3.connect(':memory:')

    #todo - delete comment
    # d_items = pd.read_csv(d_items_file_name, compression='gzip',
    #                          # nrows=100000,
    #                          error_bad_lines=False)
    # d_items.to_sql('D_ITEMS', conn, index=False)
    # print(d_items.shape)
    #
    # INPUTEVENTS_MV = pd.read_csv(inputevents_mv_file_name, compression='gzip',
    #                         # nrows=1000000, #todo
    #                         error_bad_lines=False)
    #
    # print(INPUTEVENTS_MV.shape)
    # # write the tables
    # INPUTEVENTS_MV.to_sql('INPUTEVENTS_MV', conn, index=False)
    #
    #
    #
    # # sometimes end_time< start_time and amount is negative, it mix the amount - changed it to originalamount
    # qry = '''select COALESCE(rate,amount*60) as rate,subject_id  as patient_id, hadm_id,icustay_id,
    #                 starttime as start_date, endtime as end_date,
    #                 label as order_name, amount,originalamount , amountuom,originalrate,patientweight
    #         from INPUTEVENTS_MV a, D_ITEMS b
    #         where a.ITEMID = b.ITEMID
    #         and UNITNAME = 'mL'
    #         and CATEGORY in ('Fluids - Other (Not In Use)','Fluids/Intake','Nutrition - Enteral',
    #                 'Nutrition - Parenteral','Blood Products/Colloids')'''
    # print (qry)
    # all_liquid_df = pd.read_sql_query(qry, conn)
    # all_liquid_df.to_csv(output_dir+'tmp_input_liquid.csv')
    all_liquid_df = Utils.read_data(output_dir+'tmp_input_liquid.csv')
    df_join = pd.merge(all_liquid_df[['patient_id','rate','start_date','end_date','order_name']],
                       control_df[['patient_id','date_sample']],
                                      on = 'patient_id',how = 'inner')
    df_join.date_sample = pd.to_datetime(df_join.date_sample)
    df_join.time = pd.to_datetime(df_join.time)
    df_join['admission start day icu'] = pd.to_datetime(df_join['admission start day icu'])


    filter_df_join = df_join.loc[(df_join.time <= df_join.date_sample) & (
        df_join.date_sample.subtract(df_join.time) <= pd.to_timedelta(num_days, unit='d')), :]

    return convert_commans_to_min_amount(filter_df_join,output_file)

def prepare_diagnosis_data (control_file, diagnoses_icd_file_name = mimic.create_control_set.diagnoses_icd_file_name,
                             diagmosis_types_file = data_dir+'D_ICD_DIAGNOSES_marked.xlsx'):


    diagnoses_icd_df = pd.read_csv(diagnoses_icd_file_name, compression='gzip',
                             # nrows=100000,
                             error_bad_lines=False)
    
    control_df = Utils.read_data(control_file)
    diagnoses_types_df = Utils.read_data(diagmosis_types_file)
    diagnoses_icd_df.ICD9_CODE = diagnoses_icd_df.ICD9_CODE.astype(str)
    diagnoses_types_df.ICD9_CODE = diagnoses_types_df.ICD9_CODE.astype(str)

    diagnoses_df = pd.merge (diagnoses_types_df,diagnoses_icd_df, on =['ICD9_CODE'],how='inner')

    result_df = control_df[['HADM_ID','patient_id','date_sample']]
    types = diagnoses_types_df.category.drop_duplicates()

    for type in types:
        if str(type)=='nan':
            continue
        print (type)
        diagnoses_df_tmp = diagnoses_df[diagnoses_df.category==type]
        mask = (result_df.HADM_ID.isin(diagnoses_df_tmp.HADM_ID))
        col_name = 'diagnoses_is_'+str(type.replace (' ','_'))
        result_df[col_name] = 0
        result_df.loc[mask,col_name] =1


    result_df.to_csv(output_dir+'signals//'+'diagnoses.csv')

    # get all diagnosis for control
    # for any type  add column + mask
    
def prepare_GSC_data (gsc_file =data_dir + 'GSC_score_data_for_modeling.csv'):

    print ('start prepare_GSC')
    gsc_df = Utils.read_data(gsc_file)
    gsc_df = gsc_df.rename(
        columns={'SUBJECT_ID': 'patient_id', 'VALUE': 'value', 'CHARTTIME': 'time', 'LABEL': 'parameter_name'})

    group_functions = {'VALUENUM': {'value': 'sum'}}
    gsc_df_grp = gsc_df.groupby(['patient_id','time']).agg(group_functions)
    gsc_df_grp.columns = gsc_df_grp.columns.droplevel(0)
    gsc_df_grp.reset_index(level=0, inplace=True)
    gsc_df_grp.reset_index(level=0, inplace=True)
    gsc_df_grp ['parameter_name'] = 'gsc'
    gsc_df_grp.to_csv(data_dir+'mimic_gsc_score.csv')


def get_intubation_query_change():
    query = '''
        select subject_id,hadm_id
      icustay_id, charttime
      -- case statement determining whether it is an instance of mech vent
      , max(
        case
          when itemid is null or value is null then 0 -- can't have null values
          --Michael when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
          --Michael when itemid = 223848 and value != 'Other' THEN 1
          --Michael when itemid = 223849 then 1 -- ventilator mode
          --Michael when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
          when itemid = 648 and value = 'Intubated/trach' THEN 1 -- Speech = intubated *
          --Michael when itemid = 223900 and value = 'No Response-ETT' THEN 1
          when itemid in
            (
            445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
            , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
            , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
            , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
            , 543 -- PlateauPressure
            , 5865,5866,224707,224709,224705,224706 -- APRV pressure
            , 60,437,505,506,686,220339,224700 -- PEEP
            , 3459 -- high pressure relief
            , 501,502,503,224702 -- PCV
            , 223,667,668,669,670,671,672 -- TCPCV
            , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT*
            , 224701 -- PSVlevel
            )
            THEN 1
          else 0
        end
        ) as MechVent
        , max(
          case when itemid is null or value is null then 0
            -- extubated indicates ventilation event has ended
            when itemid = 640 and value = 'Extubated' then 1*
            when itemid = 640 and value = 'Self Extubation' then 1 *
            -- initiation of oxygen therapy indicates the ventilation has ended
            when itemid = 226732 and value in
            (
              'Nasal cannula', -- 153714 observations
              'Face tent', -- 24601 observations
              'Aerosol-cool', -- 24560 observations
              --'Trach mask ', -- 16435 observations (not extubated)
              'High flow neb', -- 10785 observations
              'Non-rebreather', -- 5182 observations
              'Venti mask ', -- 1947 observations
              'Medium conc mask ', -- 1888 observations
              --'T-piece', -- 1135 observations (not extubated)*
              'High flow nasal cannula', -- 925 observations
              'Ultrasonic neb', -- 9 observations
              'Vapomist' -- 3 observations
            ) then 1
            when itemid = 467 and value in
            (
              'Cannula', -- 278252 observations
              'Nasal Cannula', -- 248299 observations
              'None', -- 95498 observations
              'Face Tent', -- 35766 observations
              'Aerosol-Cool', -- 33919 observations
              --'Trach Mask', -- 32655 observations (not extubated)
              'Hi Flow Neb', -- 14070 observations
              'Non-Rebreather', -- 10856 observations
              'Venti Mask', -- 4279 observations
              'Medium Conc Mask', -- 2114 observations
              'Vapotherm', -- 1655 observations
             -- 'T-Piece', -- 779 observations (not extubated)*
              'Hood', -- 670 observations
              'Hut', -- 150 observations
              --'TranstrachealCat', -- 78 observations (not extubated) *
              'Heated Neb', -- 37 observations
              'Ultrasonic Neb' -- 2 observations
            ) then 1
          else 0
          end
          )
          as Extubated
        , max(
          case when itemid is null or value is null then 0
            when itemid = 640 and value = 'Self Extubation' then 1
          else 0
          end
          )
          as SelfExtubated
    from chartevents ce
    where ce.value is not null
    -- exclude rows marked as error
    and ifnull(ce.error,-1) <> 1
    -- and ce.error IS DISTINCT FROM 1
    and itemid in
    (
        -- the below are settings used to indicate ventilation
          648, 223900 -- speech
        , 720, 223849 -- vent mode
        , 223848 -- vent type
        , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
        , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
        , 218,436,535,444,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
        , 543 -- PlateauPressure
        , 5865,5866,224707,224709,224705,224706 -- APRV pressure
        , 60,437,505,506,686,220339,224700 -- PEEP
        , 3459 -- high pressure relief
        , 501,502,503,224702 -- PCV
        , 223,667,668,669,670,671,672 -- TCPCV
        , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT
        , 224701 -- PSVlevel

        -- the below are settings used to indicate extubation
        , 640 -- extubated

        -- the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
        , 468 -- O2 Delivery Device#2
        , 469 -- O2 Delivery Mode
        , 470 -- O2 Flow (lpm)
        , 471 -- O2 Flow (lpm) #2
        , 227287 -- O2 Flow (additional cannula)
        , 226732 -- O2 Delivery Device(s)
        , 223834 -- O2 Flow

        -- used in both oxygen + vent calculation
        , 467 -- O2 Delivery Device
    )
    group by icustay_id, subject_id,hadm_id,charttime

    '''
    return query

def get_intubation_rows ():

    query = '''
    select *
     from chartevents ce
    where ce.value is not null
    -- exclude rows marked as error
    and ifnull(ce.error,-1) <> 1
    -- and ce.error IS DISTINCT FROM 1
    and itemid in
    (
        -- the below are settings used to indicate ventilation
          648, 223900 -- speech
        , 720, 223849 -- vent mode
        , 223848 -- vent type
        , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
        , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
        , 218,436,535,444,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
        , 543 -- PlateauPressure
        , 5865,5866,224707,224709,224705,224706 -- APRV pressure
        , 60,437,505,506,686,220339,224700 -- PEEP
        , 3459 -- high pressure relief
        , 501,502,503,224702 -- PCV
        , 223,667,668,669,670,671,672 -- TCPCV
        , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT
        , 224701 -- PSVlevel
    
        -- the below are settings used to indicate extubation
        , 640 -- extubated
    
        -- the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
      --anat  , 468 -- O2 Delivery Device#2
       --anat , 469 -- O2 Delivery Mode
       --anat , 470 -- O2 Flow (lpm)
       --anat , 471 -- O2 Flow (lpm) #2
       --anat , 227287 -- O2 Flow (additional cannula)
      --anat  , 226732 -- O2 Delivery Device(s)
      --anat  , 223834 -- O2 Flow
    
        -- used in both oxygen + vent calculation
       --anat , 467 -- O2 Delivery Device
    )
    
    '''
    return query
def get_intubation_query():

    query = '''
        select subject_id,hadm_id
      icustay_id, charttime
      -- case statement determining whether it is an instance of mech vent
      , max(
        case
          when itemid is null or value is null then 0 -- can't have null values
          when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
          when itemid = 223848 and value != 'Other' THEN 1
          when itemid = 223849 then 1 -- ventilator mode
          when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
          when itemid = 648 and value = 'Intubated/trach' THEN 1 -- Speech = intubated
          when itemid = 223900 and value = 'No Response-ETT' THEN 1
          when itemid in
            (
            445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
            , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
            , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
            , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
            , 543 -- PlateauPressure
            , 5865,5866,224707,224709,224705,224706 -- APRV pressure
            , 60,437,505,506,686,220339,224700 -- PEEP
            , 3459 -- high pressure relief
            , 501,502,503,224702 -- PCV
            , 223,667,668,669,670,671,672 -- TCPCV
            , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT
            , 224701 -- PSVlevel
            )
            THEN 1
          else 0
        end
        ) as MechVent
        , max(
          case when itemid is null or value is null then 0
            -- extubated indicates ventilation event has ended
            when itemid = 640 and value = 'Extubated' then 1
            when itemid = 640 and value = 'Self Extubation' then 1
            -- initiation of oxygen therapy indicates the ventilation has ended
            when itemid = 226732 and value in
            (
              'Nasal cannula', -- 153714 observations
              'Face tent', -- 24601 observations
              'Aerosol-cool', -- 24560 observations
              --'Trach mask ', -- 16435 observations (not extubated)
              'High flow neb', -- 10785 observations
              'Non-rebreather', -- 5182 observations
              'Venti mask ', -- 1947 observations
              'Medium conc mask ', -- 1888 observations
              --'T-piece', -- 1135 observations (not extubated)
              'High flow nasal cannula', -- 925 observations
              'Ultrasonic neb', -- 9 observations
              'Vapomist' -- 3 observations
            ) then 1
            when itemid = 467 and value in
            (
              'Cannula', -- 278252 observations
              'Nasal Cannula', -- 248299 observations
              'None', -- 95498 observations
              'Face Tent', -- 35766 observations
              'Aerosol-Cool', -- 33919 observations
              --'Trach Mask', -- 32655 observations (not extubated)
              'Hi Flow Neb', -- 14070 observations
              'Non-Rebreather', -- 10856 observations
              'Venti Mask', -- 4279 observations
              'Medium Conc Mask', -- 2114 observations
              'Vapotherm', -- 1655 observations
             -- 'T-Piece', -- 779 observations (not extubated)
              'Hood', -- 670 observations
              'Hut', -- 150 observations
              --'TranstrachealCat', -- 78 observations (not extubated)
              'Heated Neb', -- 37 observations
              'Ultrasonic Neb' -- 2 observations
            ) then 1
          else 0
          end
          )
          as Extubated
        , max(
          case when itemid is null or value is null then 0
            when itemid = 640 and value = 'Self Extubation' then 1
          else 0
          end
          )
          as SelfExtubated
    from chartevents ce
    where ce.value is not null
    -- exclude rows marked as error
    and ifnull(ce.error,-1) <> 1
    -- and ce.error IS DISTINCT FROM 1
    and itemid in
    (
        -- the below are settings used to indicate ventilation
          648, 223900 -- speech
        , 720, 223849 -- vent mode
        , 223848 -- vent type
        , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
        , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
        , 218,436,535,444,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
        , 543 -- PlateauPressure
        , 5865,5866,224707,224709,224705,224706 -- APRV pressure
        , 60,437,505,506,686,220339,224700 -- PEEP
        , 3459 -- high pressure relief
        , 501,502,503,224702 -- PCV
        , 223,667,668,669,670,671,672 -- TCPCV
        , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT
        , 224701 -- PSVlevel
    
        -- the below are settings used to indicate extubation
        , 640 -- extubated
    
        -- the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
        , 468 -- O2 Delivery Device#2
        , 469 -- O2 Delivery Mode
        , 470 -- O2 Flow (lpm)
        , 471 -- O2 Flow (lpm) #2
        , 227287 -- O2 Flow (additional cannula)
        , 226732 -- O2 Delivery Device(s)
        , 223834 -- O2 Flow
    
        -- used in both oxygen + vent calculation
        , 467 -- O2 Delivery Device
    )
    group by icustay_id, subject_id,hadm_id,charttime

    '''
    return query

def prepare_intubation_data():
    conn = sqlite3.connect(':memory:')

    extubation_file = data_dir + 'extubation_data_for_modeling.csv'
    prepare_extubation_data_from_procedure_mv(extubation_file)

    intubation_file = data_dir + 'intubation_data_for_modeling.csv'
    intubation_rel_data_file = data_dir + 'intubation_relevant_rows_for_modeling.csv'
    prepare_intubation_data_from_chartevents(intubation_file, intubation_data_file=intubation_rel_data_file)


    # -----------To use ventsetting script s in mimic....
    # intubation_df = Utils.read_data(intubation_file)
    # extubation_df = Utils.read_data(extubation_file)
    #
    # intubation_df = intubation_df.rename(columns = {'CHARTTIME':'charttime',
    #                                                'ICUSTAY_ID':'icustay_id',
    #                                                'SUBJECT_ID':'subject_id',
    #                                                'HADM_ID':'hadm_id'
    #                                                })
    # extubation_df = extubation_df.rename(columns = {'CHARTTIME':'charttime',
    #                                                'ICUSTAY_ID':'icustay_id',
    #                                                'SUBJECT_ID':'subject_id',
    #                                                'HADM_ID':'hadm_id'
    #                                                })
    #
    # ventsettings = pd.concat([intubation_df,extubation_df])
    # ventsettings.to_csv(data_dir+'ventsettings.csv',compression = 'zip')
    # ------------end ventsetting-----------


    # union_df.to_sql('ventsettings', conn, index=False)

    # ventsettings = ventsettings.sort_values(by=['icustay_id', 'charttime'], ascending=[True, True])
    #
    # ventsettings['prev_charttime'] = ventsettings.groupby(['hadm_id','icustay_id','subject_id',
    #                                                        'Extubated','MechVent','SelfExtubated'])['charttime'].shift(1)
    #
    # ventsettings['diff_from_prev_charttime'] = ventsettings['charttime'].subtract(ventsettings['prev_charttime'])

    # ventsettings['newvent']= 0
    # ventsettings.loc[ventsettings.Extubated ==1, 'newvent'] = 0
    # ventsettings.loc[ventsettings.diff_from_prev_charttime>pd.to_timedelta(8,'h'),'newvent'] = 1
    # ventsettings.to_csv(output_dir + 'ventsettings1.csv')

    # qry = '''
    #         -- create the durations for each mechanical ventilation instance
    #     select icustay_id, subject_id, hadm_id,ventnum
    #       , min(charttime) as starttime
    #       , max(charttime) as endtime
    #       , extract(epoch from max(charttime)-min(charttime))/60/60 AS duration_hours
    #     from
    #     (
    #       select vd1.*
    #       -- create a cumulative sum of the instances of new ventilation
    #       -- this results in a monotonic integer assigned to each instance of ventilation
    #       , case when MechVent=1 or Extubated = 1 then
    #           SUM( newvent )
    #           OVER ( partition by icustay_id, subject_id, hadm_id order by charttime )
    #         else null end
    #         as ventnum
    #       --- now we convert CHARTTIME of ventilator settings into durations
    #       from ( -- vd1
    #           select
    #               icustay_id, subject_id, hadm_id
    #               -- this carries over the previous charttime which had a mechanical ventilation event
    #               , case
    #                   when MechVent=1 then
    #                     LAG(CHARTTIME, 1) OVER (partition by icustay_id, subject_id, hadm_id, MechVent order by charttime)
    #                   else null
    #                 end as charttime_lag
    #               , charttime
    #               , MechVent
    #               , Extubated
    #               , SelfExtubated
    #
    #               -- if this is a mechanical ventilation event, we calculate the time since the last event
    #               , case
    #                   -- if the current observation indicates mechanical ventilation is present
    #                   when MechVent=1 then
    #                   -- copy over the previous charttime where mechanical ventilation was present
    #                     CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, subject_id, hadm_id, MechVent order by charttime))
    #                   else null
    #                 end as ventduration
    #
    #               -- now we determine if the current mech vent event is a "new", i.e. they've just been intubated
    #               , case
    #                 -- if there is an extubation flag, we mark any subsequent ventilation as a new ventilation event
    #                   when Extubated = 1 then 0 -- extubation is *not* a new ventilation event, the *subsequent* row is
    #                   when
    #                     LAG(Extubated,1)
    #                     OVER
    #                     (
    #                     partition by icustay_id, subject_id, hadm_id, case when MechVent=1 or Extubated=1 then 1 else 0 end
    #                     order by charttime
    #                     )
    #                     = 1 then 1
    #                     -- if there is less than 8 hours between vent settings, we do not treat this as a new ventilation event
    #                   when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, subject_id, hadm_id, MechVent order by charttime))) <= interval '8' hour
    #                     then 0
    #                 else 1
    #                 end as newvent
    #           -- use the staging table with only vent settings from chart events
    #           FROM ventsettings
    #       ) AS vd1
    #       -- now we can isolate to just rows with ventilation settings/extubation settings
    #       -- (before we had rows with extubation flags)
    #       -- this removes any null values for newvent
    #       where
    #         (MechVent = 1 or Extubated = 1)
    #     ) AS vd2
    #     -- exclude the "0th" occurence of mech vent
    #     -- this is usually NIV/oxygen, which is our surrogate for extubation,
    #     -- occurring before the actual mechvent event
    #     where ventnum > 0
    #     group by icustay_id, subject_id, hadm_id, ventnum
    #     order by icustay_id, subject_id, hadm_id, ventnum
    #     '''
    # print(qry)
    # intubation_extubation_df = pd.read_sql_query(qry, conn)
    # intubation_extubation_df.to_csv(data_dir+'mimic_intubation_extubation_data.csv')


def prepare_extubation_data_from_procedure_mv(outputfile):
    # -- add in the extubation flags from procedureevents_mv
    # -- note that we only need the start time for the extubation
    # -- (extubation is always charted as ending 1 minute after it started)
    conn = sqlite3.connect(':memory:')

    procedureevents_mv_file_name = mimic.create_control_set.procedureevents_mv_file_name

    print(' try read ' + procedureevents_mv_file_name)
    procedureevents_mv = pd.read_csv(procedureevents_mv_file_name, compression='gzip',
                                     # nrows=100000,
                                     error_bad_lines=False)

    print(procedureevents_mv.shape)
    # write the tables
    procedureevents_mv.to_sql('procedureevents_mv', conn, index=False)

    qry = '''
    select a.*, 
    case when a.itemid=227194 then 'Extubation'
    when a.itemid=225468 then 'Unplanned Extubation (patient-initiated)'
    when a.itemid=225477 then 'Unplanned Extubation (non-patient initiated)'
    when a.itemid=225448 then 'Percutaneous Tracheostomy'
    when a.itemid=226237 then 'Open Tracheostomy'
    when a.itemid=224385 then 'Intubation'
    else '-1' end as description
    from procedureevents_mv a
    where itemid in
    (
      227194 -- "Extubation"
    , 225468 -- "Unplanned Extubation (patient-initiated)"
    , 225477 -- "Unplanned Extubation (non-patient initiated)"
    ,225448 --"Percutaneous Tracheostomy"
    ,226237 --"Open Tracheostomy"
    ,224385 --"Intubation"
    )
    '''
    print(qry)
    extubation_df = pd.read_sql_query(qry, conn)
    extubation_df.to_csv(outputfile)

    # qry = '''select
    #   icustay_id, subject_id,hadm_id,starttime as charttime
    #   , case when itemid = 224385 then 1 else 0 end as MechVent
    #   , case when itemid in (227194,225468,225477) then 1 else 0 end as Extubated
    #   , case when itemid = 225468 then 1 else 0 end as SelfExtubated
    #   , case when itemid in (225448,226237) then 1 else 0 end as trachostomy
    # from procedureevents_mv
    # where itemid in
    # (
    #   227194 -- "Extubation"
    # , 225468 -- "Unplanned Extubation (patient-initiated)"
    # , 225477 -- "Unplanned Extubation (non-patient initiated)"
    # ,225448 --"Percutaneous Tracheostomy"
    # ,226237 --"Open Tracheostomy"
    # ,224385 --"Intubation"
    # )
    # '''
    # print(qry)
    # extubation_df = pd.read_sql_query(qry, conn)
    # extubation_df.to_csv(outputfile)


def prepare_extubation_data_from_procedure_mv_bu(outputfile):
    # -- add in the extubation flags from procedureevents_mv
    # -- note that we only need the start time for the extubation
    # -- (extubation is always charted as ending 1 minute after it started)
    conn = sqlite3.connect(':memory:')

    procedureevents_mv_file_name = mimic.create_control_set.procedureevents_mv_file_name

    print(' try read ' + procedureevents_mv_file_name)
    procedureevents_mv = pd.read_csv(procedureevents_mv_file_name, compression='gzip',
                                     # nrows=100000,
                                     error_bad_lines=False)

    print(procedureevents_mv.shape)
    # write the tables
    procedureevents_mv.to_sql('procedureevents_mv', conn, index=False)

    qry = '''select
      icustay_id, subject_id,hadm_id,starttime as charttime
      , 0 as MechVent
      , 1 as Extubated
      , case when itemid = 225468 then 1 else 0 end as SelfExtubated
      , case when itemid in (225448,226237)
    from procedureevents_mv
    where itemid in
    (
      227194 -- "Extubation"
    , 225468 -- "Unplanned Extubation (patient-initiated)"
    , 225477 -- "Unplanned Extubation (non-patient initiated)"
    ,225448 --"Percutaneous Tracheostomy"
    ,226237 --"Open Tracheostomy"
    )
    '''
    print(qry)
    extubation_df = pd.read_sql_query(qry, conn)
    extubation_df.to_csv(outputfile)

def prepare_intubation_data_from_chartevents (intubation_file,chartevents_dir = data_dir + 'CHARTEVENTS_files',
                                              intubation_data_file = data_dir + 'intubation_relevant_rows_for_modeling.csv'):


    files = os.listdir(chartevents_dir)
    chartevents_intubation_partial_list = list()
    chartevents_partial_list = list ()
    try:

        for idx, file_name in enumerate(files):
            print (file_name)

            # if idx>3:
            #     break
            if idx%10 ==0 :
                print ('handle '+ str(idx) + ' files from chartevents dir')
            skiprows = 0
            if (idx==0):
                skiprows=1
            partial_chartevents = pd.read_csv(chartevents_dir+'//'+file_name,
                                  # nrows=100,#todo
                                  error_bad_lines=False,
                                              skiprows=skiprows)# first row in second parts is header

            conn = sqlite3.connect(':memory:')
            # write the tables

            partial_chartevents.to_sql('chartevents', conn, index=False)

            # --------for original qry od mimic - all intubation time, not only ETT
            # qry = get_intubation_query()
            # # print(qry)
            # tmp_df = pd.read_sql_query(qry, conn)
            # print (tmp_df.shape)
            # chartevents_partial_list.append (tmp_df)
            #-----------------

            qry = get_intubation_rows()
            # print(qry)
            tmp_df_intu = pd.read_sql_query(qry, conn)
            print (tmp_df_intu.shape)
            chartevents_intubation_partial_list.append (tmp_df_intu)


    except Exception as e:
        print(e)
        print ('problem in file ' + file_name)
    finally:
        # join all df in each type
        # -------for original qry od mimic - all intubation time, not only ETT
        # joined_df =pd.concat(chartevents_partial_list)
        # # add parameter name
        #
        # tmp_file_name = intubation_file
        # joined_df.to_csv(tmp_file_name)
        # print ('created '+tmp_file_name)
        # -----------------


        joined_df =pd.concat(chartevents_intubation_partial_list)
        # add parameter name

        tmp_file_name = intubation_data_file
        joined_df.to_csv(tmp_file_name)
        print ('created '+tmp_file_name)


def prepare_chartevents_files (d_items_file_name,chartevents_dir = data_dir + 'CHARTEVENTS_files'):
    #item id
    # 223762 = Temperature Celsius
    # 220045 = Heart rate
    # 220050 - Arterial Pressure Systolic
    # 220051 - Arterial Blood Pressure diastolic
    # 220052 - Arterial Blood Pressure mean
    # 220210 - Respiratory Rate

    d_items = pd.read_csv(d_items_file_name, compression='gzip',
                             # nrows=100000,
                             error_bad_lines=False)

    chartevents_param_list = list ()
    # todo - delete comment
    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[224639,226512,226707],param_name='general')) #height, weight
    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[227543, 227546, 227549, 224842, 220059, 220060,220061,220069, 220074, 220088,226272, 223771, 223772],param_name='Hemodynamics')) #CO (Arterial),SVV (Arterial),ScvO2 (Presep),
    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[227073],param_name='Labs'))
    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[228368,228369,228370,228371,228374,228375,228376,228377],param_name='NICOM'))

    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[228176,228177,228178,228179,228180,228181,228182,228183,228184,228185],param_name='PiCCO'))

    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[227187,227287,224685,224687,224690,224691,224695,224696,224697,224700,224701,227579,227580,224746,224747,220210,220277,220283,220339,223835],param_name='Respiratory')) ,

    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[220181,228640,224359,220045,220050,220051,220052,225309,225310,225312,223761,224167,227243,220179],param_name='Routine_Vital_Signs'))
    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[228610,228611,228612,228613,228614,228615,228616,228617,228618,228619,228620,228621,
                                                                                     228622,228623,228624,228625,228626,228627,228628,228629,227596,227597,227602,227603,227614,227615,224846,224951,224952,224953,224954,224955,
                                                                                     224956,224957,224562,224563,224895,224896,224897,224898,224899,224900,224901,224916,224917,224918,224919,224920,224921,224922],param_name='Skin'))

    chartevents_param_list.append(ChartEventParam.Chartevent_param_data(itemid_list=[223900,223901,220739],param_name='GSC_score')) ,




    files = os.listdir(chartevents_dir)
    try:

        for idx, file_name in enumerate(files):
            print (file_name)

            # if idx>10:
            #     break
            if idx%10 ==0 :
                print ('handle '+ str(idx) + ' files from chartevents dir')
            skiprows = 0
            if (idx==0):
                skiprows=1
            partial_chartevents = pd.read_csv(chartevents_dir+'//'+file_name,
                                  # nrows=100,#todo
                                  error_bad_lines=False,
                                              skiprows=skiprows)# first row in second parts is header
            # print (partial_chartevents.keys())
            partial_chartevents.ITEMID = pd.to_numeric(partial_chartevents.ITEMID ,errors = 'ignore')
            partial_chartevents.HADM_ID = pd.to_numeric(partial_chartevents.HADM_ID, errors = 'ignore')
            for chartev in chartevents_param_list:

                tmp_df = partial_chartevents[partial_chartevents.ITEMID.isin(chartev.itemid_list)]
                chartev.add_df (tmp_df)
                # print (chartev.param_name+':' +str(tmp_df.shape[0]))

                # tmp1 = partial_chartevents[partial_chartevents.HADM_ID.isin([138376])]
                # if tmp1.shape[0]>0:
                #     print (chartev.param_name)
                #     print ('num rows for 138376 is '+ str(tmp1.shape[0]))

    except Exception as e:
        print(e)
        print ('problem in file ' + file_name)
    finally:
        # join all df in each type
        for chartev in chartevents_param_list:
            print ('join '+ chartev.param_name)
            joined_df =pd.concat(chartev.df_list)
            # add parameter name
            joined_df_with_label = pd.merge(d_items[['ITEMID', 'LABEL']], joined_df, on=['ITEMID'], how='inner')
            tmp_file_name = data_dir+chartev.param_name+'_data_for_modeling.csv'

            if joined_df_with_label.shape[0]>100:
                joined_df_with_label.to_csv(tmp_file_name)
                print ('created '+tmp_file_name)
            else :
                print ('few rows for '+ chartev.param_name)



    print ('end prepare_chartevents_files')



def create_diarhea_med_data(prescriptions_file_name):
    '''
    ('Lactulose',
'Lactuose Enema',
'lactul',
'Bisacodyl',
'Polyethylene Glycol 400 (Bulk)',
'Fleet Enema',
'Biscolax',
'Glycerin (Laxative)'
    :param prescriptions_file_name:
    :return:
    '''
    pass

if __name__ == '__main__':
    # to create csv from zip file
    create_stool_data(mimic.create_control_set.outputevents_file_name)
