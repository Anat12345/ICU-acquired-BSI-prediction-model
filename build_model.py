import my_modeling
import Model_Param
import pandas as pd
import run_stat_test

# result_20181219170654.csv
train_file_2 = 'all_features_cleaned_train_2_days_in_icu_4_days_after_pos_20181219165315.csv'
test_file_2 = 'all_features_cleaned_test_2_days_in_icu_4_days_after_pos_20181219165315.csv'

# result_20181221000741
train_file_2 = 'all_features_cleaned_train_2_days_in_icu_4_days_after_pos_20181221000437.csv' #fix bugs , all icu with elective
test_file_2 = 'all_features_cleaned_test_2_days_in_icu_4_days_after_pos_20181221000437.csv' #fix bugs , all icu with elective


# after add fft + smoothing
train_file_2 = 'all_features_cleaned_train_2_days_in_icu_4_days_after_pos_20181223212906.csv'
all_file = 'all_features_cleaned_2_days_in_icu_4_days_after_pos_20181223212809.csv'

# after fix - result_20190103090331
train_file_2 = 'all_features_cleaned_train_2_days_in_icu_4_days_after_pos_20181229224258.csv'
output_dir = "C://Users//michael//Desktop//predicting BSI//files//mimic//Output_2//"
my_modeling.output_dir = output_dir

# fix fio2, add 5 days features+... result_20190117215403 0.89
train_file_2 = 'all_features_cleaned_train_2_days_in_icu_4_days_after_pos_20190117212610.csv'
test_file_2 = 'all_features_cleaned_test_2_days_in_icu_4_days_after_pos_20190117212610.csv'

#intubation - fix bug in death as 0(failure) result_20190306215737 - 0.9
output_dir = 'C://Users//michael//Desktop//predicting BSI//files//mimic//Output_intubation_2//'
train_file_2 ='all_features_cleaned_2_days_in_icu_2_days_after_pos_20190306213841.csv'

# exclude death in next 2 month - result_20190307104552 - 0.85
train_file_2 = 'all_features_cleaned_2_days_in_icu_2_days_after_pos_20190307092928.csv'
def run_modeling():

    my_modeling.run_cv(data_train_file=output_dir+train_file_2, use_fs=True,column_to_ignore=['death_in_a_month'])

def check_on_test (result_file ='result_20181221000741.csv', additional_desc = ''):
    result_df = pd.read_csv(output_dir +result_file)
    import json
    import ast
    result_df_unnique = result_df[['features','weights','model_idx','num_features_per_model','sample_ratio_majority','sample_ratio_minority']].drop_duplicates()

    for index, row in result_df_unnique.iterrows():
        print (row.model_idx)
        weights = json.loads(row.weights)
        features = ast.literal_eval(row.features)
        sample_ratio_majority = row.sample_ratio_majority
        sample_ratio_minority = row.sample_ratio_minority
        num_features = row.num_features_per_model
        fig_name = result_file.replace('.csv','.png')
        fig_name = fig_name.replace('.png', '_' + str(row.model_idx) + additional_desc + '.png')
        my_modeling.run_correct_process(train_file=output_dir+train_file_2
                            , test_file=output_dir+test_file_2,
                            current_model=Model_Param.Model_Param(
                                feature_selection_method='xgboost',
                                model=None,
                                sample_ratio_majority=sample_ratio_majority,
                                num_features=num_features,
                                sample_ratio_minority=sample_ratio_minority,
                                num_models=1, scaling='none',
                                weights=weights),
                            features_in=features, run_cv_flag=False,
                            fig_name=fig_name, use_search=False)

if __name__ == "__main__":
    run_modeling()
    # check_on_test('result_20190117215403.csv',additional_desc = '_old_boostapp')
    # run_stat_test.get_stat_numeric( output_dir+all_file, output_dir_in=output_dir)
