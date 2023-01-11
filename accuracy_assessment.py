#this script will be used for accuracy assessment for the ref fires


#import the modules
import pandas as pd

#make a tool
def accuracy_info(table_data, ref, mtbs, lba, lba_bp, frap, mine, out_path=None):
    '''

    :param data:
    :param ref:
    :param mtbs:
    :param lba:
    :param frap:
    :param mine:
    :return:
    '''
    #read the csv
    if(type(table_data)==str):
        df = pd.read_csv(table_data)
    else:
        df = table_data

    #make a list of the columns
    col_list = [mtbs, lba, lba_bp, frap, mine]

    #check for masked values
    for col in col_list:
        mask_sum = (df[col]==10).sum()
        if(mask_sum > 0):
            print('{} has {} masked values'.format(col, mask_sum))

    ref_sum = (df[ref]==10).sum()
    if(ref_sum > 0 ):
        print('{} has {} masked values'.format(ref, ref_sum))


    #get the accuracy'
    #make an accuracy dictionary
    acc_dict = {}
    for col in col_list:
        table_accuracy = ((df[col] == df[ref]).sum() / 300) * 100
        #write it to the dict
        acc_dict[col] = table_accuracy

    #Get the omission errors
    #make an omission dictionart
    om_dict = {}
    for col in col_list:
        #make a list of sorts
        om_err = df[(df[ref] == 1) & (df[col]==0)].count()
        #get the column number from the list and turn it into omission rate
        om_err_rate = (om_err[col] / 300) * 100
        #add it to the dictionary
        om_dict[col] = om_err_rate

    #get the comission errors
    #make an commission dictionart
    com_dict = {}
    for col in col_list:
        #make a list of sorts
        com_err = df[(df[ref] == 0) & (df[col]==1)].count()
        #get the column number from the list and turn it into omission rate
        com_err_rate = (com_err[col] / 300) * 100
        #add it to the dictionary
        com_dict[col] = com_err_rate

    #return the results
    result_list = []
    # result_dict['accuracy'] = acc_dict
    # result_dict['omission'] = om_dict
    # result_dict['commission'] = com_dict

    #append the dicts to the list
    result_list.append(acc_dict)
    result_list.append(om_dict)
    result_list.append(com_dict)

    #make a df with the results
    out_df = pd.DataFrame(result_list, ['accuracy', 'omission', 'commission'], [mtbs, lba, lba_bp, frap, mine])

    #rename the columns to something reasonable
    out_df = out_df.rename(columns={mtbs: 'MTBS', lba: 'LBA_BC', lba_bp: 'LBA_BP', frap: 'FRAP', mine: 'L-GBM'})
    # print(out_df)

    if (out_path == None):
        return out_df
    else:
        out_df.to_csv(out_path, )

#=======================================================================================================================
#                                Private/Public Functions to read in the data from Quiz 3
#=======================================================================================================================
if __name__ == '__main__':
    df = pd.read_csv(r"D:\my_burned_maps\accuracy_assessment\updated_aa\t80\cedar_t80.csv")
    for col in df.columns:
        print(col)
    ref_col = 'ref_t50_ce'
    mtbs_col = 'X'
    lba_col = 'X'
    lba_bp_col = 'lba_bp_t80'
    frap_col = 'X'
    my_col = 'X'
    out_data = r"D:\my_burned_maps\accuracy_assessment\updated_aa\out_accuracies\cedar_accuracies_t80.csv"
    accuracy_info(df, ref_col, mtbs_col, lba_col, lba_bp_col, frap_col, my_col, out_data)

    # print(final_df)
    #print the output data
    # print('Accuracies: {} \n\nOmission: {} \n\nCommission: {}'.format(out_dict['accuracy'], out_dict['omission'],
    #                                                                   out_dict['commission']))

