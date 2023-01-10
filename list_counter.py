#list counter


#The goal of this script is to read in lists, and see what SVIs are used the most

#naming convention
#method_agg_samp%_threshold%

#The Datasets
imp_non_20_20 = ['dSAVI', 'dNDVI', 'VI43', 'dVI43','NDMI','VI45','NDVI','SAVI']
imp_non_20_50 = ['dSAVI','GEMI','dNDVI','SAVI','VI43','NDWI','NDVI','dNBR2']
imp_non_20_80 = ['GEMI','NDWI','dNBR2','SAVI','NDVI','VI43','CSI','dNDVI']
imp_non_10_20 = ['dNDVI','dVI43','NBR'  ,'VI43' ,'dSAVI','dVI45','GEMI' ,'dNDWI']
imp_non_10_50 = ['VI43','dSAVI','dNDVI','GEMI' ,'NDVI', 'dVI43','SAVI' ,'NBR']
imp_non_10_80 = ['NBR','dNDVI' ,'GEMI'  ,'VI43'  ,'NDVI'  ,'dSAVI' ,'SAVI'  ,'CSI']
imp_agg_20_20 = ['dNBR2','NBR2','dVI57','VI57','GEMI' ,'SAVI' ,'NDWI' ,'dVI46']
imp_agg_20_50 = ['dNBR2','dVI57','NBR2','VI57','GEMI','SAVI','NDWI','NDVI']
imp_agg_20_80 = ['dNBR2','GEMI' ,'dVI57','NBR2' ,'VI57' ,'SAVI' ,'NDWI' ,'NDVI' ]
#---------------
#Bad data, may not want to add
imp_agg_10_20 = ['NBR2_AVG','VI57_AVG','VI6T_AVG','VI46_AVG','VI43_STD','dNBR2'   ,'SAVI_STD','VI46_STD']
imp_agg_10_50 = ['VI46_AVG','VI6T_AVG' ,'NBR2_AVG' ,'NBRT1_STD','VI57_AVG' ,'VI43_STD' ,'NDVI_STD' ,'NDMI_STD']
imp_agg_10_80 = ['VI57_AVG','NBR2_AVG','NBR2','VI6T_AVG','VI46_AVG','dNBR2','VI43_STD','dVI57']
#End of Bad
#--------------------
fwd_non_20_20 = ['GEMI','MIRBI','GEMI_AVG','VI43_AVG','NDMI_AVG','NBRT1','dNBR2','dNDMI','dVI6T']
fwd_non_20_50 = ['VI43','NBR2_AVG','GEMI_AVG','NDMI_AVG','NBR2_STD','dMIRBI','dNBR','dNBR2','dVI57']
fwd_non_20_80 = ['NDWI','GEMI','GEMI_AVG','VI6T_AVG','VI46_AVG','EVI','EVI_STD','dNBR2','dVI45']
fwd_non_10_20 = ['GEMI','NBR','GEMI_STD','NDVI_STD','dMIRBI','dSAVI','dVI6T','dVI46','dVI57']
fwd_non_10_50 = ['GEMI','VI57','MIRBI','GEMI_AVG','NDMI_AVG','EVI_AVG','EVI_AVG','dNDMI','dVI46']
fwd_non_10_80 = ['VI57','CSI','GEMI','CSI_AVG','NBR2_AVG','VI6T_AVG','BAI_STD','dMIRBI','dNDVI']
fwd_agg_20_20 = ['BAI_AVG','NBR2_AVG','NDMI_AVG','NDVI_AVG','VI6T_AVG','SAVI_AVG','NDWI_STD','VI6T_STD','dVI57']
fwd_agg_20_50 = ['NBR2','EVI_AVG','NDMI_AVG','NDWI_AVG','VI46_AVG','VI57_AVG','MIRBI_STD','VI6T_STD','VI43_STD']
fwd_agg_20_80 = ['NDWI','NBRT1_AVG','SAVI_AVG','VI46_AVG','VI57_AVG','NDWI_STD','VI6T_STD','VI57_STD','dNBR2']
#---------------
#Below data mey be bad
fwd_agg_10_20 = ['BAI','NBR2','NBR2_AVG','NBRT1_AVG','VI46_AVG','NDVI_STD','VI6T_STD','dVI46']
fwd_agg_10_50 = ['CSI_AVG','GEMI_AVG','NBR2_AVG','VI6T_AVG','CSI_STD','VI43_STD','VI46_STD','VI57_STD']
fwd_agg_10_80 = ['GEMI','MIRBI_AVG','VI46_AVG','VI57_AVG','CSI_STD','NDMI_STD','NDWI_STD','VI46_STD','dEVI']
#End of bad data
#---------------

master_list = imp_non_20_20 + imp_non_20_50 + imp_non_20_80 + imp_non_10_20 +imp_non_10_50 + imp_non_10_80+\
imp_agg_20_20+ imp_agg_20_50+imp_agg_20_80 +imp_agg_10_20+imp_agg_10_50+imp_agg_10_80+fwd_non_20_20+ fwd_non_20_50+\
fwd_non_20_80+fwd_non_10_20+fwd_non_10_50+fwd_non_10_80+fwd_agg_20_20+fwd_agg_20_50+fwd_agg_20_80+fwd_agg_10_20+\
fwd_agg_10_50+fwd_agg_10_80


#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('Everything \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')

#----------------
#none of the bad data from the 10% or other stuff

master_list = imp_non_20_20 + imp_non_20_50 + imp_non_20_80 + imp_non_10_20 + imp_non_10_50 + imp_non_10_80+\
imp_agg_20_20+ imp_agg_20_50+imp_agg_20_80+fwd_non_20_20+ fwd_non_20_50+\
fwd_non_20_80+fwd_non_10_20+fwd_non_10_50+fwd_non_10_80+fwd_agg_20_20+fwd_agg_20_50+fwd_agg_20_80

#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('Without the 10% agg data \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')

#----------------
#----------------
#none of the bad data from the 10% or other stuff

master_list = imp_non_20_20 + imp_non_20_50 + imp_non_20_80 +\
imp_agg_20_20+ imp_agg_20_50+imp_agg_20_80+fwd_non_20_20+ fwd_non_20_50+\
fwd_non_20_80+fwd_agg_20_20+fwd_agg_20_50+fwd_agg_20_80

#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('20% sample All \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')

#----------------
#----------------
#none of the bad data from the 10% or other stuff

master_list = imp_agg_20_20 + imp_agg_20_50 + imp_agg_20_80

#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('20% sample Importance Aggregated \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')

#----------------
#----------------
#none of the bad data from the 10% or other stuff

master_list = fwd_agg_20_20 + fwd_agg_20_50 + fwd_agg_20_80

#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('20% sample Forward Aggregated \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')


#----------------
#----------------
#none of the bad data from the 10% or other stuff

master_list = imp_non_20_20 + imp_non_20_50 + imp_non_20_80

#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('20% sample Importance Not Aggregated \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')


#----------------
#----------------
#none of the bad data from the 10% or other stuff

master_list = imp_non_20_20 + imp_non_20_50 + imp_non_20_80 + imp_agg_20_20 + imp_agg_20_50 + imp_agg_20_80

#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('20% sample Importance Not Aggregated & Aggregated \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')


#----------------
#----------------
#none of the bad data from the 10% or other stuff

master_list = fwd_non_20_20 + fwd_non_20_50 + fwd_non_20_80 + fwd_agg_20_20 + fwd_agg_20_50 + fwd_agg_20_80

#Count how many times each item is repeated in the list
my_dict = {i:master_list.count(i) for i in master_list}

#order the dictionary by how many times it was used
my_dict_s = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

print('20% sample Forward Not Aggregated & Aggregated \n',my_dict_s)

#The amount of predictors
print(len(my_dict_s), '\n')

