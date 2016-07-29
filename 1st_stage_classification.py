import numpy as np
from construct_training_data_stage_1 import get_txt_files
from construct_training_data_stage_1 import read_txt_to_list
from construct_training_data_stage_1 import get_save_folder
from skimage.measure import label, regionprops
from sklearn.ensemble import RandomForestClassifier
import scipy.ndimage as ndimage
import os
import openslide
from openslide.deepzoom import DeepZoomGenerator

def generate_prob_map(wsi,intem_folder):
  f = openslide.OpenSlide(wsi)
  dz = DeepZoomGenerator(f,256,0,True)
  
  tile_index = dz.level_count-1
  tile_count = dz.level_tiles[tile_index]
  prob_map = np.zeros((tile_count[1],tile_count[0]),np.float32)
  basename = os.path.basename(wsi)
  
  tissue_file = basename[:-4] + '_tissue.npy'
  tissue_file = os.path.join(intem_folder,tissue_file)
  pred_file = basename[:-4] + '_predicted.npy'
  pred_file = os.path.join(intem_folder,pred_file)
  
  potential_list = np.load(tissue_file)
  predicted = np.load(pred_file)
  label = 0
  if basename[0] == 't' or basename[0] == 'T':
    label = 1
    
  for idx,address in enumerate(potential_list):
    prob_map[address[1],address[0]] = predicted[idx]
    
  return prob_map,label,len(potential_list)
  
def calculate_area(bbox_list):
  min_row, min_col, max_row, max_col = bbox_list
  return (max_row-min_row)*(max_col-min_col)  
  
def calculate_feature(prob_map,tissue_area):
  t1 = prob_map > 0.5
  t2 = prob_map > 0.9
  t1.dtype = np.uint8
  t2.dtype = np.uint8
  t1_label_image = label(t1,connectivity=t1.ndim)
  t2_label_image = label(t2,connectivity=t2.ndim)
  props_t1 = regionprops(t1_label_image,prob_map)
  props_t2 = regionprops(t2_label_image)
  #props
  
  feature = np.zeros((35,),np.float32)
  # the area of the largest tumor region
  if np.sum(t2) == 0:
    return feature
  feature_04 = np.sum(t2)/tissue_area
  index = np.argmax([props_t1[i].area for i in range(len(props_t1))])
  feature_05 = props_t1[index].area
  feature_08 = props_t1[index].eccentricity
  feature_09 = np.sum(t1)/sum([calculate_area(props_t1[i].bbox) \
      for i in range(len(props_t1))])
  feature_10 = props_t1[index].major_axis_length
  
  #feature_02 = props_t1[index].equivalent_diameter
  #feature_03 = props_t1[index].euler_number
  #[[feature_06,feature_07];[feature_11,feature_12]] = props_t1[index].inertia_tensor
  #feature_13,feature_14 = props_t1[index].inertia_tensor_eigvals
  #feature_15 = props_t1[index].minor_axis_length
  
  feature[0] = feature_04
  feature[1] = feature_05
  feature[2] = feature_08
  feature[3] = feature_09
  feature[4] = feature_10
  feature[5] = props_t1[index].convex_area
  feature[6] = props_t1[index].equivalent_diameter
  ind = 7
  feature[ind:ind+4] = props_t1[index].inertia_tensor.ravel()
  ind +=4 #11
  feature[ind] = props_t1[index].euler_number
  ind +=1
  feature[ind:ind+2] = props_t1[index].inertia_tensor_eigvals
  ind +=2 # 14
  feature[ind] = props_t1[index].minor_axis_length
  ind +=1
  moments_feature = props_t1[index].weighted_moments_normalized.ravel()
  feature[ind:ind+2] = moments_feature[2:4]
  ind +=2
  feature[ind:ind+11] = moments_feature[5:]
  ind +=11
  feature[ind:ind+7] = props_t1[index].weighted_moments_hu
  
  #print feature
  return feature
  
def from_wsis_to_features(wsis,intem_folder):
  num_example = len(wsis)
  features = np.zeros((num_example,35),np.float32)
  labels = np.zeros((num_example,),np.float32)
  
  for idx,wsi in enumerate(wsis):
    prob_map,label,tissue_area = generate_prob_map(wsi,intem_folder)
    prob_map = ndimage.gaussian_filter(prob_map, 0.5)
    features[idx] = calculate_feature(prob_map,tissue_area)
    labels[idx] = label
    
  return features,labels

if __name__=='__main__':
  # fill in the tumor and normal whole slide image directory here
  SAVE_DIR = '/data/pathology/CAMELYON16_EXAMPLE'
  _,_,train_norm_wsis,val_norm_wsis,\
      train_tumu_wsis, val_tumu_wsis = get_txt_files(SAVE_DIR,5)
  _,_,_,_,_,intem_folder,_ = get_save_folder(SAVE_DIR)
      
  train_wsis = read_txt_to_list(train_norm_wsis[0])
  train_wsis.extend(read_txt_to_list(train_tumu_wsis[0]))
  test_wsis = read_txt_to_list(val_norm_wsis[0])
  test_wsis.extend(read_txt_to_list(val_tumu_wsis[0]))
  
  print "Generating features for training WSIs"
  X_train,train_label = from_wsis_to_features(train_wsis,intem_folder)
  print "Generating features for testing WSIs"
  X_test,test_label = from_wsis_to_features(test_wsis,intem_folder)
  
  rfc = RandomForestClassifier(n_estimators=140,max_features=35)
  rfc.fit(X_train,train_label)
  predicted = rfc.predict(X_test)
  accu = np.sum(predicted == test_label)/float(len(predicted))
  print accu
  
  print train_norm_wsis[0]
  print train_tumu_wsis[0]
  print val_norm_wsis[0]
  print val_tumu_wsis[0]