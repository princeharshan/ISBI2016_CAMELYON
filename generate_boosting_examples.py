from construct_training_data_stage_1 import get_save_folder 
from construct_training_data_stage_1 import get_txt_files 
from construct_training_data_stage_1 import read_txt_to_list 
from construct_training_data_stage_1 import get_model_files
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os
import numpy as np
from train_final_classifier import batch_predict

def get_prediction(wsi,intem_folder,dep_file, model_file):
  f = openslide.OpenSlide(wsi)
  dz = DeepZoomGenerator(f,256,0,True)
  tile_index = dz.level_count-1
  tile_count = dz.level_tiles[tile_index]
  
  address_file = os.path.basename(wsi)[:-4]+'_tissue.npy'
  address_file = os.path.join(intem_folder,address_file)
  potential_list = np.load(address_file)
  predicted = batch_predict(potential_list,dz,tile_index,dep_file, model_file)
  
  predicted_file = os.path.basename(wsi)[:-4]+'_predicted.npy'
  predicted_file = os.path.join(intem_folder,predicted_file)
  np.save(predicted_file,predicted)
  #print predicted
  

if __name__=='__main__':
  

  #train_normal = 'txt_files/train_normal_wsis1.txt'
  #interm_dir = '/data/pathology/CAMELYON16_EXAMPLE/interm'
  ##mask_dir = '/data/pathology/CAMELYON16/TrainingData/Ground_Truth/Mask'
  #f_normal = open(train_normal,'r')
  #neg_wsi_img = f_normal.readlines()
  #f_normal.close()
  #
  #file_name = neg_wsi_img[0].strip()
  #f = openslide.OpenSlide(file_name)
  #dz = DeepZoomGenerator(f,256,0,True)
  #tile_index = dz.level_count-1
  #tile_count = dz.level_tiles[tile_index]
  #
  ##potential_list = get_all_possible_patches(f,dz)
  ##potential_list = np.array(potential_list)
  #label_file = os.path.basename(file_name)[:-4]
  #label_file = os.path.join(interm_dir,label_file)
  #label_file += '_label.npy'
  #tissue_file = label_file[:-9]+'tissue.npy'
  #potential_list = np.load(tissue_file)
  #ground_truth = np.load(label_file)
  #predicted = batch_predict(potential_list,dz,tile_index)
  #
  #print predicted > 0.5
  #
  #print np.sum(ground_truth)
  #print predicted 
  #predicted = predicted > 0.5
  #predicted = np.array(predicted,np.int)
  #print np.sum(ground_truth!=predicted)
  
  
  
  #ground_truth = np.empty((len(potential_list),),np.float32)
  #print neg_wsi_img[0].strip()
  #if os.path.basename(neg_wsi_img[0])[0] == 'N' or \
  #    os.path.basename(neg_wsi_img[0])[0] == 'n':
  #  ground_truth = np.zeros((len(potential_list),),np.float32)
  #else:
  #  print "Processing the positive WSI"
  #  file_name = os.path.basename(file_name)
  # f_mask = os.path.join(mask_dir,file_name[:-4]+'_Mask.tif')
  #  f_mask = openslide.OpenSlide(f_mask)
  #  dz_mask = DeepZoomGenerator(f_mask,256,0,True)
  #  ground_truth = generate_labels(dz,dz_mask,potential_list,tile_index)
  #  print ground_truth
  
  SAVE_DIR = '/data/pathology/CAMELYON16_EXAMPLE'
  _,_,_,_,_,intem_folder,_ = get_save_folder(SAVE_DIR)
  _,_,train_norm_wsis,val_norm_wsis,\
      train_tumu_wsis, val_tumu_wsis = get_txt_files(SAVE_DIR,5)
  dep_file, model_files, _, _ = \
      get_model_files(SAVE_DIR,5,40000)
      
  all_wsis = read_txt_to_list(train_norm_wsis[0])
  all_wsis.extend(read_txt_to_list(val_norm_wsis[0]))
  all_wsis.extend(read_txt_to_list(train_tumu_wsis[0]))
  all_wsis.extend(read_txt_to_list(val_tumu_wsis[0]))
  
  for idx,one_file in enumerate(all_wsis):
    print "Processing the %dth/%d WSI" %(idx+1,len(all_wsis))
    get_prediction(one_file,intem_folder,dep_file, model_files[0])
  
