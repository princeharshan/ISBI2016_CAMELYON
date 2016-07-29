import glob
import os
from utils import construct_data_split_negative
from utils import construct_data_split_positive
from sklearn.cross_validation import KFold
from random import shuffle

def write_list(file_list,file_index,txt_file):
  f = open(txt_file,'w')
  for index in file_index:
    f.write(file_list[index]+'\n')
  f.close()
  
def read_txt_to_list(txt_file):
  f = open(txt_file,'r')
  all_files = f.readlines()
  
  clean_files = []
  for one_file in all_files:
    clean_files.append(one_file.strip())
  return clean_files

# split the training data into several folds and write them into txt files
def split_wsis(all_tumor_wsi_imgs,all_normal_wsi_imgs,txt_folder,fold_num):
  #tumor_wsis = glob.glob(os.path.join(tumor_dir,'*.tif'))
  #normal_wsis = glob.glob(os.path.join(normal_dir,'*.tif'))
  kf_tumor = KFold(len(all_tumor_wsi_imgs),n_folds = fold_num)
  kf_normal = KFold(len(all_normal_wsi_imgs),n_folds = fold_num)
  tumor_train_list = []
  tumor_val_list = []
  normal_train_list = []
  normal_val_list = []
  
  count = 1
  for tumor_train, tumor_val in kf_tumor:
    train_txt = os.path.join(txt_folder,'train_tumor_wsis%d.txt' %count)
    val_txt = os.path.join(txt_folder,'val_tumor_wsis%d.txt' %count)
    
    write_list(all_tumor_wsi_imgs,tumor_train,train_txt)
    write_list(all_tumor_wsi_imgs,tumor_val,val_txt)
    tumor_train_list.append(train_txt)
    tumor_val_list.append(val_txt)
    
    #print("TRAIN:", tumor_index, "TEST,", test_index)
    count += 1
    
  count = 1
  for normal_train, normal_val in kf_normal:
    train_txt = os.path.join(txt_folder,'train_normal_wsis%d.txt' %count)
    val_txt = os.path.join(txt_folder,'val_normal_wsis%d.txt' %count)
    
    write_list(all_normal_wsi_imgs,normal_train,train_txt)
    write_list(all_normal_wsi_imgs,normal_val,val_txt)
    normal_train_list.append(train_txt)
    normal_val_list.append(val_txt)
    
    #print("TRAIN:", tumor_index, "TEST,", test_index)
    count += 1  
    
  return tumor_train_list, tumor_val_list, normal_train_list, normal_val_list

def write_train_val_patches_to_txts(tumor_train_list, tumor_val_list, \
      normal_train_list, normal_val_list, patch_folder_pos,\
      patch_folder_neg, write_dir):
  def get_traini_wsi_from_txts(tumor_train,normal_train,write_txt,):
    f_tumor = open(tumor_train,'r')
    f_normal = open(normal_train,'r')
    all_files = f_tumor.readlines()
    all_files.extend(f_normal.readlines())
    f_tumor.close()
    f_normal.close()
    temp_files = []
    for one_file in all_files:
      temp_files.append(one_file.strip())
      
    training_patches_pos = []
    training_patches_neg = []
    for one_file in temp_files:
      wsi_img = os.path.basename(one_file)[:-4]
      #search_pos = os.path.join(patch_dir,'pos')
      search_pos = os.path.join(patch_folder_pos,wsi_img)+'*'
      #search_neg = os.path.join(patch_dir,'neg')
      search_neg = os.path.join(patch_folder_neg,wsi_img)+'*'
      training_patches_pos.extend(glob.glob(search_pos))
      training_patches_neg.extend(glob.glob(search_neg))
      
    temp_files = []
    for one_file in training_patches_pos:
      temp_files.append(one_file + ' 1\n')
    for one_file in training_patches_neg:
      temp_files.append(one_file + ' 0\n')
    shuffle(temp_files)
    
    f_id = open(write_txt,'w')
    for one_file in temp_files:
      f_id.write(one_file)
    f_id.close()
      
#  if not os.path.exists(txt_folder):
#    os.makedirs(txt_folder)   

  for idx in range(len(tumor_train_list)):
    tumor_train_txt = tumor_train_list[idx]
    normal_train_txt = normal_train_list[idx]
    tumor_val_txt = tumor_val_list[idx]
    normal_val_txt = normal_val_list[idx]
    
    txt_file = os.path.join(write_dir,'train%d.txt' %(1+idx))
    get_traini_wsi_from_txts(tumor_train_txt,normal_train_txt,txt_file)
    txt_file = os.path.join(write_dir,'val%d.txt' %(1+idx))
    get_traini_wsi_from_txts(tumor_val_txt,normal_val_txt,txt_file)
  
 
def get_training_dirs(train_folder):
  pos_wsi_dir = os.path.join(train_folder,'TrainingData/Train_Tumor')
  neg_wsi_dir = os.path.join(train_folder,'TrainingData/Train_Normal')
  xml_dir     = os.path.join(train_folder,'TrainingData/Ground_Truth/XML')
  mask_dir    = os.path.join(train_folder,'TrainingData/Ground_Truth/Mask')
  
  return pos_wsi_dir, neg_wsi_dir, xml_dir, mask_dir

def get_model_files(train_folder,n_folds,iter_num):
  train_folder = os.path.join(train_folder,'neural_nets')
  dep_file = os.path.join(train_folder,'deploy.prototxt')
  model_files = [os.path.join(train_folder,'fold%d_iter_%d.caffemodel' \
             %(i+1,iter_num)) for i in range(n_folds)]
  train_val_files = [os.path.join(train_folder,'train_val%d.prototxt' \
             %(i+1)) for i in range(n_folds)]
  solver_files = [os.path.join(train_folder,'solver%d.prototxt' \
             %(i+1)) for i in range(n_folds)]
  
  return dep_file, model_files, train_val_files, solver_files
  
# file directories:
#           folder for txt
#           folder for stage 1 patches
#                     folder for negative patches
#                     folder for positive patches
#           folder for boosting patches
#                     folder for positive patches
#                     folder for negative patches
#           folder for intemediate results
#           folder for neural nets  

def get_txt_files(train_folder,n_folds):
  train_folder = os.path.join(train_folder,'txt_files')
  train_patch_files = \
    [os.path.join(train_folder,'train%d.txt'%(1+i)) for i in range(n_folds)]
  val_patch_files = \
    [os.path.join(train_folder,'val%d.txt'%(1+i)) for i in range(n_folds)]
  train_norm_wsis = \
    [os.path.join(train_folder,'train_normal_wsis%d.txt'%(1+i)) for i in range(n_folds)]
  val_norm_wsis = \
    [os.path.join(train_folder,'val_normal_wsis%d.txt'%(1+i)) for i in range(n_folds)]
  train_tumu_wsis = \
    [os.path.join(train_folder,'train_tumor_wsis%d.txt'%(1+i)) for i in range(n_folds)]
  val_tumu_wsis = \
    [os.path.join(train_folder,'val_tumor_wsis%d.txt'%(1+i)) for i in range(n_folds)]
    
  return train_patch_files,val_patch_files,train_norm_wsis,val_norm_wsis,\
      train_tumu_wsis, val_tumu_wsis
  
def get_save_folder(save_dir):
  txt_folder = os.path.join(save_dir,'txt_files')
  patch_folder = os.path.join(save_dir,'1stage_patches')
  patch_folder_pos = os.path.join(patch_folder,'pos')
  patch_folder_neg = os.path.join(patch_folder,'neg')
  patch_folder_boost = os.path.join(save_dir,'boost_patches')
  patch_folder_boost_pos = os.path.join(patch_folder_boost,'pos')
  patch_folder_boost_neg = os.path.join(patch_folder_boost,'neg')
  intem_folder = os.path.join(save_dir,'interm')
  neural_folder = os.path.join(save_dir,'neural_nets')
  
  if not os.path.exists(txt_folder):
    os.makedirs(txt_folder)  
  if not os.path.exists(patch_folder_pos):
    os.makedirs(patch_folder_pos)  
  if not os.path.exists(patch_folder_neg):
    os.makedirs(patch_folder_neg)  
  if not os.path.exists(patch_folder_boost_pos):
    os.makedirs(patch_folder_boost_pos)  
  if not os.path.exists(patch_folder_boost_neg):
    os.makedirs(patch_folder_boost_neg) 
  if not os.path.exists(intem_folder):
    os.makedirs(intem_folder)
  if not os.path.exists(neural_folder):
    os.makedirs(neural_folder)
  
  return txt_folder,patch_folder_pos,patch_folder_neg,\
    patch_folder_boost_pos,patch_folder_boost_neg,\
    intem_folder,neural_folder
    
if __name__=='__main__':
  # fill in the tumor and normal whole slide image directory here
  CAMELYON_FOLDER = '/data/pathology/CAMELYON16'
  SAVE_DIR = '/data/pathology/CAMELYON16_EXAMPLE'
  pos_wsi_dir, neg_wsi_dir, xml_dir, mask_dir = get_training_dirs(CAMELYON_FOLDER)
  
  txt_folder,patch_folder_pos,patch_folder_neg,\
    patch_folder_boost_pos,patch_folder_boost_neg,\
    intem_folder,neural_folder = get_save_folder(SAVE_DIR)
  num_processors = 20
  fold_num = 5
  
  # select patches from normal whole slide images
  all_normal_wsi_imgs = glob.glob(os.path.join(neg_wsi_dir,'*.tif'))
  construct_data_split_negative.construct(patch_folder_neg,intem_folder,\
    all_normal_wsi_imgs,num_processors)
  
  #select positive and negative patches from tumor whole slide images
  all_tumor_wsi_imgs = glob.glob(os.path.join(pos_wsi_dir,'*.tif'))
  construct_data_split_positive.construct(patch_folder_neg,patch_folder_pos,intem_folder,\
    all_tumor_wsi_imgs,xml_dir,mask_dir,num_processors)

  tumor_train_list, tumor_val_list, normal_train_list, normal_val_list = \
    split_wsis(all_tumor_wsi_imgs,all_normal_wsi_imgs,txt_folder,fold_num)
  
  write_train_val_patches_to_txts(tumor_train_list, tumor_val_list, \
      normal_train_list, normal_val_list, patch_folder_pos,patch_folder_neg, txt_folder)