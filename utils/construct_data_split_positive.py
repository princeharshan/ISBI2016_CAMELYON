import glob
import Image
import openslide
from openslide.deepzoom import DeepZoomGenerator
from skimage.filters import threshold_otsu
import numpy as np
from skimage.color import rgb2gray
from multiprocessing import Pool
import skimage.transform
import scipy.io
import numpy.random
import os
import read_xml
import common_module

global MASK_DIR
global XML_DIR
global patch_folder_neg
global patch_folder_pos
global intem_folder

#def get_middle_size_img(wsi_img,offset):
#  level_index = wsi_img.level_count-1
#  img = wsi_img.read_region((0,0),level_index,\
#    wsi_img.level_dimensions[level_index])

#  while (img.size[0]<1000 or img.size[1]<1000) and\
#      img.size[0]<2000 and img.size[1]<2000 and level_index!=0:
#    level_index -= 1
#    img = wsi_img.read_region((0,0),level_index,\
#      wsi_img.level_dimensions[level_index])
    
#  level_index -= offset
#  img = wsi_img.read_region((0,0),level_index,\
#      wsi_img.level_dimensions[level_index]) 
#  img = img.convert('RGB')
  #img_gray = rgb2gray(np.array(img)) 
#  img = np.array(img)
#  img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
#  return img_gray

#def compute_threshold(wsi_img):
#  img_gray = get_middle_size_img(wsi_img,2)
#  threshold_global = threshold_otsu(img_gray)
#  return threshold_global

#def generate_training_examples(dz_wsi_img,threshold_global):
  #dz_wsi_img = DeepZoomGenerator(f,256,0,True)
#  tile_index = dz_wsi_img.level_count-1
#  tile_count = dz_wsi_img.level_tiles[tile_index]

  # generate the training examples
#  for row in range(tile_count[0]):
#    for col in range(tile_count[1]):
#      tile = dz_wsi_img.get_tile(tile_index,(row,col))
      #print tile.size

def get_save_dir():
  global MASK_DIR
  global XML_DIR
  global patch_folder_neg
  global patch_folder_pos
  global intem_folder
  return  MASK_DIR, XML_DIR, patch_folder_neg, \
        patch_folder_pos, intem_folder
  #global SAVE_DIR
  #return os.path.join(SAVE_DIR,'neg'),os.path.join(SAVE_DIR,'pos'),\
  #    os.path.join(SAVE_DIR,'tissue_area')

#def mkdir_if_not_exists():
#  # directory to save image patches
#  save_folder_neg,save_folder_pos,tissue_area_folder = \
#      get_save_dir()
#  
#  if not os.path.exists(save_folder_neg):
#    os.makedirs(save_folder_neg)
#  if not os.path.exists(save_folder_pos):
#    os.makedirs(save_folder_pos)
#  if not os.path.exists(tissue_area_folder):
#    os.makedirs(tissue_area_folder)
  
def single_WSI_processor(wsi_img):
  MASK_DIR, XML_DIR, save_folder_neg, \
        save_folder_pos, tissue_area_folder = get_save_dir()
  # directory to save image patches
  #global MASK_DIR
  #global XML_DIR
  #save_folder_neg,save_folder_pos,tissue_area_folder = \
  #    get_save_dir()
  wsi_mask = 'T'+os.path.basename(wsi_img)[1:-4]+'_Mask.tif'
  wsi_mask = os.path.join(MASK_DIR,wsi_mask)
  wsi_xml = os.path.basename(wsi_img)[:-4]+'.xml'
  xml_file = os.path.join(XML_DIR,wsi_xml)
  
  # load whole slide image and get the global otsu threshold
  f = openslide.OpenSlide(wsi_img)
  mask = openslide.OpenSlide(wsi_mask)
  dz_mask = DeepZoomGenerator(mask,256,0,True)
  dz = DeepZoomGenerator(f,256,0,True)
  #threshold_global_otsu = compute_threshold(f)
  #print threshold_global_otsu
  #generate_training_examples(dz,threshold_global_otsu)
  
  # some of the whole slide image regions don't contain anything
  tile_index = dz.level_count-1
  #tile_count = dz.level_tiles[tile_index]
  #img_gray = get_middle_size_img(f,2)
  #print img_gray.shape
  #resized_gray = skimage.transform.resize(img_gray,(tile_count[1]*5,tile_count[0]*5))
  #potential_list = []
  #print tile_count
  #print resized_gray.shape
  #for row in range(tile_count[1]):
  #  for col in range(tile_count[0]):
  #    curr_img = resized_gray[row*5:(row+1)*5,col*5:(col+1)*5]
  #    if len(curr_img[curr_img<threshold_global_otsu]) >1:
  #      potential_list.append((col,row))
  potential_list = common_module.get_all_possible_patches(f,dz) 
  save_file_name = os.path.basename(wsi_img)[:-4]+'_tissue.npy'
  save_file_name = os.path.join(tissue_area_folder,save_file_name)
  np.save(save_file_name,potential_list) 
        
  #purged_patches = []
  #pos_patches = []
  #neg_patches = []  
  # purge all the purges
  #for address in potential_list:
  #  tile = dz.get_tile(tile_index,address)
  #  img = tile.convert('RGB')
   # img = np.array(img)
  #  img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
  #  if np.std(img_gray) > 6:#len(img_gray[img_gray<threshold_global_otsu]) >256*256/10.0 and np.std(img_gray) > 0.05:
  #    #purged_patches.append(address)
  #    tile_mask = dz_mask.get_tile(tile_index,address)
  #   mask_img = np.array(tile_mask.convert('RGB'))[:,:,0]
  #   if len(mask_img[mask_img==255]) > 256*256/2:
  #      pos_patches.append(address)
  #    else:
  #      neg_patches.append(address)
  pos_patches,neg_patches,labels = common_module.get_purged_patches_for_tumor_wsi( \
          dz,dz_mask,potential_list,tile_index)
  
  save_file_name = os.path.basename(wsi_img)[:-4]+'_label.npy'
  save_file_name = os.path.join(tissue_area_folder,save_file_name)
  np.save(save_file_name,labels)
  
  # get the negtive examples and save them to the disk
  wsi_pre = os.path.basename(wsi_img)[:-4]
  wsi_pre_neg = os.path.join(save_folder_neg,wsi_pre)
  index = numpy.random.permutation(len(neg_patches))
  #txt_file = open('train1.txt','w')
  choose_size = 1000 
  if len(neg_patches) < choose_size:
    choose_size = len(neg_patches)
  for ind in range(choose_size):
    address = neg_patches[index[ind]]
    tile = dz.get_tile(tile_index,address)
    tile_name = wsi_pre_neg + '_%d_%d.tif' %(address[0],address[1])
    tile.save(tile_name)
  
  wsi_pre_pos = os.path.join(save_folder_pos,wsi_pre)
  index = numpy.random.permutation(len(pos_patches))
  choose_size = 1000
  final_choose = choose_size
  #if len(pos_patches) < choose_size:
  #  choose_size = len(pos_patches)
  #for ind in range(choose_size):
  #  address = pos_patches[index[ind]]
  #  tile = dz.get_tile(tile_index,address)
  #  tile_name = wsi_pre_pos + '_%d_%d.tif' %(address[0],address[1])
  #  tile.save(tile_name)
  #if len(pos_patches) < choose_size:
  #  read_xml.generate_possible_patches(xml_file,wsi_mask,wsi_img,wsi_pre_pos)
  #else:
  #  for ind in range(choose_size):
  #    address = pos_patches[index[ind]]
  #    tile = dz.get_tile(tile_index,address)
  #    tile_name = wsi_pre_pos + '_%d_%d.tif' %(address[0],address[1])
  #    tile.save(tile_name)
  if len(pos_patches) < choose_size:
    final_choose = len(pos_patches)
  for ind in range(final_choose):
    address = pos_patches[index[ind]]
    tile = dz.get_tile(tile_index,address)
    tile_name = wsi_pre_pos + '_%d_%d.tif' %(address[0],address[1])
    tile.save(tile_name)
  if final_choose < choose_size:
    extra_num = choose_size - final_choose
    read_xml.generate_possible_patches(xml_file,wsi_mask,wsi_img,wsi_pre_pos,extra_num)
  print 'potential size: %d/ %d/ %d/ %s' % (len(neg_patches),len(pos_patches),\
      len(potential_list),os.path.basename(wsi_img))
  
class BatchProcessor(object):
  def __init__(self, func):
    self.func = func

def read_txt_to_list(txt_file):
  f = open(txt_file,'r')
  txt_list = f.readlines()
  return_list = []
  for one_file in txt_list:
    return_list.append(one_file.strip())
  return return_list
    
def construct(patch_folder_neg_dir,patch_folder_pos_dir,intem_folder_dir,\
                           tumor_wsi_imgs,xml_dir,mask_dir,pool_size):
  #global SAVE_DIR
  global MASK_DIR
  global XML_DIR
  global patch_folder_neg
  global patch_folder_pos
  global intem_folder
  
  #SAVE_DIR = save_dir
  patch_folder_neg = patch_folder_neg_dir
  patch_folder_pos = patch_folder_pos_dir
  intem_folder = intem_folder_dir
  MASK_DIR = mask_dir
  XML_DIR = xml_dir
  
  #mkdir_if_not_exists()
  pool = Pool(processes=pool_size)
  
  batch_processor = BatchProcessor(single_WSI_processor)
  pool.map(batch_processor.func,tumor_wsi_imgs)
  
  
  
if __name__=='__main__':
  global SAVE_DIR
  global MASK_DIR
  global XML_DIR
  #DATA_DIR = '/data/pathology/CAMELYON16/TrainingData/Train_Tumor/'
  TXT_FILE = 'fold_dir/val_tumor1.txt'
  MASK_DIR = '/data/pathology/CAMELYON16/TrainingData/Ground_Truth/Mask'
  XML_DIR = '/data/pathology/CAMELYON16/TrainingData/Ground_Truth/XML'
  SAVE_DIR = '/data/pathology/CAMELYON16_EXAMPLE/val/'
  POOL_SIZE = 20
  
  mkdir_if()
  image_files = read_txt_to_list(TXT_FILE)
  #image_files = glob.glob(DATA_DIR + '*.tif')
  pool = Pool(processes=POOL_SIZE)
  
  batch_processor = BatchProcessor(single_WSI_processor)
  pool.map(batch_processor.func,image_files)
