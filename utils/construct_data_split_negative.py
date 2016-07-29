import openslide
from openslide.deepzoom import DeepZoomGenerator
from skimage.filters import threshold_otsu
import numpy as np
from skimage.color import rgb2gray
from multiprocessing import Pool
import skimage.transform
import scipy.io
import numpy.random
#from scipy.ndimage.filters import gaussian_filter
import os
import common_module

global patch_folder
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
#  for row in range(tile_count[0]):#
#    for col in range(tile_count[1]):
#      tile = dz_wsi_img.get_tile(tile_index,(row,col))
      #print tile.size
      
def get_save_dir():
  global patch_folder
  global intem_folder
  return patch_folder,intem_folder
  #global SAVE_DIR
  #return os.path.join(SAVE_DIR,'neg'), \
  #    os.path.join(SAVE_DIR,'tissue_area')
  
#def mkdir_if_not_exists():
  # directory to save image patches
  #global SAVE_DIR
#  save_folder_neg,tissue_area_folder = get_save_dir()
#  if not os.path.exists(save_folder_neg):
#    os.makedirs(save_folder_neg)
#  if not os.path.exists(tissue_area_folder):
#    os.makedirs(tissue_area_folder)
  
def single_WSI_processor(wsi_img):
  # directory to save image patches
  #save_folder_neg,tissue_area_folder = get_save_dir()
  patch_folder,intem_folder = get_save_dir()
  
  # load whole slide image and get the global otsu threshold
  f = openslide.OpenSlide(wsi_img)
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
  #resized_gray = gaussian_filter(resized_gray,sigma = 7)
  
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
  save_file_name = os.path.join(intem_folder,save_file_name)
  np.save(save_file_name,potential_list)  
  save_file_name = os.path.basename(wsi_img)[:-4]+'_label.npy'
  save_file_name = os.path.join(intem_folder,save_file_name)
  np.save(save_file_name,np.zeros((len(potential_list),)))
  
  #purged_patches = []      
  # purge all the purges
  #for address in potential_list:
  #  tile = dz.get_tile(tile_index,address)
  #  img = tile.convert('RGB')
  #  img = np.array(img)
  #  img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
  #  if np.std(img_gray) > 6:#len(img_gray[img_gray<threshold_global_otsu]) >256*256/10.0 and 
  #    purged_patches.append(address)
  purged_patches = common_module.get_purged_patches(dz,potential_list,tile_index)
  
  # get the negtive examples and save them to the disk
  wsi_pre = os.path.basename(wsi_img)[:-4]
  wsi_pre = os.path.join(patch_folder,wsi_pre)
  index = numpy.random.permutation(len(purged_patches))
  #txt_file = open('train1.txt','w')
  choose_size = 1000 
  if len(purged_patches) < choose_size:
    choose_size = len(purged_patches)
  for ind in range(choose_size):
    address = purged_patches[index[ind]]
    tile = dz.get_tile(tile_index,address)
    tile_name = wsi_pre + '_%d_%d.tif' %(address[0],address[1])
    tile.save(tile_name)
  print 'potential size: %d/ %d/ %s' % (len(purged_patches),len(potential_list),os.path.basename(wsi_img))
  
class BatchProcessor(object):
  def __init__(self, func):
    self.func = func
    
#def read_txt_to_list(txt_file):
#  f = open(txt_file,'r')
#  txt_list = f.readlines()
#  return_list = []
#  for one_file in txt_list:
#    return_list.append(one_file.strip())
#  return return_list

  
# randomly select 1,000 patches from 
def construct(patch_folder_neg,intem_dir,wsi_imgs,pool_size):
  global patch_folder
  global intem_folder
  #global SAVE_DIR
  patch_folder = patch_folder_neg
  intem_folder = intem_dir
  #mkdir_if_not_exists()
  pool = Pool(processes=pool_size)
  
  batch_processor = BatchProcessor(single_WSI_processor)
  pool.map(batch_processor.func,wsi_imgs)
  
  
if __name__=='__main__':
  global SAVE_DIR
  TXT_FILE = 'fold_dir/train_normal1.txt'
  SAVE_DIR = '/data/pathology/CAMELYON16_EXAMPLE/train/'
  POOL_SIZE = 20
  
  mkdir_if_not_exists()
  image_files = read_txt_to_list(TXT_FILE)
  #print "number of files: %d" %len(image_files)
  pool = Pool(processes=POOL_SIZE)
  
  batch_processor = BatchProcessor(single_WSI_processor)
  pool.map(batch_processor.func,image_files)
