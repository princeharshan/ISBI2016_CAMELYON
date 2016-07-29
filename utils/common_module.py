import numpy as np
from skimage.filters import threshold_otsu
import skimage.transform

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
#  #img_gray = rgb2gray(np.array(img)) 
# img = np.array(img)
#  img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
#  return img_gray

def get_middle_size_img(wsi_img):
  img_size = wsi_img.level_dimensions[0]
  factor_w = img_size[0]/4000.0
  factor_h = img_size[1]/4000.0
  factor = factor_w
  if factor_h < factor:
    factor = factor_h
    
  level_index = wsi_img.get_best_level_for_downsample(factor)
  factor = 2**level_index
  size_x = int(np.floor(float(img_size[0])/factor + 0.5))
  size_y = int(np.floor(float(img_size[1])/factor + 0.5))
  
  img = wsi_img.read_region((0,0),level_index,\
      [size_x,size_y])
  img = img.convert('RGB')
  img = np.array(img)
  img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
  return img_gray
  
  

def compute_threshold(wsi_img):
  #img_gray = get_middle_size_img(wsi_img,0)
  img_gray = get_middle_size_img(wsi_img)
  threshold_global = threshold_otsu(img_gray)
  return threshold_global

# get all possible patches that lies in tissue regions
def get_all_possible_patches(f,dz):
  threshold_global_otsu = compute_threshold(f)
  # some of the whole slide image regions don't contain anything
  tile_index = dz.level_count-1
  tile_count = dz.level_tiles[tile_index]
  #img_gray = get_middle_size_img(f,2)
  img_gray = get_middle_size_img(f)
  #print img_gray.shape
  resized_gray = skimage.transform.resize(img_gray,(tile_count[1]*5,tile_count[0]*5))
  potential_list = []
  #print tile_count
  #print resized_gray.shape
  
  for row in range(tile_count[1]):
    for col in range(tile_count[0]):
      curr_img = resized_gray[row*5:(row+1)*5,col*5:(col+1)*5]
      if len(curr_img[curr_img<threshold_global_otsu]) >1:
        potential_list.append((col,row))  
  return potential_list
  
def get_purged_patches(dz,potential_list,tile_index):
  purged_patches = []      
  # purge all the purges
  for address in potential_list:
    tile = dz.get_tile(tile_index,address)
    img = tile.convert('RGB')
    img = np.array(img)
    img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
    if np.std(img_gray) > 6:#len(img_gray[img_gray<threshold_global_otsu]) >256*256/10.0 and 
      purged_patches.append(address)
  
  return purged_patches
  
def generate_labels(dz,dz_mask,potential_list,tile_index):
  #purged_patches = []
  #pos_patches = []
  #neg_patches = []  
  # purge all the purges
  ground_truth = np.zeros((len(potential_list),),np.float32)
  for idx, address in enumerate(potential_list):
    tile = dz.get_tile(tile_index,address)
    img = tile.convert('RGB')
    img = np.array(img)
    #img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
    #if np.std(img_gray) > 6:#len(img_gray[img_gray<threshold_global_otsu]) >256*256/10.0 and np.std(img_gray) > 0.05:
      #purged_patches.append(address)
    tile_mask = dz_mask.get_tile(tile_index,address)
    mask_img = np.array(tile_mask.convert('RGB'))[:,:,0]
    if len(mask_img[mask_img==255]) > 256*256/2:
      #pos_patches.append(address)
      ground_truth[idx] = 1
    else:
      ground_truth[idx] = 0
  return ground_truth 
  
def get_purged_patches_for_tumor_wsi(dz,dz_mask,potential_list,tile_index):
  purged_patches = []
  pos_patches = []
  neg_patches = []  
  labels = np.zeros((len(potential_list),))
  # purge all the purges
  for idx,address in enumerate(potential_list):
    tile = dz.get_tile(tile_index,address)
    img = tile.convert('RGB')
    img = np.array(img)
    img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
    
    tile_mask = dz_mask.get_tile(tile_index,address)
    mask_img = np.array(tile_mask.convert('RGB'))[:,:,0]
    if len(mask_img[mask_img==255]) > 256*256/2:
      pos_patches.append(address)
      labels[idx] = 1
    else:
      labels[idx] = 0
      if np.std(img_gray) > 6:
        neg_patches.append(address)
    
    #if np.std(img_gray) > 6:#len(img_gray[img_gray<threshold_global_otsu]) >256*256/10.0 and np.std(img_gray) > 0.05:
    #  #purged_patches.append(address)
    #  tile_mask = dz_mask.get_tile(tile_index,address)
    #  mask_img = np.array(tile_mask.convert('RGB'))[:,:,0]
    #  if len(mask_img[mask_img==255]) > 256*256/2:
    #    pos_patches.append(address)
    #  else:
    #    neg_patches.append(address)
  return pos_patches,neg_patches,labels