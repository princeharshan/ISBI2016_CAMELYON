import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from skimage.filters import threshold_otsu
import skimage.transform
import time
from multiprocessing import Process
import multiprocessing as mp
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from multiprocessing import Process, Manager
from scipy.io import savemat
from skimage.measure import label, regionprops
from utils.common_module import get_all_possible_patches

#def get_middle_size_img(wsi_img,offset):
#  level_index = wsi_img.level_count-1
#  img = wsi_img.read_region((0,0),level_index,\
#    wsi_img.level_dimensions[level_index])
#
#  while (img.size[0]<1000 or img.size[1]<1000) and\
#      img.size[0]<2000 and img.size[1]<2000 and level_index!=0:
#    level_index -= 1
#    img = wsi_img.read_region((0,0),level_index,\
#      wsi_img.level_dimensions[level_index])
#    
#  level_index -= offset
#  img = wsi_img.read_region((0,0),level_index,\
#      wsi_img.level_dimensions[level_index]) 
#  img = img.convert('RGB')
#  #img_gray = rgb2gray(np.array(img)) 
#  img = np.array(img)
#  img_gray = 0.4*img[:,:,0]+0.2*img[:,:,1]+0.4*img[:,:,2]
#  return img_gray
#
#def compute_threshold(wsi_img):
#  img_gray = get_middle_size_img(wsi_img,0)
#  threshold_global = threshold_otsu(img_gray)
#  return threshold_global
#
## get all possible patches that lies in tissue regions
#def get_all_possible_patches(f,dz):
#  threshold_global_otsu = compute_threshold(f)
#  # some of the whole slide image regions don't contain anything
#  tile_index = dz.level_count-1
#  tile_count = dz.level_tiles[tile_index]
#  img_gray = get_middle_size_img(f,2)
#  #print img_gray.shape
#  resized_gray = skimage.transform.resize(img_gray,(tile_count[1]*5,tile_count[0]*5))
#  potential_list = []
#  #print tile_count
#  #print resized_gray.shape
#  
#  for row in range(tile_count[1]):
#    for col in range(tile_count[0]):
#      curr_img = resized_gray[row*5:(row+1)*5,col*5:(col+1)*5]
#      if len(curr_img[curr_img<threshold_global_otsu]) >1:
#        potential_list.append((col,row))  
#  return potential_list
  
def classify_with_1_gpu(dz,tile_index,address_list,gpu_id,model_def,model_weights,final_result):
  #Confiugre here
  batch_size = 340

  caffe.set_mode_gpu()
  caffe.set_device(gpu_id)
  net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
  mu = np.zeros((3,))
  mu[0] = 170
  mu[1] = 131
  mu[2] = 171
  
  HEIGHT = 224
  WIDTH = 224
  shape = (1,3,HEIGHT,WIDTH)

  transformer = caffe.io.Transformer({'data':shape})
  transformer.set_mean('data',mu)
  #transformer.set_raw_scale('data',255)
  transformer.set_channel_swap('data',(2,1,0))
  transformer.set_transpose('data',(2,0,1))
  net.blobs['data'].reshape(batch_size,shape[1],shape[2],shape[3])
  
  caffe_in = np.zeros((batch_size,3,224,224))
  num_fold = address_list.shape[0]
  result = np.zeros((num_fold,),np.float32)
  round = num_fold / batch_size
  if num_fold % batch_size != 0:
    round += 1
  for i in range(round):
    #print "The %dth round" % i    
    #start = time.clock()
    start_pos = i*batch_size
    if i==round-1:
      start_pos = num_fold - batch_size
    for j in range(batch_size):
      tile = dz.get_tile(tile_index,address_list[start_pos+j])
      img = tile.convert('RGB')
      img = np.array(img)
      temp_caffe_in = caffe.io.oversample([img],[224,224])[0]
      #print temp_caffe_in.shape
      caffe_in[j] = transformer.preprocess('data',temp_caffe_in)
    #end = time.clock()
    #runtime = end - start
    #print "Running time %d" % runtime
    #caffe_in = transformer.preprocess('data',caffe_in)
    #print caffe_in.shape
    #start = time.clock()
    net.blobs['data'].data[...] = caffe_in
    result[start_pos:start_pos+batch_size] = net.forward()['prob'][:,1]
    #end = time.clock()
    #runtime = end - start
  #final_result.put((gpu_id,result))
  final_result[gpu_id] = result
  
def batch_predict(potential_list,dz,tile_index,model_def,model_weights):
  #potential_list = get_all_possible_patches(f,dz)
  #potential_list = np.array(potential_list)

  processes = []
  #final_result = mp.Queue()
  manager = Manager()
  return_dict = manager.dict()
  fold_num = 7
  num_patches = potential_list.shape[0]
  fold_count = num_patches/fold_num
  
  for gpu_id in range(7):
    start_pos = gpu_id*fold_count
    end_pos = (gpu_id+1)*fold_count
    if gpu_id == fold_num-1:
      end_pos = num_patches
    one_fold = potential_list[start_pos:end_pos]
    processes.append(Process(target=classify_with_1_gpu,args=\
      (dz,tile_index,one_fold,gpu_id+2,model_def,model_weights,return_dict)))
    #process.start()
  [process.start() for process in processes]
    
  for process in processes:
    process.join()
    
  #results = [final_result.get() for p in processes]
  #results.sort()
  #results = return_dict.values()
  result = np.zeros((num_patches),np.float32)
  start_pos = 0
  for gpu_id in range(7):
    one_result = return_dict[gpu_id+2]
    end_pos = start_pos + one_result.shape[0]
    result[start_pos:end_pos] = one_result
    start_pos += one_result.shape[0]
    
  return result
  
def generate_prob_map(wsi_img):#,net,transformer,batch_size):
  f = openslide.OpenSlide(wsi_img)
  dz = DeepZoomGenerator(f,256,0,True)
  tile_index = dz.level_count-1
  tile_count = dz.level_tiles[tile_index]
  
  potential_list = get_all_possible_patches(f,dz)
  potential_list = np.array(potential_list)

  #processes = []
  #final_result = mp.Queue()
  #manager = Manager()
  #return_dict = manager.dict()
  #fold_num = 7
  #num_patches = potential_list.shape[0]
  #fold_count = num_patches/fold_num
  
  #for gpu_id in range(7):
  #  start_pos = gpu_id*fold_count
  #  end_pos = (gpu_id+1)*fold_count
  #  if gpu_id == fold_num-1:
  #    end_pos = num_patches
  #  one_fold = potential_list[start_pos:end_pos]
  #  processes.append(Process(target=classify_with_1_gpu,args=\
  #    (dz,tile_index,one_fold,gpu_id+2,return_dict)))
    #process.start()
  #[process.start() for process in processes]
    
  #for process in processes:
  #  process.join()
    
  #results = [final_result.get() for p in processes]
  #results.sort()
  #results = return_dict.values()
  #result = np.zeros((num_patches),np.float32)
  #start_pos = 0
  #for gpu_id in range(7):
  #  one_result = return_dict[gpu_id+2]
  #  end_pos = start_pos + one_result.shape[0]
  #  result[start_pos:end_pos] = one_result
  #  start_pos += one_result.shape[0]
  #print(result)
  #print (len(result))
  
  model_def = 'model_files/deploy.prototxt'
  model_weights = 'model_files/bvlc_googlenet_iter_40000.caffemodel'
  
  result = batch_predict(potential_list,dz,tile_index,model_def,model_weights)
  
  final_map = np.zeros((tile_count[1],tile_count[0]),np.float32)
  num_patches = result.shape[0]
  for i in range(num_patches):
    final_map[potential_list[i,1],potential_list[i,0]]=result[i]
      
  return final_map,num_patches
  #savemat('example.mat',{'probability':final_map})
  
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
  
  
if __name__=='__main__':
  train_normal = 'txt_files/train_normal_wsis1.txt'
  train_tumor = 'txt_files/train_tumor_wsis1.txt'
  
  # read WSI file names from txt file
  f_normal = open(train_normal,'r')
  f_tumor = open(train_tumor,'r')
  pos_wsi_img = f_tumor.readlines()
  neg_wsi_img = f_normal.readlines()
  f_normal.close()
  f_tumor.close()
  label_normal = [0]*len(neg_wsi_img)
  label_tumor = [1]*len(pos_wsi_img)
  
  # strip \n from the line scan
  all_wsi_img = neg_wsi_img
  all_wsi_img.extend(pos_wsi_img)
  all_label = label_normal
  all_label.extend(label_tumor)
  
  temp_list = []
  for one_wsi_img in all_wsi_img:
    temp_list.append(one_wsi_img.strip())
  all_wsi_img = temp_list
  
  #prob_map,tissue_area=generate_prob_map(all_wsi_img[0])#,net,transformer,batch_size)
  #calculate_feature(prob_map,tissue_area)
  #prob_map,tissue_area=generate_prob_map(all_wsi_img[-1])
  #calculate_feature(prob_map,tissue_area)
  
  start = time.clock()
  all_features = np.zeros((len(all_wsi_img),35),np.float32)
  for idx, one_wsi_img in enumerate(all_wsi_img):
    print "Processing the %dth/%d Whole Slide Image" %(idx+1,len(all_wsi_img))
    prob_map,tissue_area = generate_prob_map(one_wsi_img)
    all_features[idx,:] = calculate_feature(prob_map,tissue_area)
    mat_file_name = os.path.basename(one_wsi_img[:-4])+'.mat'
    mat_file_name = os.path.join('prob_maps_train',mat_file_name)
    savemat(mat_file_name,{'probability':prob_map})
    
  end = time.clock()
  runtime = end - start
  print "Running time %fs" % runtime
    
  np.save('all_features_train.npy',all_features)
  np.save('label_train.npy',np.array(all_label))
