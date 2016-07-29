import xmltodict
import numpy as np
import skimage.transform
import numpy_boost_camelyon
import openslide
import numpy.random

def is_overlap(bdbox1,bdbox2):
  [x1_min,y1_min,x1_max,y1_max] = bdbox1
  [x2_min,y2_min,x2_max,y2_max] = bdbox2
  
  x2_min -= 256
  y2_min -= 256
  x2_max += 256
  y2_max += 256
  
  if (x1_min < x2_max and x1_min > x2_min and y1_min < y2_max and y1_min > y2_min) or \
     (x1_max < x2_max and x1_max > x2_min and y1_max < y2_max and y1_max > y2_min) or \
     (x1_max < x2_max and x1_max > x2_min and y1_min < y2_max and y1_min > y2_min) or \
     (x1_min < x2_max and x1_min > x2_min and y1_max < y2_max and y1_max > y2_min):
    return True
  else:
    return False

# if the bounding box don't overlap with other bounding box return -1
# else return the last overlapping bounding box index
def get_overlap_index(bounding_box,bounding_boxes):
  index = -1
  for idx,bdbox in enumerate(bounding_boxes):
    if is_overlap(bdbox,bounding_box) or is_overlap(bounding_box,bdbox) and bdbox != bounding_box:
      index = idx
  return index

def get_aggregated_boxes(bounding_boxes):
  previous_size = len(bounding_boxes)
  current_size = 0
  while previous_size != current_size:
    aggregated_boxes = []
    for bounding_box in bounding_boxes:
      if len(aggregated_boxes) == 0:
        aggregated_boxes.append(bounding_box)
      else:
        index = get_overlap_index(bounding_box,aggregated_boxes)
        if index == -1:
          aggregated_boxes.append(bounding_box)
        else:
          [x1_min,y1_min,x1_max,y1_max] = aggregated_boxes[index]
          [x2_min,y2_min,x2_max,y2_max] = bounding_box
          aggregated_boxes[index] = [min(x1_min,x2_min),min(y1_min,y2_min),\
                                  max(x1_max,x2_max),max(y1_max,y2_max)]  
    previous_size = current_size
    current_size = len(aggregated_boxes)
    bounding_boxes = aggregated_boxes
  return aggregated_boxes
 
def generate_possible_patches(xml_file,mask_file,wsi_img,wsi_pre_pos,choose_num):
  # read whole slide image for mask
  f = openslide.OpenSlide(mask_file)
  wsi = openslide.OpenSlide(wsi_img)
  with open(xml_file) as fd:
    doc = xmltodict.parse(fd.read())
    
  # read annotations from xml file and calculate the bounding_boxes  
  annotations = doc['ASAP_Annotations']['Annotations']['Annotation']
  bounding_boxes = []
  if not isinstance(annotations,list):
    annotations = [annotations]
  for annotation in annotations:
    coordinates = annotation['Coordinates']['Coordinate']
    x_coords = [float(coordinates[i]['@X']) for i in range(len(coordinates))]
    y_coords = [float(coordinates[i]['@Y']) for i in range(len(coordinates))]
    bounding_boxes.append([np.min(x_coords),np.min(y_coords),np.max(x_coords),np.max(y_coords)])    
  
  # some of the bounding_boxes overlap with each other, aggregate them together to form a larger one
  aggregated_boxes = get_aggregated_boxes(bounding_boxes)
  possible_pixels = (np.zeros((0,),np.int64),np.zeros((0,),np.int64))
  offsets = (np.zeros((0,),np.int64),np.zeros((0,),np.int64))
  for bounding_box in aggregated_boxes:
    [x_min,y_min,x_max,y_max] = bounding_box
    #print np.floor(x_min),np.floor(y_min),np.ceil(x_max),np.ceil(y_max)
    #mask = np.zeros((int(y_max-y_min),int(x_max-x_min)),np.uint8)
    #print (np.floor(x_min)-256,np.floor(y_min)-256)
    mask = f.read_region((int(np.floor(x_min))-256,int(np.floor(y_min))-256),
        0,(int(np.ceil(x_max)-np.floor(x_min))+256,int(np.ceil(y_max)-np.floor(y_min))+256))
    #mask.save('mask.tif')
    mask = mask.convert('RGB')
    mask = np.array(mask)[:,:,0]/255
    img_integral = skimage.transform.integral_image(mask)
    #possible_mask = np.zeros((int(y_max-y_min),int(x_max-x_min)),np.uint8)
    
    #print img_integral.dtype
    result = numpy_boost_camelyon.compute(img_integral)
    pixels = np.nonzero(result>256*256/2.0)
    #print "Number of possible pixels: %d" % pixels[0].shape[0]
    offset_x = np.int64(np.floor(x_min))
    offset_y = np.int64(np.floor(y_min))
    possible_pixels = (np.concatenate((possible_pixels[0],pixels[0])), np.concatenate((possible_pixels[1],pixels[1])))
    offsets = (np.concatenate((offsets[0],np.ones((pixels[0].shape[0],),np.int64)*offset_y)), np.concatenate((offsets[1],np.ones((pixels[0].shape[0],),np.int64)*offset_x)))
    
  # generate the positive examples  
  index = numpy.random.permutation(possible_pixels[0].shape[0])
  #print "Number of possible pixels: %d" %possible_pixels[0].shape[0]
  for ind in range(choose_num):
    x = offsets[1][index[ind]]+possible_pixels[1][index[ind]]-256
    y = offsets[0][index[ind]]+possible_pixels[0][index[ind]]-256
    tile = wsi.read_region((x,y),0,(256,256))
    tile_name = wsi_pre_pos + '_%d.tif' %ind
    tile.save(tile_name)
    #tile.save('test/tile_%d.tif' % ind)
  
if __name__=='__main__':
  test_file = '/data/pathology/CAMELYON16/TrainingData/Ground_Truth/XML/Tumor_002.xml'
  mask_file = '/data/pathology/CAMELYON16/TrainingData/Ground_Truth/Mask/Tumor_002_Mask.tif'
  
  generate_possible_patches(test_file,mask_file)
  #f = openslide.OpenSlide(mask_file)
  #with open(test_file) as fd:
  #  doc = xmltodict.parse(fd.read())
  #  
  #annotations = doc['ASAP_Annotations']['Annotations']['Annotation']
  #bounding_boxes = []
  #if not isinstance(annotations,list):
  #  annotations = [annotations]
  #for annotation in annotations:
  #  coordinates = annotation['Coordinates']['Coordinate']
  #  x_coords = [float(coordinates[i]['@X']) for i in range(len(coordinates))]
  #  y_coords = [float(coordinates[i]['@Y']) for i in range(len(coordinates))]
  #  bounding_boxes.append([np.min(x_coords),np.min(y_coords),np.max(x_coords),np.max(y_coords)])
  #
  #aggregated_boxes = get_aggregated_boxes(bounding_boxes)
  #possible_pixels = (np.zeros((0,),np.int64),np.zeros((0,),np.int64))
  #offsets = (np.zeros((0,),np.int64),np.zeros((0,),np.int64))
  #for bounding_box in aggregated_boxes:
  #  [x_min,y_min,x_max,y_max] = bounding_box
  #  print np.floor(x_min),np.floor(y_min),np.ceil(x_max),np.ceil(y_max)
  #  #mask = np.zeros((int(y_max-y_min),int(x_max-x_min)),np.uint8)
  #  #print (np.floor(x_min)-256,np.floor(y_min)-256)
  #  mask = f.read_region((int(np.floor(x_min))-256,int(np.floor(y_min))-256),0,(int(np.ceil(x_max)-np.floor(x_min)),int(np.ceil(y_max)-np.floor(y_min))))
  #  #mask.save('mask.tif')
  #  mask = mask.convert('RGB')
  #  mask = np.array(mask)[:,:,0]/255
  #  img_integral = skimage.transform.integral_image(mask)
  #  #possible_mask = np.zeros((int(y_max-y_min),int(x_max-x_min)),np.uint8)
  #  
  #  #print img_integral.dtype
  #  result = numpy_boost_camelyon.compute(img_integral)
  #  pixels = np.nonzero(result>256*256/2.0)
  #  #print "Number of possible pixels: %d" % pixels[0].shape[0]
  #  offset_x = np.int64(np.floor(x_min))
  #  offset_y = np.int64(np.floor(y_min))
  #  possible_pixels = (np.concatenate((possible_pixels[0],pixels[0])), np.concatenate((possible_pixels[1],pixels[1])))
  #  offsets = (np.concatenate((offsets[0],np.ones((pixels[0].shape[0],),np.int64)*offset_y)), np.concatenate((offsets[1],np.ones((pixels[0].shape[0],),np.int64)*offset_x)))
  #    
  #index = numpy.random.permutation(possible_pixels[0].shape[0])
  #print "Number of possible pixels: %d" %possible_pixels[0].shape[0]
  #for ind in range(1000):
  #  x = offsets[1][index[ind]]+possible_pixels[1][index[ind]]-256
  #  y = offsets[0][index[ind]]+possible_pixels[0][index[ind]]-256
  #  tile = f.read_region((x,y),0,(256,256))
  #  tile.save('test/tile_%d.tif' % ind)
  #  
  ##for bounding_box in bounding_boxes:
  ##  for x in range(int(bounding_box[0]),int(bounding_box[2])):
  ##    for y in range(int(bounding_box[1]),int(bounding_box[3])):
  ##      possible_pixels.append((x,y))
