"""

This file pickles the output of object detection into a file structure mirroring the input structure
(Specific to the structures we currently use)

Provide the path to the top level directory of the source image 
(For our case it will be train/ dev/ ...)

Toggle the "NESTED = True" if there are subdirectories that contain the images

Output file names are set under "NEST_FOLDER" for subdirectory based structure
Output file names are set under "FLAT_FOLDER" for subdirectory based structure

DONE represents a list of subdirectories to ignore when working with nested file structure

FAIL represents name of the file to log failures 

E.g:

Training Data: arg=train/, NESTED = True, NEST_FOLDER=train
Development Data: arg=dev/, NESTED = False, NEST_FOLDER=dev
Test Data: arg=test1/, NESTED = False, NEST_FOLDER=test

"""

import sys
import os
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from yolo import Yolo

# Directory to save output
NEST_FOLDER = "train"
FLAT_FOLDER = "test"

# Adhoc flag for whether what structure files are stored
NESTED = True 

# File for name of failed images
FAIL = "failed"

# Global Yolo Instance
yolo = Yolo()

# List of subrepos to ignore
DONE = []
        

def main():
  if NESTED:
    nest_process()
  else:
    flat_process()


def nest_process():
  os.mkdir(NEST_FOLDER)
  directory = sys.argv[1]
  
  fail = open(FAIL + ".txt", "a+")

  for root, dirs, _ in os.walk(directory):
    for subdir in dirs:
      if str(subdir) in DONE:
        continue;

      subdir_path = os.path.join(root, subdir)
      output_path = NEST_FOLDER + "/"+ str(subdir)

      os.mkdir(output_path)
      for _, _, files in os.walk(subdir_path):
        for image in files:
          output_file = output_path + "/" + str(image) + ".txt"
          link = os.path.join(subdir_path, image)
          img_data = object_detect(link)
          
          try:
            with open(output_file, 'wb') as f:
              pickle.dump(img_data, f)
          except:
            fail.write(str(output_file) + "\n")
            print("failure on " + str(output_file))

          """
          with open(output_file, 'rb') as f:
            a = pickle.load(f)
            if a != img_data:
              print(img_data)
              print(a)
          """


def flat_process():
  os.mkdir(FLAT_FOLDER)
  directory = sys.argv[1]
  for root, _, files in os.walk(directory):
    for image in files:
      output_file = FLAT_FOLDER + "/" + str(image) + ".txt"
      link = os.path.join(root, image)
      img_data = object_detect(link)

      with open(output_file, 'wb') as f:
        pickle.dump(img_data, f)

      """
      with open(output_file, 'rb') as f:
        a = pickle.load(f)
        if a != img_data:
          print(img_data)
          print(a)
      """


def object_detect(link: str):
  img_data = yolo.detect(link.encode())
  return img_data

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage: python3 preyolo.py <image directory>")
    sys.exit(1)

  main()
