import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from glob import glob
path = "D:/video_crowd_count/dataset/HT21/train/"
direct = os.listdir(path)

print(direct)
channels_sum, channels_squared_sum, num_batches = 0, 0, 0
for i in direct:
	fullpath = os.path.join(path,i,"img1/*.jpg")
	for file in tqdm(glob(fullpath)):
		img = cv2.imread(file)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		img = transforms.ToTensor()(img)
		channels_sum += torch.mean(img,dim=[1,2])
		# print(channels_sum)
		channels_squared_sum += torch.mean(img**2,dim=[1,2])
		# print(channels_squared_sum)
		num_batches += 1

# print(channels_sum)
# print(channels_squared_sum)
# mean = channels_sum / num_batches

# std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

# print(mean)
# print(std)
# tensor([0.3467, 0.5197, 0.4980])
# tensor([0.2125, 0.0232, 0.0410])


path = "D:/video_crowd_count/dataset/SENSE/video_ori"
direct = os.listdir(path)


channels_sum, channels_squared_sum, num_batches = 0, 0, 0
for i in tqdm(direct):
	fullpath = os.path.join(path,i,"*.jpg")
	for file in glob(fullpath):
		img = cv2.imread(file)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		img = transforms.ToTensor()(img)
		channels_sum += torch.mean(img,dim=[1,2])
		# print(channels_sum)
		channels_squared_sum += torch.mean(img**2,dim=[1,2])
		# print(channels_squared_sum)
		num_batches += 1

print(channels_sum)
print(channels_squared_sum)
mean = channels_sum / num_batches

std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

print(mean)
print(std)
# tensor([0.5037, 0.5132, 0.5140])
# tensor([0.2257, 0.0302, 0.0414])
