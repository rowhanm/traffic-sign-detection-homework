from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

from model import Net


from helpers import get_transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def tta(model, size, exp_name):
    test_dir = 'images/test_images'
    outfile = exp_name+'_out.csv'
    t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11 = get_transforms(size)
    output_file = open(outfile, "w")
    output_file.write("Filename,ClassId\n")
    for f in tqdm(os.listdir(test_dir)):
        if 'ppm' in f:
            data1 = t11(pil_loader(test_dir + '/' + f))
            data2 = t1((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data3 = t2((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data4 = t3((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data5 = t4((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data6 = t5((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data7 = t6((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data8 = t7((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data9 = t8((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data10 = t9((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data11 = t10((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data12 = t0((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data13 = t0((pil_loader(test_dir + '/' + f))).unsqueeze(0)

            final_input = torch.cat((data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13),0)

            ntrans, c, h, w = final_input.size()
            res = model(Variable(final_input.view(-1, c, h, w)).to(device))
            
            output = res.mean(0)
            pred = output.data.max(0, keepdim=True)[1]

            file_id = f[0:5]
            output_file.write("%s,%d\n" % (file_id, pred))

    output_file.close()

    print("Succesfully wrote " + outfile + ', you can upload this file to the kaggle '
          'competition at https://www.kaggle.com/c/nyu-cv-fall-2018/')
    
def tta_ensemble(model1, model2, size, exp_name):
    test_dir = 'images/test_images'
    outfile = exp_name+'_out.csv'
    t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11 = get_transforms(size)
    output_file = open(outfile, "w")
    output_file.write("Filename,ClassId\n")
    for f in tqdm(os.listdir(test_dir)):
        if 'ppm' in f:
            data1 = t11(pil_loader(test_dir + '/' + f))
            data2 = t1((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data3 = t2((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data4 = t3((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data5 = t4((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data6 = t5((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data7 = t6((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data8 = t7((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data9 = t8((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data10 = t9((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data11 = t10((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data12 = t0((pil_loader(test_dir + '/' + f))).unsqueeze(0)
            data13 = t0((pil_loader(test_dir + '/' + f))).unsqueeze(0)

            final_input = torch.cat((data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13),0)

            ntrans, c, h, w = final_input.size()
            temp_output_1 = model1(Variable(final_input.view(-1, c, h, w)).to(device))
            temp_output_2 = model2(Variable(final_input.view(-1, c, h, w)).to(device))

            ensemble_res = torch.cat((temp_output_1, temp_output_2),0)
            output = ensemble_res.mean(0)
            pred = output.data.max(0, keepdim=True)[1]

            file_id = f[0:5]
            output_file.write("%s,%d\n" % (file_id, pred))

    output_file.close()

    print("Succesfully wrote " + outfile + ', you can upload this file to the kaggle '
          'competition at https://www.kaggle.com/c/nyu-cv-fall-2018/')
        