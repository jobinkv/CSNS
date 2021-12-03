import os
import time
import numpy as np
import torch
from PIL import Image
from ipdb import set_trace as st
np.seterr(divide='ignore', invalid='ignore')

synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#default dataset wise

class Eval():
    def __init__(self,name_classes):
        self.num_class = len(name_classes)
        self.name_classes = name_classes
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.ignore_index = None
        self.pixelAcc = 0
        self.pIoU = 0
        self.cMIoU =0
        self.mbAcc =0
        self.labelDistib = 0
        self.synthia = True if self.num_class == 16 else False


    def Pixel_Accuracy(self):
        return self.pixelAcc/self.items
    def Pixel_Intersection_over_Union(self):
        return self.cMIoU/self.items
    def Classwise_Intersection_over_Union(self):
        out_miou = {}
        classwise_miou = np.nanmean(self.classMIoU,axis=1)
        i=0
        for keys in self.name_classes:
            out_miou[keys]=round(classwise_miou[i]*100,2)    
            i+=1
        out_miou['MIoU']=round(self.cMIoU*100/self.items,2)
        return out_miou 
    def Mean_Balanced_Pixel_Accuracy(self,alpha):
        #ipdb.set_trace()
        temp11 = self.mbAcc*(np.ones(self.num_class,)-alpha)
        temp22 = temp11.mean()
        temp33 = temp22/self.items
        out = self.mbAcc*alpha
        print (self.labelDistib)
    def Mean_Pixel_Accuracy(self, out_16_13=False):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.synthia:
            MPA_16 = np.nanmean(MPA[:self.ignore_index])
            MPA_13 = np.nanmean(MPA[synthia_set_16_to_13])
            return MPA_16, MPA_13
        if out_16_13:
            MPA_16 = np.nanmean(MPA[synthia_set_16])
            MPA_13 = np.nanmean(MPA[synthia_set_13])
            return MPA_16, MPA_13
        MPA = np.nanmean(MPA[:self.ignore_index])

        return MPA

    def Mean_Intersection_over_Union(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if self.synthia:
            MIoU_16 = np.nanmean(MIoU[:self.ignore_index])
            MIoU_13 = np.nanmean(MIoU[synthia_set_16_to_13])
            return MIoU_16, MIoU_13
        if out_16_13:
            MIoU_16 = np.nanmean(MIoU[synthia_set_16])
            MIoU_13 = np.nanmean(MIoU[synthia_set_13])
            return MIoU_16, MIoU_13
        MIoU = np.nanmean(MIoU[:self.ignore_index])

        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self, out_16_13=False):
        FWIoU = np.multiply(np.sum(self.confusion_matrix, axis=1), np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                         np.diag(self.confusion_matrix))
        if self.synthia:
            FWIoU_16 = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_16_to_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        if out_16_13:
            FWIoU_16 = np.sum(i for i in FWIoU[synthia_set_16] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_13] if not np.isnan(i)) / np.sum(self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)

        return FWIoU

    def Mean_Precision(self, out_16_13=False):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        if self.synthia:
            Precision_16 = np.nanmean(Precision[:self.ignore_index])
            Precision_13 = np.nanmean(Precision[synthia_set_16_to_13])
            return Precision_16, Precision_13
        if out_16_13:
            Precision_16 = np.nanmean(Precision[synthia_set_16])
            Precision_13 = np.nanmean(Precision[synthia_set_13])
            return Precision_16, Precision_13
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision
    
    def Print_Every_class_Eval(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Class_ratio = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        print('===>Everyclass:\t' + 'MPA\t' + 'MIoU\t' + 'PC\t' + 'Ratio\t' + 'Pred_Retio')
        if out_16_13: MIoU = MIoU[synthia_set_16]
        classWiseMiOu={}
        for ind_class in range(len(MIoU)):
            pa = str(round(MPA[ind_class] * 100, 2)) if not np.isnan(MPA[ind_class]) else 'nan'
            iou = str(round(MIoU[ind_class] * 100, 2)) if not np.isnan(MIoU[ind_class]) else 'nan'
            iou1 = round(MIoU[ind_class] * 100, 2) if not np.isnan(MIoU[ind_class]) else 0
            pc = str(round(Precision[ind_class] * 100, 2)) if not np.isnan(Precision[ind_class]) else 'nan'
            cr = str(round(Class_ratio[ind_class] * 100, 2)) if not np.isnan(Class_ratio[ind_class]) else 'nan'
            pr = str(round(Pred_retio[ind_class] * 100, 2)) if not np.isnan(Pred_retio[ind_class]) else 'nan'
            print('===>' + self.name_classes[ind_class] + ':\t' + pa + '\t' + iou + '\t' + pc + '\t' + cr + '\t' + pr)
            classWiseMiOu[self.name_classes[ind_class]]=iou1
        return classWiseMiOu
    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    def __generate_pixel_Accuracy(self,gt_image,pre_image):
        b,c,h,w = gt_image.shape
        #ipdb.set_trace()
        # make a binary array where true only if all the labels are correctly detect
        #currectMetrics = (gt_image==1)== (pre_image==1)
        #currectMetrics = (gt_image==1)== (gt_image==1)
        currectMetrics = gt_image==pre_image
        # get the pixel have all true 
        curImg = np.all(currectMetrics,axis=1)
        # return the ration of true in the total pixels
        return np.sum(curImg)/(b*h*w)
    def __generate_Pixel_Intersection_over_Union_old(self,gt_image,pre_image):
        # total true pixels
        sij = np.sum(gt_image)+np.sum(pre_image) # = \sum_c(L_{i,j}^{k,c}+P_{i,j}^{k,c})
        # calculating nij, i.e, intersection of labe; and predicted
        # find out the intersection ()
        # in the addition of label and pred 
        # intersection give the value of 2
        sum_of_pNl = gt_image+pre_image
        nij = np.count_nonzero(sum_of_pNl == 2)
        return nij/(sij-nij)
    def __generate_Pixel_Intersection_over_Union(self,gt_image,pre_image):
        # total true pixels
        currect_pixels = gt_image+pre_image # = \sum_c(L_{i,j}^{k,c}+P_{i,j}^{k,c})
        currect_pixels[currect_pixels<2]=0 #BCWH
        currect_pixels[currect_pixels>1]=1
        currect_pixels = currect_pixels.sum(axis=2)
        currect_pixels = currect_pixels.sum(axis=2)
        gt_image = gt_image.sum(axis=2)
        gt_image = gt_image.sum(axis=2)
        pre_image = pre_image.sum(axis=2)
        pre_image = pre_image.sum(axis=2)
        mean_iou = currect_pixels/(gt_image + pre_image - currect_pixels)
        temp =  np.nanmean(mean_iou, axis=1) #across labels
        return  np.nanmean(temp),np.nanmean(mean_iou, axis=0) #across images

    def __generate_Mean_Balanced_Pixel_Accuracy(self,gt_image,pre_image):
        temp = gt_image+pre_image
        temp[temp<2]=0
        temp[temp>1]=1
        temp1 = temp.sum(axis=2)
        temp2 = temp1.sum(axis=2)
        return temp2.sum(axis=0)
    def __generate_Label_Distribution(self,gt_image):
        temp1 = gt_image.sum(axis=2)
        temp2 = temp1.sum(axis=2)
        return temp2.sum(axis=0)
    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape
        self.items = self.items+1
        self.pixelAcc += self.__generate_pixel_Accuracy(gt_image,pre_image) 
        cMIoU, classMIoU = self.__generate_Pixel_Intersection_over_Union(gt_image,pre_image)
        self.cMIoU += cMIoU
        self.classMIoU = np.concatenate((self.classMIoU,classMIoU.reshape((self.num_class, 1))),axis=1)
        #if self.items==1: 
        #    self.cMIoU = self.__generate_Pixel_Intersection_over_Union(gt_image,pre_image)
        #else:
        #    self.cMIoU = np.column_stack((self.cMIoU,self.__generate_Pixel_Intersection_over_Union(gt_image,pre_image))) 
        #self.mbAcc += self.__generate_Mean_Balanced_Pixel_Accuracy(gt_image,pre_image) 
        #self.labelDistib += self.__generate_Label_Distribution(gt_image) 
        #self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.classMIoU = np.empty((self.num_class,1))
        self.classMIoU [:] = np.NaN
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.pixelAcc = 0
        self.items=0
        self.pIoU=0
        self.cMIoU = 0
        self.mbAcc = np.zeros(self.num_class,)
        self.labelDistib = np.zeros(self.num_class,)

def softmax(k, axis=None):
    exp_k = np.exp(k)
    return exp_k / np.sum(exp_k, axis=axis, keepdims=True)

