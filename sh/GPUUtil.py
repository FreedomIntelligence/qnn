# coding=utf-8
'''
Created on Aug 22, 2017

@author: colinliang
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from subprocess import check_output

def __execute_process(command_shell):
    stdout = check_output(command_shell, shell=True).strip()
    if not isinstance(stdout, (str)):
        stdout = stdout.decode()
    return stdout

def listGPUsWithScore(min_free_gpu_usage=0.4,min_free_gpu_memory=0.1,gpu_weight=1.0,memory_weight=5.0,max_rand_score=0.0, verbose=False):
    ''' 列出可用的GPU编号（绝对的，参见https://github.com/wookayin/gpustat）
    Inputs:  TODO
        min_free_gpu_usage: 取值区间[0.0, 1.0)， 该GPU的空闲率在该值之上时才可能被选中
        min_free_memory:    取值在[0.0, 1.0) 表示空闲的内存在该比例之上时才可能被选中； 如果取值在[1, 正无穷)， 则为 空闲缓存在  该 MB以上的才会被选中
        gpu_weight: 计算得分时gpu空闲率的权重
        memory_weight ： 计算得分时显存空闲率的权重
        max_rand_score： 为了GPU使用均衡，最终得分会加上一个(-max_rand_score,max_rand_score)间的随机数
    Return:
        [(gpu_idx, score),(gpu_idx, score), ...] 其中的score为降序排列
    '''
    import numpy as np
    # 可以通过 man nvidia-smi查看更多用法！！ 注意 带空格的选项要用下划线代替
    #cmd=nvidia-smi --query-gpu="driver_version,index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total,pci.bus_id" --format=csv,noheader,nounits
    #cmd='nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
    cmd='nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total  --format=csv,noheader,nounits'
    gpu_status=__execute_process(cmd)
    if(verbose):
        print('----GPU status:  ')
        print(cmd)
        print(gpu_status)
        print('---\n')
    
    idx2gpu={}
    idx2mem={}
    idx2score={}
    for l in gpu_status.split('\n'):
        idx,gpu_percent,mem_used,mem_total=l.split(',')
        g=1.0-float(gpu_percent.strip())/100.0
        
        idx2gpu[idx.strip()]=g
        m=1.0-float(mem_used)/float(mem_total);
        min_free_gpu_memory_this_gpu = min_free_gpu_memory if min_free_gpu_memory<1.0 else float(min_free_gpu_memory)/float(mem_total)
        idx2mem[idx.strip()]=m
        idx2score[idx.strip()]=0 if(g<min_free_gpu_usage or m <min_free_gpu_memory_this_gpu) else g*gpu_weight+memory_weight*m
        idx2score[idx.strip()]+=max_rand_score* np.random.uniform(-max_rand_score,max_rand_score)
    s=sorted(idx2score.items(),key=lambda x:x[1],reverse=True)
#    print(s)
    return s
    
def setCUDA_VISIBLE_DEVICES(num_GPUs=1, min_free_gpu_memory=1024, min_free_gpu_usage=0.1,verbose=False):
    ''' 自动设置tensoflow可见的GPU； 注意，该代码必须在import任何CUDA代码（比如tensorflow）之前运行
	num_GPUs: 需要多少个GPU
	min_free_gpu_memory: 该显卡至少要有min_free_gpu_usage MB的空闲显存
	min_free_gpu_usage: 该GPU的占用率应当在该值以下才符合要求
	
    '''
    GPUs=listGPUsWithScore(min_free_gpu_usage=min_free_gpu_usage,min_free_gpu_memory=min_free_gpu_memory,gpu_weight=1.0,memory_weight=2.0,max_rand_score=0.0001,verbose=verbose)
    GPUs=[g for g,s in GPUs if g>0.0]
    if(len(GPUs)<num_GPUs): 
        return -1
    GPUs=GPUs[:num_GPUs]
    gpu_str=','.join(GPUs)
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_str
    if(verbose):
        print('INFO: using GPUs with  PCI index: %s'%gpu_str)
    return 0

if __name__ == '__main__':
    setCUDA_VISIBLE_DEVICES(num_GPUs=1, verbose=True)
    import tensorflow as tf
    sess=tf.InteractiveSession()
    a=tf.placeholder(tf.float32)
    b=tf.placeholder(tf.float32)