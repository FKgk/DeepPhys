import os
import sys
import math
import numpy as np
import pandas as pd
import multiprocessing as mp

def lgth_transform(ecg, ws):
    lgth=ecg.shape[0]
    sqr_diff=np.zeros(lgth)
    diff=np.zeros(lgth)
    ecg=np.pad(ecg, ws, 'edge')
    for i in range(lgth):
        temp=ecg[i:i+ws+ws+1]
        left=temp[ws]-temp[0]
        right=temp[ws]-temp[-1]
        diff[i]=min(left, right)
        diff[diff<0]=0

    return np.multiply(diff, diff)

def integrate(ecg, ws):
    lgth=ecg.shape[0]
    integrate_ecg=np.zeros(lgth)
    ecg=np.pad(ecg, math.ceil(ws/2), mode='symmetric')
    for i in range(lgth):
        integrate_ecg[i]=np.sum(ecg[i:i+ws])/ws
    return integrate_ecg

def find_peak(data, ws):
    lgth=data.shape[0]
    true_peaks=list()
    for i in range(lgth-ws+1):
        temp=data[i:i+ws]
        if np.var(temp)<5:
            continue
        index=int((ws-1)/2)
        peak=True
        for j in range(index):
            if temp[index-j]<=temp[index-j-1] or temp[index+j]<=temp[index+j+1]:
                peak=False
                break

        if peak is True:
            true_peaks.append(int(i+(ws-1)/2))
    return np.asarray(true_peaks)

def find_R_peaks(ecg, peaks, ws):
    num_peak=peaks.shape[0]
    R_peaks=list()
    for index in range(num_peak):
        i=peaks[index]
        if i-2*ws>0 and i<ecg.shape[0]:
            temp_ecg=ecg[i-2*ws:i]
            R_peaks.append(int(np.argmax(temp_ecg)+i-2*ws))
    return np.asarray(R_peaks)

def EKG_QRS_detect(ecg, fs):
    sig_lgth=ecg.shape[0]
    ecg=ecg-np.mean(ecg)
    ecg_lgth_transform=lgth_transform(ecg, int(fs/20))

    ws=int(fs/8)
    ecg_integrate=integrate(ecg_lgth_transform, ws)/ws
    ws=int(fs/6)
    ecg_integrate=integrate(ecg_integrate, ws)
    ws=int(fs/36)
    ecg_integrate=integrate(ecg_integrate, ws)
    ws=int(fs/72)
    ecg_integrate=integrate(ecg_integrate, ws)

    peaks=find_peak(ecg_integrate, int(fs/10))
    R_peaks=find_R_peaks(ecg, peaks, int(fs/40))
    
    return R_peaks

    
def get_HR(ecg, fs=256, sec=15):
    R_peaks = EKG_QRS_detect(ecg, fs)
    
    return len(R_peaks) * (60 / sec)

def get_data(index, sec=1):
    video_length = video_info.loc[index, 'length']
    video_fps = video_info.loc[index, 'fps']
    
    signal_length = signal_info.loc[index, 'length']
    signal_fps = signal_info.loc[index, 'sample_rate']
    data = np.load(signal_info.loc[index, 'signal_path'])
    hr = get_HR(data, signal_fps, 60)
    
    length = int(min(video_length // video_fps, signal_length // signal_fps))
    
    sec_length = sec * signal_fps
    start_length = 59 * signal_fps
    iter_length = length - 59
    
    for i in range(iter_length):
        X = data[start_length + sec_length * i: start_length + sec_length * (i + 1)]
        Y = data[sec_length * i: start_length + sec_length * (i + 1)]
        
        
        yield (X, Y, hr)

def make_dataset(que, video_info, signal_info, start, end):
    print(f"{os.getpid()} : start make dataset")
    dataset = pd.DataFrame(columns=['X', 'Y', 'HR'])
    
    for i in range(start, end):
        for j, data in enumerate(get_data(i)):
            dataset.loc[i * 100 + j] =  data
        
        print(f"{i}/{end-start+1}")
    
    que.put(dataset)


base = os.path.join(os.getcwd(), 'data')
video_info = pd.read_csv(os.path.join(base, 'video_info.csv'))
signal_info = pd.read_csv(os.path.join(base, 'signal_info.csv'))
video_size = len(video_info.index)

iter_range = list(range(0, video_size, 50)) + [563]
process = list()
que = [mp.Queue() for i in range(len(iter_range) - 1)]

if __name__ == '__main__':
    for i, q in enumerate(que):
        p = mp.Process(target=make_dataset, args=(q, video_info.copy(), signal_info.copy(), \
                                                  iter_range[i], iter_range[i + 1]))
        process.append(p)

    for p in process:
        p.start()

    data = list()
    for p, q in zip(process, que):
        p.join()
        data.append(q.get())

    dataset = pd.concat(data)
    dataset.to_csv(os.path.join(base, 'signal_data_mp.csv'))

