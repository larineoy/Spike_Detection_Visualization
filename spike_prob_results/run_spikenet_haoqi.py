import os, glob
import numpy as np
#import pandas as pd
from tqdm import tqdm
import mne
from keras.models import model_from_json


def read_edf(path):
    edf = mne.io.read_raw_edf(path, verbose=False, preload=False)

    mono_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
    mono_channels2 = ['EEG 1', 'EEG 3', 'EEG 5', 'EEG 7', 'EEG 11', 'EEG 13', 'EEG 15', 'EEG 9', 'EEG 17', 'EEG 28', 'EEG 19', 'EEG 2', 'EEG 4', 'EEG 6', 'EEG 8', 'EEG 12', 'EEG 14', 'EEG 16', 'EEG 10']
    bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    
    eeg = edf.get_data(picks=mono_channels2)
    eeg *= 1e6
    eeg_mean = np.nanmean(eeg, axis=0)
    eeg[np.isnan(eeg)] = 0
    eeg[np.isinf(eeg)] = 0
    
    # montages
    bipolar_ids=np.array([[mono_channels.index(bc.split('-')[0]),mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_eeg = eeg[bipolar_ids[:,0]] - eeg[bipolar_ids[:,1]]
    average_eeg = eeg - eeg_mean
    eeg = np.concatenate([average_eeg, bipolar_eeg], axis=0)
    
    return eeg, edf.info['sfreq']
    
    
def main():
    data_dir = '/home/haoqisun/Downloads/asdtdegidata-edf/TD'
    data_paths = glob.glob(os.path.join(data_dir, '*.edf'))

    # load model 
    with open("model/spikenet1.o_structure.txt","r") as ff:
        json_string=ff.read()
    model = model_from_json(json_string)
    model.load_weights("model/spikenet1.o_weights.h5")
    newFs = 128    
    L = int(round(1*newFs))
    step = 4
    batch_size = 10000
    
    output_dir = '/home/haoqisun/Downloads/asdtdegidata-edf/spike_prob_results-TD'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, data_path in enumerate(tqdm(data_paths)):
        sid = os.path.basename(data_path).replace('.edf', '')
        eeg, Fs = read_edf(data_path)
        
        # preprocess
        eeg = mne.filter.notch_filter(eeg, Fs, 50, verbose=False)
        eeg = mne.filter.filter_data(eeg, Fs, 0.5, 45, verbose=False)
        eeg = mne.filter.resample(eeg, up=1.0, down=Fs/newFs, verbose=False)

        # run model
        start_ids = np.arange(0,eeg.shape[1]-L+1,step)
        start_ids = np.array_split(start_ids,int(np.ceil(len(start_ids)*1./batch_size)))
        yp = []
        for startid in tqdm(start_ids,leave=False):
            X = eeg[:,list(map(lambda x:np.arange(x,x+L),startid))].transpose(1,2,0)
            X = np.expand_dims(X,axis=2)
            yp.extend(model.predict(X, verbose=False).flatten())
        yp = np.array(yp).astype(float)
        padleft = (eeg.shape[1]-len(yp)*step)//2
        padright = eeg.shape[1]-len(yp)*step-padleft
        yp = np.r_[np.zeros(padleft)+yp[0], np.repeat(yp,step,axis=0), np.zeros(padright)+yp[-1]]
        
        #start_seconds = np.arange(len(yp))/newFs
        np.savez_compressed(os.path.join(output_dir, f'spike_prob_{sid}.npz'), p_spike=yp.astype('float32'), Fs=newFs)#, start_seconds=start_seconds)

if __name__=='__main__':
    main()
    
