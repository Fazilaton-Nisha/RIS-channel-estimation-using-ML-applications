import numpy as np
import math
import time
import os.path
from torch.utils.data import Dataset
import h5py

pi = np.pi

class DatasetFolder(Dataset):
    def __init__(self, td):
        self.Yp=td[0]
        self.Hlabel=td[1]
        self.Hperf=td[2]
        self.Indicator=td[3]

    def __len__(self):
        return self.Yp.shape[0]

    def __getitem__(self, index):
        return [self.Yp[index], self.Hlabel[index], self.Hperf[index], self.Indicator[index]]

class DatasetFolder_DML(Dataset):
    def __init__(self, td00,td01,td02,td10,td11,td12,td20,td21,td22):
        self.td00=td00
        self.td01=td01
        self.td02=td02
        self.td10=td10
        self.td11=td11
        self.td12=td12
        self.td20=td20
        self.td21=td21
        self.td22=td22

    def __len__(self):
        return self.td00[0].shape[0]

    def __getitem__(self, index):
        return [self.td00[0][index],self.td00[1][index],self.td00[2][index],self.td00[3][index]],[self.td01[0][index],self.td01[1][index],self.td01[2][index],self.td01[3][index]],\
               [self.td02[0][index],self.td02[1][index],self.td02[2][index],self.td02[3][index]],[self.td10[0][index],self.td10[1][index],self.td10[2][index],self.td10[3][index]], \
               [self.td11[0][index], self.td11[1][index], self.td11[2][index], self.td11[3][index]], [self.td12[0][index],self.td12[1][index], self.td12[2][index], self.td12[3][index]], \
               [self.td20[0][index], self.td20[1][index], self.td20[2][index], self.td20[3][index]], [self.td21[0][index], \
               self.td21[1][index], self.td21[2][index], self.td21[3][index]], [self.td22[0][index], self.td22[1][index], self.td22[2][index], self.td22[3][index]]

def generate_data(hh,Pilot_num,SS):
    G = np.load('available_data/G_64_16.npy')
    CascadedH = np.matmul(np.diag(hh),G)
    Cascadedh = np.reshape(CascadedH,[-1],order='F')
    sigma2 =10 ** (-SS / 10)
    seed = math.floor(math.modf(time.time())[0] * 500 * 320000) ** 2 % (2 ** 32 - 1)
    np.random.seed(seed)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*Cascadedh.shape) + 1j * np.random.randn(*Cascadedh.shape))
    hh = Cascadedh+noise
    Psi = np.load('available_data/Psi_1024_' + str(Pilot_num)+'.npy')
    yy = np.matmul(Psi,hh)
    return yy,hh,Cascadedh

# generate the training data
def generate_datapair(Ns,Pilot_num,index,SNRdb,start,training_data_len):
    Yp = []
    Hlabel = []
    Hperfect = []
    Indicator = []
    if index==-1:
        data_hr=[]
        data_index=[]
        for sid in range(3):
            mat = h5py.File('available_data/DeepMIMO_channels_scenario' + str(sid + 1) +'_'+str(training_data_len)+ '.mat', 'r')
            data = np.transpose(mat['channels'])
            data = data['real'] + 1j * data['imag']
            if sid==2:
                data_hr.extend(data[start:start+Ns-2*int(Ns/3)])
                data_index.extend([sid] *(Ns-2*int(Ns/3)))
            else:
                data_hr.extend(data[start:start+int(Ns/3)])
                data_index.extend([sid] * int(Ns/3))

        start=0
        # data_hr=np.stack(data_hr,axis=0)
    else:
        mat = h5py.File('available_data/DeepMIMO_channels_scenario'+str(index+1)+'_'+str(training_data_len)+ '.mat', 'r')
        data = np.transpose(mat['channels'])
        data_hr = data['real'] + 1j * data['imag']


    for i in range(Ns):
        hr = data_hr[start+i]
        # print(np.sum(np.abs(hr)**2))
        SS = SNRdb
        if SNRdb == -2:
            SS = np.random.uniform(10, 20)
        if SNRdb == -3:
            SS = np.random.uniform(0, 10)
        if SNRdb == -1:
            SS = np.random.uniform(0, 20)
        yy, hh, hperf = generate_data(hr, Pilot_num, SS)
        Yp.append(yy)
        Hlabel.append(hh)
        Hperfect.append(hperf)
        if index==-1:
            Indicator.append(data_index[i])
        else:
            Indicator.append(index)

    Yp = np.stack(Yp, axis=0)
    Hlabel = np.stack(Hlabel, axis=0)
    Hperfect = np.stack(Hperfect, axis=0)
    Indicator = np.stack(Indicator, axis=0)

    return Yp, Hlabel, Hperfect, Indicator

def generate_MMSE_estimate(Hhat_LS,sigma2):
    # Hhat_MMSE=np.zeros(shape=Hhat_LS.shape,dtype=np.complex64)
    # for i in range(len(Hhat_LS)):
    #     temph=np.matmul(np.matmul(Rh,np.linalg.inv(Rh+(sigma2*np.eye(len(Rh))))),Hhat_LS[i])
    #     Hhat_MMSE[i]=temph
    Sample_num=len(Hhat_LS)
    Rh = np.zeros([1024, 1024], dtype=np.complex64)
    for s in range(Sample_num):
        temph=Hhat_LS[s]
        temph=temph.reshape([-1,1])
        Rh = Rh + np.matmul(temph, temph.transpose(1,0).conj())
    Rh = Rh / Sample_num


    return np.matmul(np.matmul(Rh,np.linalg.inv(Rh+(sigma2*np.eye(len(Rh))))),Hhat_LS.transpose(1,0)).transpose(1,0)

if __name__ == '__main__':

    M1 = 4
    M2 = 4
    N1 = 8
    N2 = 8

    data_len = 20000
    SNRdb = 10

    # generate G
    # if not os.path.isfile('available_data/G_' + str(N1*N2)+'_'+str(M1*M2)+ '.npy'):
    #     G = generate_G(M1=M1, M2=M2, N1=N1, N2=N2, num_paths=3)
    #     np.save('available_data/G_' + str(N1*N2)+'_'+str(M1*M2)+ '.npy',G)
    #     print('available_data/G_' + str(N1*N2)+'_'+str(M1*M2)+ '.npy has been saved!')

    for i in range(1):
        if i ==0:
            Pilot_num = 128
        else:
            if i ==1:
                Pilot_num = 64
        # generate Psi!
        # if not os.path.isfile('available_data/Psi_'+str(M1*M2*N1*N2)+'_'+str(Pilot_num)+'.npy'):
        #     Psi = np.sqrt(1 / Pilot_num) * (2 * (np.random.rand(Pilot_num, M1*M2*N1*N2) > 0.5) - 1)
        #     np.save('available_data/Psi_'+str(M1*M2*N1*N2)+'_'+str(Pilot_num)+'.npy',Psi)
        #     print('available_data/Psi_'+str(M1*M2*N1*N2)+'_'+str(Pilot_num)+'.npy has been saved!')

        # generate datapair for training for each scenario and each user
        print('start to generate data pair for Pilot_num=' + str(Pilot_num) + '!')
        for sid in range(3):
            for uid in range(3):
                # generate datapair for scenario 0
                print('generate data for scenario ' + str(sid) +' when Pilot_num=' + str(Pilot_num) + ' and User_id=' + str(uid) + '!')
                Yp, Hlabel, Hperf, Indicator = generate_datapair(Ns=data_len, Pilot_num=Pilot_num, index=sid,SNRdb=SNRdb,start=uid*data_len,training_data_len=data_len)

                np.save('available_data/Yp'+str(sid)+'_'+str(Pilot_num)+'_1024_'+str(SNRdb)+'dB_'+str(uid)+'_datalen_'+str(data_len)+'.npy',Yp)
                np.save('available_data/Hlabel'+str(sid)+'_'+str(Pilot_num)+'_1024_'+str(SNRdb)+'dB_'+str(uid)+'_datalen_'+str(data_len)+'.npy', Hlabel)
                np.save('available_data/Hperf'+str(sid)+'_'+str(Pilot_num)+'_1024_'+str(SNRdb)+'dB_'+str(uid)+'_datalen_'+str(data_len)+'.npy', Hperf)
                print('save data for scenario '+str(sid)+' when Pilot_num=' + str(Pilot_num)+ ' and User_id='+str(uid) + '!')


