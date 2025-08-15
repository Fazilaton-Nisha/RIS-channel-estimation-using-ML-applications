r'''
    This is training runner for channel estimation. 
'''

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os

from Estimators import  NMSELoss,DCE_P128,SC_P128,Conv_P128,FC_P128
from generate_data import DatasetFolder,DatasetFolder_DML


class Y2HRunner():
    def __init__(self):
        self.Pilot_num = 128
        self.data_len = 20000
        self.SNRdb = 10
        self.num_workers = 8
        self.batch_size = 256
        self.batch_size_DML = 256
        self.lr = 1e-3
        self.lr_decay = 30
        self.lr_threshold = 1e-6
        self.n_epochs = 100
        self.print_freq = 50
        self.optimizer = 'adam'
        self.train_test_ratio = 0.9

    def get_optimizer(self, parameters, lr):
        if self.optimizer == 'adam':
            return optim.Adam(parameters, lr=lr)
        elif self.optimizer == 'sgd':
            return optim.SGD(parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.optimizer))

    def get_data(self,data_len,indicator,uid):

        Yp = np.load('available_data/Yp'+str(indicator)+'_' + str(self.Pilot_num) + '_1024_' + str(self.SNRdb) + 'dB_' + str(uid) +'_datalen_'+str(data_len)+'.npy')
        Hlabel = np.load('available_data/Hlabel'+str(indicator)+'_' + str(self.Pilot_num) + '_1024_' + str(self.SNRdb) + 'dB_' + str(uid) + '_datalen_'+str(data_len)+'.npy')
        Hperf = np.load('available_data/Hperf'+str(indicator)+'_' + str(self.Pilot_num) + '_1024_' + str(self.SNRdb) + 'dB_' + str(uid) + '_datalen_'+str(data_len)+'.npy')

        print('data loaded for scenario'+str(indicator)+' user'+str(uid)+'!')
        Indicator = []
        for i in range(data_len):
            Indicator.append(indicator)
        Indicator = np.stack(Indicator, axis=0)

        Yp = Yp[:data_len]
        Hlabel = Hlabel[:data_len]
        Hperf = Hperf[:data_len]

        start = int(Yp.shape[0] * self.train_test_ratio)
        Yp_train, Yp_val = Yp[:start], Yp[start:]
        Hlabel_train, Hlabel_val = Hlabel[:start], Hlabel[start:]
        Hperf_train, Hperf_val = Hperf[:start],Hperf[start:]
        Indicator_train, Indicator_val = Indicator[:start], Indicator[start:]

        return [Yp_train, Hlabel_train, Hperf_train, Indicator_train],[Yp_val,Hlabel_val,Hperf_val,Indicator_val]

    def get_dataloader_DML(self, data_len):

        td00, vd00 = self.get_data(data_len, 0, 0)
        td01, vd01 = self.get_data(data_len, 0, 1)
        td02, vd02 = self.get_data(data_len, 0, 2)
        td10, vd10 = self.get_data(data_len, 1, 0)
        td11, vd11 = self.get_data(data_len, 1, 1)
        td12, vd12 = self.get_data(data_len, 1, 2)
        td20, vd20 = self.get_data(data_len, 2, 0)
        td21, vd21 = self.get_data(data_len, 2, 1)
        td22, vd22 = self.get_data(data_len, 2, 2)

        # dataLoader for training or val
        train_dataset = DatasetFolder_DML(td00,td01,td02,td10,td11,td12,td20,td21,td22)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size_DML, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        val_dataset = DatasetFolder_DML(vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size_DML, shuffle=True, num_workers=self.num_workers, pin_memory=True)

        # print(len(val_loader))
        return train_loader,val_loader

    def get_loss(self, td, CNN, criterion, device):

        Yp = td[0]
        Hlabel = td[1]
        Hperfect = td[2]
        bs = len(Yp)

        # complex--->real
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)

        # the input and output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        Hhat = CNN(Yp_input.reshape(bs, 2, 16, 8))

        loss = criterion(Hhat, label_out)
        loss_perf = criterion(Hhat, perfect_out)

        return loss, loss_perf

    def get_estimate(self, vd, CNN, device):

        Yp = vd[0]
        Hlabel = vd[1]
        Hperfect = vd[2]

        bs = Yp.shape[0]
        # complex--->real
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float()
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float()
        # the input and the output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).to(device)
        Hhat = CNN(Yp_input.reshape(bs, 2, 16, 8).float()).detach().cpu()

        return Hhat,label_out, perfect_out

    def train_DCE_for_scenario0(self):
        gpu_list = '0,1'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        CNN = DCE_P128()

        if len(gpu_list.split(',')) > 1:
            CNN = torch.nn.DataParallel(CNN).to(device)
        else:
            CNN.to(device)

        data_len = self.data_len
        td,vd=self.get_data(self.data_len,0,0)
        # dataLoader for training
        train_dataset = DatasetFolder(td)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
            drop_last=True)

        # dataLoader for validation
        val_dataset = DatasetFolder(vd)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        print('Data Loaded!')

        criterion = NMSELoss()
        optimizer = self.get_optimizer(CNN.parameters(), self.lr)
        best_nmse = 1000.

        print('Everything prepared well, start to train...')

        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print('Scenario 0: ' f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CNN.train()
            for it, td in enumerate(train_loader):
                optimizer.zero_grad()
                loss, loss_perf = self.get_loss(td,CNN,criterion,device)
                loss.backward()
                optimizer.step()
                if it % self.print_freq == 0:
                    print(
                        f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}\t Loss_perf {loss_perf.item():.5f}')
            CNN.eval()
            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []
                for vd in val_loader:

                    Hhat, label_out, perfect_out=self.get_estimate(vd,CNN,device)
                    Hhat_list.append(Hhat)
                    Hlabel_list.append(label_out)
                    Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlabel = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)

                nmse = criterion(Hhat, Hlabel)
                nmse_perf = criterion(Hhat, Hperfect)

                if nmse < best_nmse:
                    torch.save({'cnn': CNN.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/DCE',f'{data_len}_{self.SNRdb}dB_best_scenario0.pth'))
                    best_nmse = nmse.item()
                    print('CNN saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer.param_groups[0]['lr'] = self.lr_threshold

    def train_DCE_for_scenario1(self):
        gpu_list = '2,3'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        CNN = DCE_P128()

        if len(gpu_list.split(',')) > 1:
            CNN = torch.nn.DataParallel(CNN).to(device)
        else:
            CNN.to(device)

        data_len = self.data_len
        td, vd = self.get_data(self.data_len, 1, 0)
        # dataLoader for training
        train_dataset = DatasetFolder(td)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
            drop_last=True)

        # dataLoader for validation
        val_dataset = DatasetFolder(vd)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        print('Data Loaded!')

        criterion = NMSELoss()
        optimizer = self.get_optimizer(CNN.parameters(), self.lr)
        best_nmse = 1000.

        print('Everything prepared well, start to train...')

        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print('Scenario 1: ' f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CNN.train()
            for it, td in enumerate(train_loader):
                optimizer.zero_grad()
                loss, loss_perf = self.get_loss(td,CNN,criterion,device)
                loss.backward()
                optimizer.step()
                if it % self.print_freq == 0:
                    print(
                        f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}\t Loss_perf {loss_perf.item():.5f}')
            CNN.eval()
            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []
                for vd in val_loader:

                    Hhat, label_out, perfect_out=self.get_estimate(vd,CNN,device)

                    Hhat_list.append(Hhat)
                    Hlabel_list.append(label_out)
                    Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlabel = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)

                nmse = criterion(Hhat, Hlabel)
                nmse_perf = criterion(Hhat, Hperfect)

                if nmse < best_nmse:
                    torch.save({'cnn': CNN.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/DCE',
                                            f'{data_len}_{self.SNRdb}dB_best_scenario1.pth'))
                    best_nmse = nmse.item()
                    print('CNN saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer.param_groups[0]['lr'] = self.lr_threshold

    def train_DCE_for_scenario2(self):
        gpu_list = '0,1,2,3'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        CNN = DCE_P128()

        if len(gpu_list.split(',')) > 1:
            CNN = torch.nn.DataParallel(CNN).to(device)
        else:
            CNN.to(device)

        data_len = self.data_len
        td, vd = self.get_data(self.data_len, 2, 0)
        # dataLoader for training
        train_dataset = DatasetFolder(td)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
            drop_last=True)

        # dataLoader for validation
        val_dataset = DatasetFolder(vd)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        print('Data Loaded!')

        criterion = NMSELoss()
        optimizer = self.get_optimizer(CNN.parameters(), self.lr)
        best_nmse = 1000.

        print('Everything prepared well, start to train...')

        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print('Scenario 2: ' f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',
                  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CNN.train()
            for it, td in enumerate(train_loader):
                optimizer.zero_grad()
                loss, loss_perf = self.get_loss(td,CNN,criterion,device)
                loss.backward()
                optimizer.step()
                if it % self.print_freq == 0:
                    print(
                        f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {loss.item():.5f}\t Loss_perf {loss_perf.item():.5f}')
            CNN.eval()
            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []
                for vd in val_loader:

                    Hhat, label_out, perfect_out=self.get_estimate(vd,CNN,device)
                    Hhat_list.append(Hhat)
                    Hlabel_list.append(label_out)
                    Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlabel = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)

                nmse = criterion(Hhat, Hlabel)
                nmse_perf = criterion(Hhat, Hperfect)

                if nmse < best_nmse:
                    torch.save({'cnn': CNN.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/DCE',
                                            f'{data_len}_{self.SNRdb}dB_best_scenario2.pth'))
                    best_nmse = nmse.item()
                    print('CNN saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer.param_groups[0]['lr'] = self.lr_threshold

    def train_DCE_for_DML(self):
        gpu_list = '4,5,6,7'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        CNN = DCE_P128()

        if len(gpu_list.split(',')) > 1:
            CNN = torch.nn.DataParallel(CNN).to(device)
        else:
            CNN.to(device)

        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')

        criterion = NMSELoss()
        optimizer = self.get_optimizer(CNN.parameters(), self.lr)
        best_nmse = 1000.

        print('Everything prepared well, start to train...')


        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print('DML:'f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CNN.train()
            for it, (td00,td01,td02,td10,td11,td12,td20,td21,td22) in enumerate(train_loader):

                sutd=[[td00,td01,td02],[td10,td11,td12],[td20,td21,td22]]
                total_loss=0
                total_loss_perf=0

                optimizer.zero_grad()

                for sid in range(3):
                    for uid in range(3):
                        loss,loss_perf=self.get_loss(sutd[sid][uid],CNN,criterion,device)
                        loss = loss/9
                        loss_perf = loss_perf/9
                        total_loss+=loss
                        total_loss_perf+=loss_perf

                        # calculate gradient
                        loss.backward()

                # update weights
                optimizer.step()
                if it % self.print_freq == 0:
                    print(f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}\t Loss_perf {total_loss_perf.item():.5f}')

            CNN.eval()

            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []

                for vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22 in val_loader:

                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            Hhat, label_out, perfect_out = self.get_estimate(suvd[sid][uid], CNN, device)
                            Hhat_list.append(Hhat)
                            Hlabel_list.append(label_out)
                            Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlabel = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)

                nmse = criterion(Hhat, Hlabel)
                nmse_perf = criterion(Hhat,Hperfect)
                if epoch==self.n_epochs-1:
                    fp = os.path.join(f'./workspace/Pn_{self.Pilot_num}/DCE',
                                      f'{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth')
                    torch.save({'cnn': CNN.state_dict()}, fp)
                    print('CNN finally saved!')

                if nmse < best_nmse:
                    torch.save({'cnn': CNN.state_dict()}, os.path.join(f'./workspace/Pn_{self.Pilot_num}/DCE', f'{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    best_nmse = nmse.item()
                    print('CNN saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f},NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')

            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer.param_groups[0]['lr'] = self.lr_threshold

    def get_HDCE_loss(self, td, Conv, CE, criterion, device):

        Yp = td[0]
        Hlabel = td[1]
        Hperfect = td[2]
        bs = len(Yp)

        # complex--->real
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)

        # the input and output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)
        h_out = Conv(Yp_input)
        Hhat = CE(h_out)

        loss = criterion(Hhat, label_out)
        loss_perf = criterion(Hhat, perfect_out)

        return loss, loss_perf

    def get_HDCE_estimate(self, vd, Conv, CE, device):

        Yp = vd[0]
        Hlabel = vd[1]
        Hperfect = vd[2]
        bs = len(Yp)

        # complex--->real
        label_out = torch.cat([Hlabel.real, Hlabel.imag], dim=1).float().to(device)
        perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float().to(device)

        # the input and output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)
        h_out = Conv(Yp_input)
        Hhat = CE(h_out)

        return Hhat,label_out,perfect_out

    def train_Conv_Linear_of_HDCE(self):
        gpu_list = '0,1,2,3'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        Conv0 = Conv_P128()
        Conv1 = Conv_P128()
        Conv2 = Conv_P128()
        CE = FC_P128()

        if len(gpu_list.split(',')) > 1:
            Conv0 = torch.nn.DataParallel(Conv0).to(device)
            Conv1 = torch.nn.DataParallel(Conv1).to(device)
            Conv2 = torch.nn.DataParallel(Conv2).to(device)
            CE = torch.nn.DataParallel(CE).to(device)
        else:
            Conv0.to(device)
            Conv1.to(device)
            Conv2.to(device)
            CE.to(device)


        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')

        criterion = NMSELoss()
        optimizer_Conv0 = self.get_optimizer(Conv0.parameters(), self.lr)
        optimizer_Conv1 = self.get_optimizer(Conv1.parameters(), self.lr)
        optimizer_Conv2 = self.get_optimizer(Conv2.parameters(), self.lr)
        optimizer_CE = self.get_optimizer(CE.parameters(), self.lr)

        best_nmse = 1000.

        print('Everything prepared well, start to train...')


        for epoch in range(self.n_epochs):
            current_lr = optimizer_Conv0.param_groups[0]['lr']
            print(
                'Conv+Linear of HDCE:' f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            Conv0.train()
            Conv1.train()
            Conv2.train()
            CE.train()
            Conv=[Conv0,Conv1,Conv2]
            for it, (td00,td01,td02,td10,td11,td12,td20,td21,td22) in enumerate(train_loader):

                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]

                optimizer_Conv0.zero_grad()
                optimizer_Conv1.zero_grad()
                optimizer_Conv2.zero_grad()
                optimizer_CE.zero_grad()

                total_loss=0
                total_loss_perf=0


                for sid in range(3):
                    for uid in range(3):
                        loss,loss_perf = self.get_HDCE_loss(sutd[sid][uid],Conv[sid],CE,criterion,device)
                        loss = loss/9
                        loss_perf=loss_perf/9
                        total_loss+=loss
                        total_loss_perf+=loss_perf
                        loss.backward()

                optimizer_Conv0.step()
                optimizer_Conv1.step()
                optimizer_Conv2.step()
                optimizer_CE.step()

                if it % self.print_freq == 0:
                    print(
                        f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}\t Loss_perf {total_loss_perf.item():.5f}')

            Conv0.eval()
            Conv1.eval()
            Conv2.eval()
            CE.eval()
            Conv=[Conv0,Conv1,Conv2]
            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []

                for vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            Hhat,label_out,perfect_out=self.get_HDCE_estimate(suvd[sid][uid],Conv[sid],CE,device)

                    Hhat_list.append(Hhat)
                    Hlabel_list.append(label_out)
                    Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlable = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)
                nmse = criterion(Hhat, Hlable)
                nmse_perf = criterion(Hhat, Hperfect)


                if epoch==self.n_epochs-1:
                    torch.save({'conv': Conv0.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv0_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'conv': Conv1.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv1_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'conv': Conv2.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv2_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'linear': CE.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Linear_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    print('HDCE finally saved!')
                if nmse < best_nmse:
                    torch.save({'conv': Conv0.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv0_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'conv': Conv1.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv1_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'conv': Conv2.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Conv2_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'linear': CE.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'Linear_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))

                    best_nmse = nmse.item()
                    print('HDCE saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer_Conv0.param_groups[0]['lr'] = optimizer_Conv0.param_groups[0]['lr'] * 0.5
                    optimizer_Conv1.param_groups[0]['lr'] = optimizer_Conv1.param_groups[0]['lr'] * 0.5
                    optimizer_Conv2.param_groups[0]['lr'] = optimizer_Conv2.param_groups[0]['lr'] * 0.5
                    optimizer_CE.param_groups[0]['lr'] = optimizer_CE.param_groups[0]['lr'] * 0.5
                if optimizer_Conv0.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer_Conv0.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_Conv1.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_Conv2.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_CE.param_groups[0]['lr'] = self.lr_threshold

    def get_SE_loss(self, td, CNN, device):

        Yp = td[0]
        indicator=td[3]

        bs = len(Yp)

        label_out = indicator.long().to(device)

        # the input and output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        pred_indicator = CNN(Yp_input.reshape(bs, 2, 16, 8))

        # loss
        loss = F.nll_loss(pred_indicator, label_out)

        return loss

    def get_SE_estimate(self, td, CNN, device):

        Yp = td[0]
        indicator = td[3]

        bs = len(Yp)

        label_out = indicator.long().to(device)

        # the input and output
        Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
        pred_indicator = CNN(Yp_input.reshape(bs, 2, 16, 8))

        return pred_indicator,label_out

    def train_SC_of_HDCE(self):
        gpu_list = '0,1,2,3'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        CNN = SC_P128()

        if len(gpu_list.split(',')) > 1:
            CNN = torch.nn.DataParallel(CNN).to(device)
        else:
            CNN.to(device)


        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')

        optimizer = self.get_optimizer(CNN.parameters(), self.lr)
        best_acc = 0

        print('Everything prepared well, start to train...')

        for epoch in range(self.n_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print('SC of HDCE: 'f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CNN.train()
            for it, (td00,td01,td02,td10,td11,td12,td20,td21,td22) in enumerate(train_loader):

                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]
                optimizer.zero_grad()
                total_loss=0

                for sid in range(3):
                    for uid in range(3):
                        loss=self.get_SE_loss(sutd[sid][uid],CNN,device)
                        loss=loss/9
                        total_loss+=loss
                        loss.backward()

                optimizer.step()
                if it % self.print_freq == 0:

                    print(f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}')


            CNN.eval()
            with torch.no_grad():
                pred_list = []
                label_list = []

                for vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            pred_indicator,label_out = self.get_SE_estimate(suvd[sid][uid],CNN,device)
                            pred = pred_indicator.argmax(dim=1)
                            pred_list.append(pred)
                            label_list.append(label_out)

                pred = torch.cat(pred_list, dim=0)
                label = torch.cat(label_list, dim=0)
                acc = pred.eq(label.view_as(pred)).sum().item()/(len(label))

                if epoch==self.n_epochs-1:
                    fp = os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE', f'{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML_SC.pth')
                    torch.save({'cnn': CNN.state_dict()}, fp)
                    print('SC finally saved!')

                if acc > best_acc:
                    torch.save({'cnn': CNN.state_dict()}, os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',f'{self.batch_size_DML}_{self.SNRdb}dB_best_DML_SC.pth'))
                    best_acc = acc
                    print('SC saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || acc {acc:.2%}, best acc: {best_acc:.2%}')
                print('==============================================================')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
                if optimizer.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer.param_groups[0]['lr'] = self.lr_threshold

    def train_SDCE(self):
        gpu_list = '4,5,6,7'
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        device = 'cuda'

        CE0=DCE_P128()
        CE1=DCE_P128()
        CE2=DCE_P128()

        if len(gpu_list.split(',')) > 1:
            CE0 = torch.nn.DataParallel(CE0).to(device)
            CE1 = torch.nn.DataParallel(CE1).to(device)
            CE2 = torch.nn.DataParallel(CE2).to(device)
        else:
            CE0.to(device)
            CE1.to(device)
            CE2.to(device)

        data_len = self.data_len
        train_loader, val_loader = self.get_dataloader_DML(data_len=data_len)
        print('Data Loaded!')

        criterion = NMSELoss()

        optimizer_CE0 = self.get_optimizer(CE0.parameters(), self.lr)
        optimizer_CE1 = self.get_optimizer(CE1.parameters(), self.lr)
        optimizer_CE2 = self.get_optimizer(CE2.parameters(), self.lr)


        best_nmse = 1000.

        print('Everything prepared well, start to train...')


        for epoch in range(self.n_epochs):
            current_lr = optimizer_CE1.param_groups[0]['lr']
            print(
                'SDCE:' f'SNR: {self.SNRdb} ' f'Epoch [{epoch}]/[{self.n_epochs}] learning rate: {current_lr:.4e}',
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            CE0.train()
            CE1.train()
            CE2.train()

            CE=[CE0,CE1,CE2]

            for it, (td00,td01,td02,td10,td11,td12,td20,td21,td22) in enumerate(train_loader):

                sutd = [[td00, td01, td02], [td10, td11, td12], [td20, td21, td22]]

                optimizer_CE0.zero_grad()
                optimizer_CE1.zero_grad()
                optimizer_CE2.zero_grad()

                total_loss=0
                total_loss_perf=0

                for sid in range(3):
                    for uid in range(3):
                        loss,loss_perf = self.get_loss(sutd[sid][uid],CE[sid],criterion,device)
                        loss = loss/9
                        loss_perf=loss_perf/9
                        total_loss+=loss
                        total_loss_perf+=loss_perf
                        loss.backward()

                optimizer_CE0.step()
                optimizer_CE1.step()
                optimizer_CE2.step()


                if it % self.print_freq == 0:
                    print(
                        f'Epoch: [{epoch}/{self.n_epochs}][{it}/{len(train_loader)}]\t Loss {total_loss.item():.5f}\t Loss_perf {total_loss_perf.item():.5f}')

            CE0.eval()
            CE1.eval()
            CE2.eval()

            CE=[CE0,CE1,CE2]
            with torch.no_grad():
                Hhat_list = []
                Hlabel_list = []
                Hperfect_list = []

                for vd00,vd01,vd02,vd10,vd11,vd12,vd20,vd21,vd22 in val_loader:
                    suvd = [[vd00, vd01, vd02], [vd10, vd11, vd12], [vd20, vd21, vd22]]

                    for sid in range(3):
                        for uid in range(3):
                            Hhat,label_out,perfect_out=self.get_estimate(suvd[sid][uid],CE[sid],device)

                    Hhat_list.append(Hhat)
                    Hlabel_list.append(label_out)
                    Hperfect_list.append(perfect_out)

                Hhat = torch.cat(Hhat_list, dim=0)
                Hlable = torch.cat(Hlabel_list, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)
                nmse = criterion(Hhat, Hlable)
                nmse_perf = criterion(Hhat, Hperfect)


                if epoch==self.n_epochs-1:
                    torch.save({'ce': CE0.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'CE0_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'ce': CE1.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'CE1_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    torch.save({'ce': CE2.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'CE2_{self.batch_size_DML}_{self.SNRdb}dB_epoch{epoch}_DML.pth'))
                    print('SDCE finally saved!')
                if nmse < best_nmse:
                    torch.save({'ce': CE0.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'CE0_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'ce': CE1.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'CE1_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))
                    torch.save({'ce': CE2.state_dict()},
                               os.path.join(f'./workspace/Pn_{self.Pilot_num}/HDCE',
                                            f'CE2_{self.batch_size_DML}_{self.SNRdb}dB_best_DML.pth'))

                    best_nmse = nmse.item()
                    print('SDCE saved!')
                print(
                    f'Epoch [{epoch}]/[{self.n_epochs}] || NMSE {nmse.item():.5f}, NMSE_perf {nmse_perf.item():.5f}, best nmse: {best_nmse:.5f}')
                print('==============================================================')
            if epoch > 0:
                if epoch % self.lr_decay == 0:
                    optimizer_CE0.param_groups[0]['lr'] = optimizer_CE0.param_groups[0]['lr'] * 0.5
                    optimizer_CE1.param_groups[0]['lr'] = optimizer_CE1.param_groups[0]['lr'] * 0.
                    optimizer_CE2.param_groups[0]['lr'] = optimizer_CE2.param_groups[0]['lr'] * 0.
                if optimizer_CE0.param_groups[0]['lr'] < self.lr_threshold:
                    optimizer_CE0.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_CE1.param_groups[0]['lr'] = self.lr_threshold
                    optimizer_CE2.param_groups[0]['lr'] = self.lr_threshold


if __name__ == '__main__':

    order_number=5

    runner = Y2HRunner()

    if order_number==0:
        runner.train_DCE_for_scenario0()
    else:
        if order_number==1:
            runner.train_DCE_for_scenario1()
        else:
            if order_number==2:
                runner.train_DCE_for_scenario2()
            else:
                if order_number==3:
                    runner.train_DCE_for_DML()
                else:
                    if order_number==4:
                        runner.train_Conv_Linear_of_HDCE()
                    else:
                        if order_number==5:
                            runner.train_SC_of_HDCE()
                        else:
                            if order_number==6:
                                runner.train_SDCE()
