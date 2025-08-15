import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from Estimators import NMSELoss,DCE_P128,SC_P128,Conv_P128,FC_P128
from generate_data import generate_MMSE_estimate,generate_datapair,DatasetFolder

import matplotlib.pyplot as plt


class model_val():
    def __init__(self):
        super().__init__()
        self.training_SNRdb = 10
        self.num_workers = 8
        self.batch_size = 200
        self.batch_size_DML = 256
        self.training_data_len = 20000
        self.indicator = -1
        self.data_len_for_test = 10000
        self.Pilot_num=128
        

    def test_for_SC(self):
        Pilot_num = 128
        SNRdb = np.arange(5, 16, 2)
        SC = SC_P128()
        SC = torch.nn.DataParallel(SC).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE', f'{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML_SC.pth')
        try:
            SC.load_state_dict(torch.load(fp))
        except:
            SC.load_state_dict(torch.load(fp)['cnn'])

        acc = []
        SC.eval()

        with torch.no_grad():

            for snr in SNRdb:

                print(
                    'generate test data for scenario ' + str(self.indicator) + ' when Pilot_num=' + str(Pilot_num) + '!')
                td = generate_datapair(Ns=self.data_len_for_test, Pilot_num=128, index=self.indicator, SNRdb=snr,start=0,training_data_len=self.training_data_len)
                test_dataset = DatasetFolder(td)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                    drop_last=False)

                print('==============================================================')
                print(f'SNR: {snr} dB')

                pred_list = []
                label_list = []


                for Yp, H_label, H_perfect, indicator in test_loader:
                    bs = Yp.shape[0]
                    # 标签 complex--->real
                    label_out = indicator.long().to(device)
                    # 网络输入
                    Yp_input = torch.cat([Yp.real, Yp.imag], dim=1).float().to(device)
                    pred_indicator = SC(Yp_input.reshape(bs, 2, 16, 8))
                    pred = pred_indicator.argmax(dim=1)

                    pred_list.append(pred)
                    label_list.append(label_out)


                pred = torch.cat(pred_list, dim=0)
                label = torch.cat(label_list, dim=0)
                acc.append(pred.eq(label.view_as(pred)).sum().item() / (len(label)))
                print(acc)

        return acc

    def test_for_CE_P128_for_scenario0(self):
        Pilot_num = 128
        SNRdb = np.arange(5, 16, 2)

        # the DCE trained by the single user from scenario 0
        CNN_for_scenario0 = DCE_P128()
        CNN_for_scenario0 = torch.nn.DataParallel(CNN_for_scenario0).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.training_data_len}_{self.training_SNRdb}dB_best_scenario0.pth')
        try:
            CNN_for_scenario0.load_state_dict(torch.load(fp))
        except:
            CNN_for_scenario0.load_state_dict(torch.load(fp)['cnn'])

        # the DCE trained by the single user from scenario 1
        CNN_for_scenario1 = DCE_P128()
        CNN_for_scenario1 = torch.nn.DataParallel(CNN_for_scenario1).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.training_data_len}_{self.training_SNRdb}dB_best_scenario1.pth')
        try:
            CNN_for_scenario1.load_state_dict(torch.load(fp))
        except:
            CNN_for_scenario1.load_state_dict(torch.load(fp)['cnn'])

        # the DCE trained by the single user from scenario 2
        CNN_for_scenario2 = DCE_P128()
        CNN_for_scenario2 = torch.nn.DataParallel(CNN_for_scenario2).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.training_data_len}_{self.training_SNRdb}dB_best_scenario2.pth')
        try:
            CNN_for_scenario2.load_state_dict(torch.load(fp))
        except:
            CNN_for_scenario2.load_state_dict(torch.load(fp)['cnn'])


        # the DCE trained by DML
        CNN_for_DML = DCE_P128()
        CNN_for_DML = torch.nn.DataParallel(CNN_for_DML).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CNN_for_DML.load_state_dict(torch.load(fp))
        except:
            CNN_for_DML.load_state_dict(torch.load(fp)['cnn'])

        # the scenario classifier of HDCE trained by DML
        SC = SC_P128()
        SC = torch.nn.DataParallel(SC).to(device)
        fp =  os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE', f'{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML_SC.pth')
        try:
            SC.load_state_dict(torch.load(fp))
        except:
            SC.load_state_dict(torch.load(fp)['cnn'])

        # the feature exttractor of HDCE trained by DML for scenario 0
        Conv0 = Conv_P128()
        Conv0 = torch.nn.DataParallel(Conv0).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',f'Conv0_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            Conv0.load_state_dict(torch.load(fp))
        except:
            Conv0.load_state_dict(torch.load(fp)['conv'])

        # the feature exttractor of HDCE trained by DML for scenario 1
        Conv1 = Conv_P128()
        Conv1 = torch.nn.DataParallel(Conv1).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'Conv1_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            Conv1.load_state_dict(torch.load(fp))
        except:
            Conv1.load_state_dict(torch.load(fp)['conv'])

        # the feature exttractor of HDCE trained by DML for scenario 2
        Conv2 = Conv_P128()
        Conv2 = torch.nn.DataParallel(Conv2).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'Conv2_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            Conv2.load_state_dict(torch.load(fp))
        except:
            Conv2.load_state_dict(torch.load(fp)['conv'])

        # the feature mapper of HDCE trained by DML
        CE = FC_P128()
        CE = torch.nn.DataParallel(CE).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'Linear_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE.load_state_dict(torch.load(fp))
        except:
            CE.load_state_dict(torch.load(fp)['linear'])

        CE0 = DCE_P128()
        CE0 = torch.nn.DataParallel(CE0).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'CE0_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE0.load_state_dict(torch.load(fp))
        except:
            CE0.load_state_dict(torch.load(fp)['ce'])

        CE1 = DCE_P128()
        CE1 = torch.nn.DataParallel(CE1).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'CE1_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE1.load_state_dict(torch.load(fp))
        except:
            CE1.load_state_dict(torch.load(fp)['ce'])

        CE2 = DCE_P128()
        CE2 = torch.nn.DataParallel(CE2).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'CE2_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE2.load_state_dict(torch.load(fp))
        except:
            CE2.load_state_dict(torch.load(fp)['ce'])

        CNN_for_scenario0.eval()
        CNN_for_scenario1.eval()
        CNN_for_scenario2.eval()
        CNN_for_DML.eval()
        SC.eval()
        Conv0.eval()
        Conv1.eval()
        Conv2.eval()
        CE.eval()
        CE0.eval()
        CE1.eval()
        CE2.eval()

        criterion = NMSELoss()


        NMSE_for_LS = []
        NMSE_for_MMSE=[]
        NMSE_for_scenario0 = []
        NMSE_for_scenario1 = []
        NMSE_for_scenario2 = []
        NMSE_for_DCE = []
        NMSE_for_HDCE = []
        NMSE_for_SDCE=[]


        with torch.no_grad():

            for snr in SNRdb:
                print(
                    'generate test data for scenario ' + str(self.indicator) + ' when Pilot_num=' + str(Pilot_num) + ' and User_id=' + str(
                        0) + '!')
                td = generate_datapair(Ns=self.data_len_for_test, Pilot_num=128, index=self.indicator, SNRdb=snr,start=self.training_data_len*3,training_data_len=self.training_data_len)
                test_dataset = DatasetFolder(td)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                    drop_last=False)

                print('==============================================================')
                print(f'SNR: {snr} dB')

                Hhat_list0 = []
                Hhat_list1 = []
                Hhat_list2 = []
                Hhat_list_DCE = []

                Hperfect_list = []
                HLS_list = []
                HMMSE_list=[]

                Hhat_list_HDCE = []
                Hhat_list_SDCE=[]
                Hperfect_list_HDCE = []

                for Yp, HLS, Hperfect, indicator in test_loader:

                    HMMSE=generate_MMSE_estimate(HLS.numpy(),sigma2 =10 ** (-snr / 10))
                    HMMSE=torch.from_numpy(HMMSE)
                    HMMSE_list.append(torch.cat([HMMSE.real, HMMSE.imag], dim=1).float())

                    bs = Yp.shape[0]
                    #  complex--->real
                    label_out = torch.cat([HLS.real, HLS.imag], dim=1).float()
                    perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float()
                    # the input and output
                    Yp_input = torch.stack([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)

                    Hhat0 = CNN_for_scenario0(Yp_input).detach().cpu()
                    Hhat_list0.append(Hhat0)

                    Hhat1 = CNN_for_scenario1(Yp_input).detach().cpu()
                    Hhat_list1.append(Hhat1)

                    Hhat2 = CNN_for_scenario2(Yp_input).detach().cpu()
                    Hhat_list2.append(Hhat2)

                    Hhat_DCE = CNN_for_DML(Yp_input).detach().cpu()
                    Hhat_list_DCE.append(Hhat_DCE)

                    HLS_list.append(label_out)
                    Hperfect_list.append(perfect_out)


                    pred_indicator = SC(Yp_input)
                    pred = pred_indicator.argmax(dim=1)
                    # pred = indicator.long().to(device)

                    Yp_class = [[], [], []]
                    label_class = [[], [], []]
                    for i, m in enumerate(pred):
                        Yp_class[m].append(Yp_input[i])
                        label_class[m].append(perfect_out[i])
                    if len(Yp_class[0]):
                        hh = label_class[0]
                        hh = torch.stack(hh, dim=0)
                        Hperfect_list_HDCE.append(hh)
                        yy = Yp_class[0]
                        yy = torch.stack(yy, dim=0)
                        h_out1 = Conv0(yy)
                        h_out1 = CE(h_out1).cpu()
                        Hhat_list_HDCE.append(h_out1)

                        Hhat_list_SDCE.append(CE0(yy).cpu())

                    if len(Yp_class[1]):
                        hh = label_class[1]
                        hh = torch.stack(hh, dim=0)
                        Hperfect_list_HDCE.append(hh)
                        yy = Yp_class[1]
                        yy = torch.stack(yy, dim=0)
                        h_out1 = Conv1(yy)
                        h_out1 = CE(h_out1).cpu()
                        Hhat_list_HDCE.append(h_out1)
                        Hhat_list_SDCE.append(CE1(yy).cpu())

                    if len(Yp_class[2]):
                        hh = label_class[2]
                        hh = torch.stack(hh, dim=0)
                        Hperfect_list_HDCE.append(hh)
                        yy = Yp_class[2]
                        yy = torch.stack(yy, dim=0)
                        h_out1 = Conv2(yy)
                        h_out1 = CE(h_out1).cpu()
                        Hhat_list_HDCE.append(h_out1)
                        Hhat_list_SDCE.append(CE2(yy).cpu())


                Hhat0 = torch.cat(Hhat_list0, dim=0)
                Hhat1 = torch.cat(Hhat_list1, dim=0)
                Hhat2 = torch.cat(Hhat_list2, dim=0)
                Hhat_DCE = torch.cat(Hhat_list_DCE, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)
                HLS = torch.cat(HLS_list, dim=0)
                HMMSE=torch.cat(HMMSE_list,dim=0)

                Hhat_HDCE = torch.cat(Hhat_list_HDCE,dim=0)
                Hhat_FullHDCE = torch.cat(Hhat_list_SDCE,dim=0)
                Hperfect_HDCE = torch.cat(Hperfect_list_HDCE,dim=0)


                nmse0 = criterion(Hhat0, Hperfect)
                nmse1 = criterion(Hhat1, Hperfect)
                nmse2 = criterion(Hhat2, Hperfect)
                nmse_DCE = criterion(Hhat_DCE, Hperfect)
                nmse_HDCE = criterion(Hhat_HDCE,Hperfect_HDCE)
                nmse_SDCE = criterion(Hhat_FullHDCE,Hperfect_HDCE)

                nmse_LS = criterion(HLS, Hperfect)
                nmse_MMSE = criterion(HMMSE, Hperfect)

                NMSE_for_scenario0.append(nmse0.item())
                NMSE_for_scenario1.append(nmse1.item())
                NMSE_for_scenario2.append(nmse2.item())
                NMSE_for_DCE.append(nmse_DCE.item())
                NMSE_for_HDCE.append(nmse_HDCE.item())
                NMSE_for_SDCE.append(nmse_SDCE.item())
                NMSE_for_LS.append(nmse_LS.item())
                NMSE_for_MMSE.append(nmse_MMSE.item())

                #
                print(f'NMSE_for_scenario0: {NMSE_for_scenario0}')
                print(f'NMSE_for_scenario1: {NMSE_for_scenario1}')
                print(f'NMSE_for_scenario2: {NMSE_for_scenario2}')
                print(f'NMSE_for_DCE: {NMSE_for_DCE}')
                print(f'NMSE_for_HDCE: {NMSE_for_HDCE}')
                print(f'NMSE_for_SDCE: {NMSE_for_SDCE}')
                print(f'NMSE_for_LS: {NMSE_for_LS}')
                print(f'NMSE_for_MMSE: {NMSE_for_MMSE}')

        plt.figure()
        plt.plot(SNRdb, 10*np.log10(NMSE_for_scenario0), 'm>-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_scenario1), 'b^-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_scenario2), 'c<-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_DCE), 'rs-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_HDCE), 'rd-')
        plt.plot(SNRdb, 10 * np.log10(NMSE_for_SDCE), 'ro-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_LS), 'k--')
        plt.plot(SNRdb, 10 * np.log10(NMSE_for_MMSE), 'r--')
        plt.grid()


        plt.legend(['DCE network only trained on scenario 1', 'DCE network only trained on scenario 2', 'DCE network only trained on scenario 3','proposed DML based DCE network','proposed DML based HDCE network','DML based SDCE network', 'LS algorithm','MMSE algorithm'])


        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.xticks(np.arange(5, 16, 2))
        plt.yticks(np.arange(-20, 1, 5))
        plt.savefig('results/NMSEvsSNR_for_128_10dB_scenario'+str(self.indicator))

        plt.show()

        return 0

    def test_for_CE_P128_for_all_scenarios(self):
        Pilot_num = 128
        SNRdb = np.arange(5, 16, 2)

        # the DCE trained by the single user from scenario 0
        CNN_for_scenario0 = DCE_P128()
        CNN_for_scenario0 = torch.nn.DataParallel(CNN_for_scenario0).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.training_data_len}_{self.training_SNRdb}dB_best_scenario0.pth')
        try:
            CNN_for_scenario0.load_state_dict(torch.load(fp))
        except:
            CNN_for_scenario0.load_state_dict(torch.load(fp)['cnn'])

        # the DCE trained by the single user from scenario 1
        CNN_for_scenario1 = DCE_P128()
        CNN_for_scenario1 = torch.nn.DataParallel(CNN_for_scenario1).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.training_data_len}_{self.training_SNRdb}dB_best_scenario1.pth')
        try:
            CNN_for_scenario1.load_state_dict(torch.load(fp))
        except:
            CNN_for_scenario1.load_state_dict(torch.load(fp)['cnn'])

        # the DCE trained by the single user from scenario 2
        CNN_for_scenario2 = DCE_P128()
        CNN_for_scenario2 = torch.nn.DataParallel(CNN_for_scenario2).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.training_data_len}_{self.training_SNRdb}dB_best_scenario2.pth')
        try:
            CNN_for_scenario2.load_state_dict(torch.load(fp))
        except:
            CNN_for_scenario2.load_state_dict(torch.load(fp)['cnn'])


        # the DCE trained by DML
        CNN_for_DML = DCE_P128()
        CNN_for_DML = torch.nn.DataParallel(CNN_for_DML).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/DCE',f'{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CNN_for_DML.load_state_dict(torch.load(fp))
        except:
            CNN_for_DML.load_state_dict(torch.load(fp)['cnn'])

        # the scenario classifier of HDCE trained by DML
        SC = SC_P128()
        SC = torch.nn.DataParallel(SC).to(device)
        fp =  os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE', f'{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML_SC.pth')
        try:
            SC.load_state_dict(torch.load(fp))
        except:
            SC.load_state_dict(torch.load(fp)['cnn'])

        # the feature exttractor of HDCE trained by DML for scenario 0
        Conv0 = Conv_P128()
        Conv0 = torch.nn.DataParallel(Conv0).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',f'Conv0_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            Conv0.load_state_dict(torch.load(fp))
        except:
            Conv0.load_state_dict(torch.load(fp)['conv'])

        # the feature exttractor of HDCE trained by DML for scenario 1
        Conv1 = Conv_P128()
        Conv1 = torch.nn.DataParallel(Conv1).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'Conv1_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            Conv1.load_state_dict(torch.load(fp))
        except:
            Conv1.load_state_dict(torch.load(fp)['conv'])

        # the feature exttractor of HDCE trained by DML for scenario 2
        Conv2 = Conv_P128()
        Conv2 = torch.nn.DataParallel(Conv2).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'Conv2_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            Conv2.load_state_dict(torch.load(fp))
        except:
            Conv2.load_state_dict(torch.load(fp)['conv'])

        # the feature mapper of HDCE trained by DML
        CE = FC_P128()
        CE = torch.nn.DataParallel(CE).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'Linear_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE.load_state_dict(torch.load(fp))
        except:
            CE.load_state_dict(torch.load(fp)['linear'])

        CE0 = DCE_P128()
        CE0 = torch.nn.DataParallel(CE0).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'CE0_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE0.load_state_dict(torch.load(fp))
        except:
            CE0.load_state_dict(torch.load(fp)['ce'])

        CE1 = DCE_P128()
        CE1 = torch.nn.DataParallel(CE1).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'CE1_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE1.load_state_dict(torch.load(fp))
        except:
            CE1.load_state_dict(torch.load(fp)['ce'])

        CE2 = DCE_P128()
        CE2 = torch.nn.DataParallel(CE2).to(device)
        fp = os.path.join(f'./workspace/Pn_{Pilot_num}/HDCE',
                          f'CE2_{self.batch_size_DML}_{self.training_SNRdb}dB_epoch99_DML.pth')
        try:
            CE2.load_state_dict(torch.load(fp))
        except:
            CE2.load_state_dict(torch.load(fp)['ce'])

        CNN_for_scenario0.eval()
        CNN_for_scenario1.eval()
        CNN_for_scenario2.eval()
        CNN_for_DML.eval()
        SC.eval()
        Conv0.eval()
        Conv1.eval()
        Conv2.eval()
        CE.eval()
        CE0.eval()
        CE1.eval()
        CE2.eval()

        criterion = NMSELoss()


        NMSE_for_LS = []
        NMSE_for_MMSE=[]
        NMSE_for_scenario0 = []
        NMSE_for_scenario1 = []
        NMSE_for_scenario2 = []
        NMSE_for_DCE = []
        NMSE_for_HDCE = []
        NMSE_for_SDCE=[]


        with torch.no_grad():

            for snr in SNRdb:
                print(
                    'generate test data for scenario ' + str(self.indicator) + ' when Pilot_num=' + str(Pilot_num) + ' and User_id=' + str(
                        0) + '!')
                td = generate_datapair(Ns=self.data_len_for_test, Pilot_num=128, index=self.indicator, SNRdb=snr,start=self.training_data_len*3,training_data_len=self.training_data_len)
                test_dataset = DatasetFolder(td)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                    drop_last=False)

                print('==============================================================')
                print(f'SNR: {snr} dB')

                Hhat_list0 = []
                Hhat_list1 = []
                Hhat_list2 = []
                Hhat_list_DCE = []

                Hperfect_list = []
                HLS_list = []
                HMMSE_list=[]

                Hhat_list_HDCE = []
                Hhat_list_SDCE=[]
                Hperfect_list_HDCE = []

                for Yp, HLS, Hperfect, indicator in test_loader:

                    HMMSE=generate_MMSE_estimate(HLS.numpy(),sigma2 =10 ** (-snr / 10))
                    HMMSE=torch.from_numpy(HMMSE)
                    HMMSE_list.append(torch.cat([HMMSE.real, HMMSE.imag], dim=1).float())

                    bs = Yp.shape[0]
                    #  complex--->real
                    label_out = torch.cat([HLS.real, HLS.imag], dim=1).float()
                    perfect_out = torch.cat([Hperfect.real, Hperfect.imag], dim=1).float()
                    # the input and output
                    Yp_input = torch.stack([Yp.real, Yp.imag], dim=1).reshape(bs, 2, 16, 8).float().to(device)

                    Hhat0 = CNN_for_scenario0(Yp_input).detach().cpu()
                    Hhat_list0.append(Hhat0)

                    Hhat1 = CNN_for_scenario1(Yp_input).detach().cpu()
                    Hhat_list1.append(Hhat1)

                    Hhat2 = CNN_for_scenario2(Yp_input).detach().cpu()
                    Hhat_list2.append(Hhat2)

                    Hhat_DCE = CNN_for_DML(Yp_input).detach().cpu()
                    Hhat_list_DCE.append(Hhat_DCE)

                    HLS_list.append(label_out)
                    Hperfect_list.append(perfect_out)


                    pred_indicator = SC(Yp_input)
                    pred = pred_indicator.argmax(dim=1)

                    Yp_class = [[], [], []]
                    label_class = [[], [], []]
                    for i, m in enumerate(pred):
                        Yp_class[m].append(Yp_input[i])
                        label_class[m].append(perfect_out[i])
                    if len(Yp_class[0]):
                        hh = label_class[0]
                        hh = torch.stack(hh, dim=0)
                        Hperfect_list_HDCE.append(hh)
                        yy = Yp_class[0]
                        yy = torch.stack(yy, dim=0)
                        h_out1 = Conv0(yy)
                        h_out1 = CE(h_out1).cpu()
                        Hhat_list_HDCE.append(h_out1)

                        Hhat_list_SDCE.append(CE0(yy).cpu())

                    if len(Yp_class[1]):
                        hh = label_class[1]
                        hh = torch.stack(hh, dim=0)
                        Hperfect_list_HDCE.append(hh)
                        yy = Yp_class[1]
                        yy = torch.stack(yy, dim=0)
                        h_out1 = Conv1(yy)
                        h_out1 = CE(h_out1).cpu()
                        Hhat_list_HDCE.append(h_out1)
                        Hhat_list_SDCE.append(CE1(yy).cpu())

                    if len(Yp_class[2]):
                        hh = label_class[2]
                        hh = torch.stack(hh, dim=0)
                        Hperfect_list_HDCE.append(hh)
                        yy = Yp_class[2]
                        yy = torch.stack(yy, dim=0)
                        h_out1 = Conv2(yy)
                        h_out1 = CE(h_out1).cpu()
                        Hhat_list_HDCE.append(h_out1)
                        Hhat_list_SDCE.append(CE2(yy).cpu())


                Hhat0 = torch.cat(Hhat_list0, dim=0)
                Hhat1 = torch.cat(Hhat_list1, dim=0)
                Hhat2 = torch.cat(Hhat_list2, dim=0)
                Hhat_DCE = torch.cat(Hhat_list_DCE, dim=0)
                Hperfect = torch.cat(Hperfect_list, dim=0)
                HLS = torch.cat(HLS_list, dim=0)
                HMMSE=torch.cat(HMMSE_list,dim=0)

                Hhat_HDCE = torch.cat(Hhat_list_HDCE,dim=0)
                Hhat_FullHDCE = torch.cat(Hhat_list_SDCE,dim=0)
                Hperfect_HDCE = torch.cat(Hperfect_list_HDCE,dim=0)


                nmse0 = criterion(Hhat0, Hperfect)
                nmse1 = criterion(Hhat1, Hperfect)
                nmse2 = criterion(Hhat2, Hperfect)
                nmse_DCE = criterion(Hhat_DCE, Hperfect)
                nmse_HDCE = criterion(Hhat_HDCE,Hperfect_HDCE)
                nmse_SDCE = criterion(Hhat_FullHDCE,Hperfect_HDCE)

                nmse_LS = criterion(HLS, Hperfect)
                nmse_MMSE = criterion(HMMSE, Hperfect)

                NMSE_for_scenario0.append(nmse0.item())
                NMSE_for_scenario1.append(nmse1.item())
                NMSE_for_scenario2.append(nmse2.item())
                NMSE_for_DCE.append(nmse_DCE.item())
                NMSE_for_HDCE.append(nmse_HDCE.item())
                NMSE_for_SDCE.append(nmse_SDCE.item())
                NMSE_for_LS.append(nmse_LS.item())
                NMSE_for_MMSE.append(nmse_MMSE.item())

                #
                print(f'NMSE_for_scenario0: {NMSE_for_scenario0}')
                print(f'NMSE_for_scenario1: {NMSE_for_scenario1}')
                print(f'NMSE_for_scenario2: {NMSE_for_scenario2}')
                print(f'NMSE_for_DCE: {NMSE_for_DCE}')
                print(f'NMSE_for_HDCE: {NMSE_for_HDCE}')
                print(f'NMSE_for_SDCE: {NMSE_for_SDCE}')
                print(f'NMSE_for_LS: {NMSE_for_LS}')
                print(f'NMSE_for_MMSE: {NMSE_for_MMSE}')

        plt.figure()
        plt.plot(SNRdb, 10*np.log10(NMSE_for_scenario0), 'm>-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_scenario1), 'b^-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_scenario2), 'c<-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_DCE), 'rs-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_HDCE), 'rd-')
        plt.plot(SNRdb, 10 * np.log10(NMSE_for_SDCE), 'ro-')
        plt.plot(SNRdb, 10*np.log10(NMSE_for_LS), 'k--')
        plt.plot(SNRdb, 10 * np.log10(NMSE_for_MMSE), 'r--')
        plt.grid()


        plt.legend(['DCE network only trained on scenario 1', 'DCE network only trained on scenario 2', 'DCE network only trained on scenario 3','proposed DML based DCE network','proposed DML based HDCE network','DML based SDCE network','LS algorithm','MMSE algorithm'])


        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.xticks(np.arange(5, 16, 2))
        plt.yticks(np.arange(-20, 1, 5))
        plt.savefig('results/NMSEvsSNR_for_128_10dB_scenario'+str(self.indicator))

        plt.show()

        return 0

if __name__ == '__main__':

    gpu_list = '6'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    device = 'cuda'


    print('start to test ...')
    test = model_val()
    print('generate the figure about NMSE performance comparison for the channel scenario 1 with the compression ratio '+ chr(947)+'= 1/8')
    test.indicator = 0
    test.test_for_CE_P128_for_scenario0()
    #
    print('generate the figure about  NMSE performance comparison for the entire cell with the compression ratio '+ chr(947)+'= 1/8')
    test.indicator = -1
    test.test_for_CE_P128_for_all_scenarios()


    print('generate the figure about the accuracy of the channel scenario prediction')
    test.test_for_SC()