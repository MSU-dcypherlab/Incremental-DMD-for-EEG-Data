clc
clearvars
close all



resultdir_all='Results';
if (~exist(resultdir_all,'dir'))
    mkdir(resultdir_all);
end
figures_resultdir_all=strcat(resultdir_all,'\Figures');
if (~exist(figures_resultdir_all,'dir'))
    mkdir(figures_resultdir_all);
end

mat_resultdir_all=strcat(resultdir_all,'\mat files');
if (~exist(mat_resultdir_all,'dir'))
    mkdir(mat_resultdir_all);
end

EEG_Pred_figures=strcat(figures_resultdir_all,'\EEG Prediction');
if (~exist(EEG_Pred_figures,'dir'))
    mkdir(EEG_Pred_figures);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Weights

weight_vec=[0.1 0.2 0.4 0.8];weight_str_vec={''}; weight_legends={''};
for w_idx=1:length(weight_vec)
    weight_str_vec{w_idx}=strcat('w',num2str(weight_vec(w_idx)*100));
    weight_legends{w_idx}=strcat('\rho=',num2str(weight_vec(w_idx)));
end


FS=512;dt=1/FS;initail_pre_event_time=1;
wind=floor(initail_pre_event_time*FS);future_pred_samples=FS/8;
elec_FCz_num=47;
data_event_str={'correcteeg_BPF_110','erroneouseeg_BPF_110'};
event_name={'Correct','Erroneous'};
reg_data_dir=strcat('processed data\');

for event_idx1=1:length(event_name)
    eegcarhpfsetfile=strcat(reg_data_dir,data_event_str{event_idx1},'.mat');
    
        fprintf('loading data for %s events.... \n',event_name{event_idx1});
        EEG=matfile(eegcarhpfsetfile);
        Data=EEG.ERP;Time=EEG.time;
        Datax=Data(:,1:end-1);
        Datay=Data(:,2:end);
        [n_data,m_data]=size(Datax);
        Data_x_wind=Datax(:,1:wind);Data_y_wind=Datay(:,1:wind);
        alpha_weight=10;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% Initial DMD model
        cov_A00=alpha_weight*eye(n_data);Ar00=zeros(n_data);
        for sample_idx0=1:wind
            xu=Data_x_wind(:,sample_idx0); yu=Data_y_wind(:,sample_idx0);
            
            err1=(yu-Ar00*xu);
            gamma=1/(1+ xu'*cov_A00*xu);
            
            Ar01=Ar00 +gamma*err1*xu'*cov_A00;
            cov_A01=cov_A00-gamma*(cov_A00*xu)*(xu'*cov_A00');
            
            Ar00=Ar01;
            cov_A00=cov_A01;
        end
        Ar10= Ar00;cov_A0=cov_A00;
        err_rms_pred=zeros(length(1:m_data-wind),1);
        err_time_pred=zeros(length(1:m_data-wind),1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% Online Weighted DMD
        
        for sample_idx=1: m_data-(wind+future_pred_samples)
            fprintf('iteration # %d out of %d .... \n',sample_idx,m_data-(wind+future_pred_samples));
            UX=[Datax(:,sample_idx)  Datax(:,sample_idx+wind)];
            VX=[Datay(:,sample_idx)  Datay(:,sample_idx+wind)];
            C=[-1 0;0 1];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%% Weighted Online DMD Model Update
            Err1=(VX-Ar10*UX);
            Gamma=eye(2)/((eye(2)/(C))+UX'*cov_A0*UX);
            Ar11=Ar10+Err1*Gamma*UX'*cov_A0;
            cov_A1=cov_A0-(cov_A0*UX)*Gamma*(UX'*cov_A0);
            Ar10=Ar11;
            cov_A0=cov_A1;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%% DMD model
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            A=Ar10;[W_telda,D_telda]=eig(A);lambda_telda=diag(D_telda);
            omega_telda=log(lambda_telda)/dt;
            phi=W_telda;
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%% Future EEG Prediction
            test_ind_0=wind+sample_idx+1;test_ind_f=test_ind_0+future_pred_samples-1;
            x_ind_0=wind+sample_idx;
            x1=Datax(:,x_ind_0);b= pinv(phi)*x1;time=(1:test_ind_f-x_ind_0)./FS;
            time_dynamics=zeros(length(b),length(time));
            for iter=1:length(time)
                time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
            end
            eeg_hat=real(phi*time_dynamics);eeg_hat=eeg_hat(:,1:length(time));
            err_time_pred(sample_idx)=(sample_idx/FS);R_y=max(Datay(elec_FCz_num,test_ind_0:test_ind_f))-min(Datay(elec_FCz_num,test_ind_0:test_ind_f));
            err_rms_pred(sample_idx)=(rms(eeg_hat(elec_FCz_num,:)-Datay(elec_FCz_num,test_ind_0:test_ind_f)))/R_y;
            
        end
        dynamics_windowed_online.(event_name{event_idx1}).rms_future_pred=err_rms_pred(1:sample_idx);
        dynamics_windowed_online.(event_name{event_idx1}).time_future_pred=err_time_pred(1:sample_idx);
        
   
    
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Plotting Predicition Error
close all
fig_num=0;
for event_idx2=1:event_idx1
    fig_num=fig_num+1;
    H_pred_rms=figure(fig_num);
    set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
    pred_time=  dynamics_windowed_online.(event_name{event_idx2}).time_future_pred;
    pred_rms=  dynamics_windowed_online.(event_name{event_idx2}).rms_future_pred;
    h_erp=semilogy(pred_time,pred_rms, 'LineWidth',3);
    %%%%Axes
    axis_font=30;lgd_font=36;
    ax=gca;
    ax.YLabel.String = 'Normalized RMS Error';ax.XLabel.String = 'Time(sec)';
    ax.FontSize = axis_font;ax.FontWeight = 'bold';
    ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
    ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
    ax.YLim=[10^-2,10^2];
    ax.XLim=[pred_time(1),pred_time(end)];
    grid on
    %%%%Title
    ax.Title.String =  strcat('Online windowed DMD');
    ax.Title.FontWeight = 'bold';ax.Title.FontSize =axis_font;
    %%%%%%%%%%%%%%%%%%%
    rms_thr_fig=strcat(EEG_Pred_figures,'\',event_name{event_idx2},'_online_windowed_DMD_pred_rms.fig');
    rms_thr_png=strcat(EEG_Pred_figures,'\',event_name{event_idx2},'_online_windowed_DMD_pred_rms');
    saveas(H_pred_rms,rms_thr_fig);
    print(rms_thr_png,'-dpdf','-fillpage')
    
    
end
DMD_dyn_file=strcat(mat_resultdir_all,'\','online_windowed_DMD_EEG_Pred.mat');
save(DMD_dyn_file,'dynamics_windowed_online');

