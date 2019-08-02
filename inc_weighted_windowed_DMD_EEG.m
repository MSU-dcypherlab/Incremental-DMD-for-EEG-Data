clc
clearvars
close all
% eeglab
% addpath('C:\Users\vaibhav\Desktop\C++\codes\files');
% addpath('C:\Users\vaibhav\Desktop\C++\codes\functions');
fprintf('loading data..... \n');
chnfilename='biosemi_64ch_loc.ced';
event_str={'correcteeg_BPF_110','erroneouseeg_BPF_110'};
event_name={'Correct','Erroneous'};event_var={'EEG_corr','EEG_telda_corr';'EEG_err','EEG_telda_err'};
thresh_vec=[0.01 0.001];thresh_str_vec={'th1','th2'};threshold_label={'0.01','0.001'};
FS=512;dt=1/FS;pre_event_time=1;
wind=floor(pre_event_time*FS);future_pred_samples=FS/8;
%reg_data_dir=strcat('E:\EEG database\Error Related Potential\data\reg_data\data_',num2str(pre_event_time),'sec\');
resultdir='C:\Users\vaibhav\Desktop\SIAM_loc\SIAMX';
resultdir_all=strcat(resultdir,'\s5');
if (~exist(resultdir_all,'dir'))
    mkdir(resultdir_all);
end

resultdir_pred_eegrms=strcat(resultdir_all,'\','EEG_Prediction_RMS');
if (~exist(resultdir_pred_eegrms,'dir'))
    mkdir(resultdir_pred_eegrms);
end
resultdir_recon_erp=strcat(resultdir_all,'\','ERP_Reconstruction');
if (~exist(resultdir_recon_erp,'dir'))
    mkdir(resultdir_recon_erp);
end
resultdir_recon_erprms=strcat(resultdir_all,'\','ERP_Reconstruction_RMS');
if (~exist(resultdir_recon_erprms,'dir'))
    mkdir(resultdir_recon_erprms);
end



elec_num=47;
%weight_str_vec={'w40','w60','w80','w100','wind512'}; weight_vec=[0.4 0.6 0.8 1];   weight_legends={'\rho=0.4','\rho=0.6','\rho=0.8','\rho=1','w=512'}; 
weight_str_vec={'w10','w20','w40','w80', 'wind512'}; weight_vec=[0.1 0.2 0.4 0.8]; weight_legends={'\rho=0.1','\rho=0.2','\rho=0.4','\rho=0.8','w=512'};
for event_idx1=1:2

    for thr_idx1=1:length(thresh_vec)
        for weight_idx1=1:length(weight_vec)
            thresh=thresh_vec(thr_idx1);thr_we_str=strcat(thresh_str_vec{thr_idx1},weight_str_vec{weight_idx1}); wt=(weight_vec(weight_idx1));
            eegcarhpfsetfile=strcat(event_str{event_idx1},'.mat');
            EEG=matfile(eegcarhpfsetfile);
            Data=EEG.ERP;Time=EEG.time;
            Time_label=cell(1,length(Time));
            for time_idx=1:length(Time)
                Time_label{time_idx}=num2str(Time(time_idx),'%.0f');
            end
             pred_time_f=0.7;
            [~,pred_ind_f]=min(abs(Time-pred_time_f));
            test_time_f=0.40;
            [~,recon_ind_f]=min(abs(Time-test_time_f));
            
            test_time_0=0.20;
            [~,recon_ind_0]=min(abs(Time-test_time_0));
            
            
            Datax=Data(:,1:end-1);
            Datay=Data(:,2:end);
            [n_data,m_data]=size(Datax);
            Data_x_wind=Datax(:,1:wind);Data_y_wind=Datay(:,1:wind);
            [U1,Seg1,V1]=svd(Data_x_wind);Seg10=(diag(Seg1))';
            Dataxhat(:,1:wind)=U1*Seg1*V1';
            rtil1 = length(find(diag(Seg1)>=thresh));
            Uo=U1;Vo=V1;Sego=Seg1;rtilo=rtil1;
            Ux=Uo(:,1:rtilo);Segx=Sego(1:rtilo,1:rtilo);Vx= Vo(:,1:rtilo);
            A0=Data_y_wind*Vx*(eye(size(Segx))/Segx)*Ux';
            Ar10=A0;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% DMD eigen values FIGURE
            [~,D_Ar10]=eig(Ar10);
            lambda=diag(D_Ar10);
            err_rms_pred=zeros(length(1:m_data-wind),1);
            err_time_pred=zeros(length(1:m_data-wind),1);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for sample_idx=1: pred_ind_f-(wind+future_pred_samples)
                fprintf('===========================================\n');
                fprintf('iteration # %d out of %d .... \n',sample_idx,m_data-wind);
                %%%%%Online Incremental DMD
                fprintf('data window # %d,updating dataset... \n',sample_idx);
                Vi=Vo;Ui=Uo;Segi=Sego;
                xu=Datax(:,sample_idx+wind); yu=Datay(:,sample_idx+wind);
                px=Ui'*xu;qx=xu-Ui*px;
                rx=norm(qx);qx=qx./rx;
                [nv,mv]=size(Vi);zu=[zeros(nv,1);1];
                pz=[Vi;zeros(1,mv)]'*zu;
                qz=zu-[Vi;zeros(1,mv)]*pz;
                rz=norm(qz);qz=qz./rz;
                Su0=[[Segi  ; zeros(1,mv)], [px*rz;0]*(1/wt)];
                [Uu1,Segu1,Vu1]=svd(Su0);
                rtilo=length(find(diag(Segu1)>=thresh));
               
                Vu2=[[Vi;zeros(1,mv)] qz]*Vu1;
                Uu2=Ui*Uu1(1:end-1,1:end-1);
                Segu2=Segu1(1:end-1,:);
                Uo=Uu2;Sego=Segu2;Vo=Vu2;
                
%                 Vu2=[[Vi;zeros(1,mv)] qz]*Vu1(:,1:end-1);
%                 Uu2=Ui*Uu1(1:end-1,1:end-1);
%                 Segu2=Segu1(1:end-1,1:end-1);
%                 Uo=Uu2;Sego=Segu2;Vo=Vu2;
               
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%% system update
                Ux=Uo(:,1:rtilo);Segx=Sego(1:rtilo,1:rtilo);vs_telda=Vo(end,1:rtilo);
                Segx_inv=(eye(rtilo)/Segx);
                err1=(yu-Ar10*xu);
                Ar11=(Ar10 +err1*vs_telda* Segx_inv*Ux'*(1/wt));
                Ar10=Ar11;
                [W_Ar10,D_Ar10]=eig(Ar10);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%% DMD model
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                A=Ux'*Ar10*Ux;[W_telda,D_telda]=eig(A);lambda_telda=diag(D_telda);
                omega_telda=log(lambda_telda)/dt;
                phi=wt*Ux*W_telda;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                test_ind_0=wind+sample_idx+1;test_ind_f=test_ind_0+future_pred_samples-1;
                x_ind_0=wind+sample_idx-del_wind;
                x1=Datax(:,x_ind_0);b= pinv(phi)*x1;time=(1:test_ind_f-x_ind_0)./FS;
                time_dynamics=zeros(length(b),length(time));
                for iter=1:length(time)
                    time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
                end
                eeg_hat=real(phi*time_dynamics);eeg_hat=eeg_hat(:,del_wind+1:length(time));
                err_time_pred(sample_idx)=Time(sample_idx+wind);R_y=max(Datay(elec_num,test_ind_0:test_ind_f))-min(Datay(elec_num,test_ind_0:test_ind_f));
                err_rms_pred(sample_idx)=(rms(eeg_hat(elec_num,:)-Datay(elec_num,test_ind_0:test_ind_f)))/R_y;

                if (sample_idx+wind==recon_ind_f)
                    x1=Datax(:,wind);b= pinv(phi)*x1;time=(1:recon_ind_f-wind)./FS;
                    time_dynamics=zeros(length(b),length(time));
                    for iter=1:length(time)
                        time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
                    end
                    eeg_hat=real(phi*time_dynamics);
                    eeg_recon=eeg_hat(elec_num,:);
                    time_recon=time;
                    
                    R_y=max(Datay(elec_num,recon_ind_0:recon_ind_f))-min(Datay(elec_num,recon_ind_0:recon_ind_f));
                    err_rms_recon=(rms(eeg_hat(elec_num,recon_ind_0-wind:recon_ind_f-wind)-Datay(elec_num,recon_ind_0:recon_ind_f)))/R_y;
                    dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).eeg_recon=eeg_hat(elec_num,:);
                    dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).time_recon=time_recon;
                    dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).rms_eeg_recon=err_rms_recon;
                    
                end

            end
            dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).rms_future_pred=err_rms_pred(1:sample_idx);
            dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).time_future_pred=err_time_pred(1:sample_idx);
            
            
            
           
            
            
        end
    end
end





pre_event_time_vec=[1 2];

for pre_idx=1:1
    pre_event_time=pre_event_time_vec(pre_idx);
    weight_idx1=weight_idx1+1;
    wind=floor(pre_event_time*FS);
    reg_data_dir=strcat('E:\EEG database\Error Related Potential\data\reg_data\data_',num2str(pre_event_time),'sec\');
    
    for event_idx1=1:2
        
        
        for thr_idx1=1:length(thresh_vec)
            thresh=thresh_vec(thr_idx1);
            eegcarhpfsetfile=strcat(event_str{event_idx1},'.mat');
            EEG=matfile(eegcarhpfsetfile);
            Data=EEG.ERP;Time=EEG.time;
            Time_label=cell(1,length(Time));
            for time_idx=1:length(Time)
                Time_label{time_idx}=num2str(Time(time_idx),'%.0f');
            end
            Datax=Data(:,1:end-1);
            Datay=Data(:,2:end);
            [n_data,m_data]=size(Datax);
            Data_x_wind=Datax(:,1:wind);Data_y_wind=Datay(:,1:wind);
            [U1,Seg1,V1]=svd(Data_x_wind);Seg10=(diag(Seg1))';
            Dataxhat(:,1:wind)=U1*Seg1*V1';
            rtilo = length(find(diag(Seg1)>=thresh));
            Uo=U1;Vo=V1;Sego=Seg1;
            Ux=Uo(:,1:rtilo);Segx=Sego(1:rtilo,1:rtilo);Vx= Vo(:,1:rtilo);
            Ar10=Data_y_wind*Vx*(eye(size(Segx))/Segx)*Ux';
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%% DMD eigen values FIGURE
            [~,D_Ar10]=eig(Ar10);
            lambda=diag(D_Ar10);
            err_rms_pred=zeros(length(1:m_data-wind),1);
            err_time_pred=zeros(length(1:m_data-wind),1);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            recon_errd=zeros(m_data,1);recon_erru=zeros(m_data,1);
            for sample_idx=1:pred_ind_f-(wind+future_pred_samples)
                fprintf('===========================================\n');
                fprintf('iteration # %d out of %d .... \n',sample_idx,m_data-wind);
                %%%%%%%%%%%%%%%%%%%%%
                %%%% decrmental DMD
                Vi=Vo;Ui=Uo;Segi=Sego;
                xd=Datax(:,sample_idx);
                px=Ui'*xd;
                [nv,~]=size(Vi);zd=[1;zeros(nv-1,1)];
                pz=Vi'*zd;
                Sd0=Segi-px*pz';
                [Ud1,Segd1,Vd1]=svd(Sd0);
                Vd2=Vi*Vd1;
                Segd2=Segd1;
                Ud2=Ui*Ud1;
                Data_xdhat_wind=Ud2*Segd2*Vd2';
                Data_xd_wind=[zeros(n_data,1) Datax(:,sample_idx+1:wind+sample_idx-1)];
                Uo=Ud2;Vo=Vd2;Sego=Segd2;
                recon_errd(sample_idx)= norm(norm(Data_xdhat_wind-Data_xd_wind));
                %%%%%%%%%%%%%%%%%%%%%
                %%%% incrmental DMD
                Vi=Vo;Ui=Uo;Segi=Sego;
                xu=Datax(:,sample_idx+wind);
                px=Ui'*xu;qx=xu-Ui*px;
                rx=norm(qx);qx=qx./rx;
                [nv,mv]=size(Vi);zu=[zeros(nv,1);1];
                pz=[Vi;zeros(1,mv)]'*zu;
                qz=zu-[Vi;zeros(1,mv)]*pz;
                rz=norm(qz);qz=qz./rz;
                Su0=[Segi , px*rz];
                [Uu1,Segu1,Vu1]=svd(Su0);
                Vu2=[[Vi;zeros(1,mv)] qz]*Vu1;
                Uu2=Ui*Uu1;
                Segu2=Segu1;
                Uo=Uu2;
                %%%%%%Full
                %Vo=Vu2(2:end,:);Sego=Segu2;
                
                %%%%%%Reduced
                Vo=Vu2(2:end,1:wind);Sego=Segu2(:,1:wind);
                
                
                Data_xu_hat=Uo*Sego*Vo';
                Data_xu_wind=Datax(:,sample_idx+1:wind+sample_idx);
                recon_erru(sample_idx)= norm(norm(Data_xu_hat-Data_xu_wind));
                
                rtilo = length(find(diag(Seg1)>=thresh)); Ux=Uo(:,1:rtilo);Vx= Vo(:,1:rtilo);
                Segx=Sego(1:rtilo,1:rtilo);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%% system update1
%               Segx_inv=(eye(rtilo)/Segx);
%               Data_y_wind=Datay(:,sample_idx+1:sample_idx+wind);Data_x_wind=Datax(:,sample_idx+1:sample_idx+wind);
%               err1=Data_y_wind-Ar10*Data_x_wind;
%               Ar11=Ar10+err1*(Vx*Segx_inv*Ux');
%               Ar10=Ar11;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%% system update2
                vs_telda=Vo(end,1:rtilo);Segx_inv=(eye(rtilo)/Segx);
                yu=Datay(:,sample_idx+wind);
                err1=(yu-Ar10*xu);
                Ar11=Ar10 +err1*vs_telda* Segx_inv*Ux';
                Ar10=Ar11;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%% system update3
                %                 Ar11=Data_y_wind*(Vx*Segx_inv*Ux');
                %                 Ar10=Ar11;
                
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%% DMD model
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                A=Ux'*Ar10*Ux;[W_telda,D_telda]=eig(A);lambda_telda=diag(D_telda);
                omega_telda=log(lambda_telda)/dt;
                phi=Ux*W_telda;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                test_ind_0=wind+sample_idx+1;test_ind_f=test_ind_0+future_pred_samples-1;
                x_ind_0=wind+sample_idx-del_wind; 
                x1=Datax(:,x_ind_0);b= pinv(phi)*x1;time=(1:test_ind_f-x_ind_0)./FS;
                time_dynamics=zeros(length(b),length(time));
                for iter=1:length(time)
                    time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
                end
                eeg_hat=real(phi*time_dynamics);eeg_hat=eeg_hat(:,del_wind+1:length(time));
                err_time_pred(sample_idx)=Time(sample_idx+wind);R_y=max(Datay(elec_num,test_ind_0:test_ind_f))-min(Datay(elec_num,test_ind_0:test_ind_f));
                err_rms_pred(sample_idx)=(rms(eeg_hat(elec_num,:)-Datay(elec_num,test_ind_0:test_ind_f)))/R_y;
                
                
                
                if (sample_idx+wind==recon_ind_f)
                    x1=Datax(:,wind);b= pinv(phi)*x1;time=(1:recon_ind_f-wind)./FS;
                    time_dynamics=zeros(length(b),length(time));
                    for iter=1:length(time)
                        time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
                    end
                    eeg_hat=real(phi*time_dynamics);
                    eeg_recon=eeg_hat(elec_num,:);
                    time_recon=time;
                    R_y=max(Datay(elec_num,recon_ind_0:recon_ind_f))-min(Datay(elec_num,recon_ind_0:recon_ind_f));
                    err_rms_recon=(rms(eeg_hat(elec_num,recon_ind_0-wind:recon_ind_f-wind)-Datay(elec_num,recon_ind_0:recon_ind_f)))/R_y;
                    
                    dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).eeg_recon=eeg_hat(elec_num,:);
                    dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).time_recon=time_recon;
                    dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).rms_eeg_recon=err_rms_recon;
                end
                
                
            end
            dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).rms_future_pred=err_rms_pred(1:sample_idx);
            dynamics.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).time_future_pred=err_time_pred(1:sample_idx);
 
        end
        dynamics.(event_name{event_idx1}).eegdata_recon=Datay(elec_num,wind+1:recon_ind_f);
        dynamics.(event_name{event_idx1}).eegdata_pred=Datay(elec_num,test_ind_0:test_ind_f);
    end
    
    
end



































%%
close all
fig_num=0;
weight_legends1={'\rho=0.1','\rho=0.2','\rho=0.4','\rho=0.8','w=512'};
weight_legends2={'\rho=0.1','\rho=0.2','\rho=0.4','\rho=0.8','w=512','original ERP'};
for event_idx2=1:event_idx1
    pred_rms_mat=zeros(thr_idx1,weight_idx1);
    for thr_idx2=1:thr_idx1
        fig_num=fig_num+1;
        H_pred_rms=figure(fig_num);
         set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
        for weight_idx2=1:weight_idx1
            pred_time= dynamics.(event_name{event_idx2}).(thresh_str_vec{thr_idx2}).(weight_str_vec{weight_idx2}).time_future_pred;
            pred_rms= dynamics.(event_name{event_idx2}).(thresh_str_vec{thr_idx2}).(weight_str_vec{weight_idx2}).rms_future_pred;
            pred_rms_mat(thr_idx2,weight_idx2)= sum(pred_rms);
            h_erp=semilogy(pred_time,pred_rms, 'LineWidth',3);
            hold on
        end
        hold off
        %%%%Axes
        axis_font=30;lgd_font=36;
        ax=gca;
        ax.YLabel.String = 'Normalized RMS Error';ax.XLabel.String = 'Time(sec)';
        ax.FontSize = axis_font;ax.FontWeight = 'bold';
        ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
        ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
        ax.YLim=[10^-2,10^2];
        %%%%Legends
        lgd = legend(weight_legends1);
        lgd.Box='on';lgd.LineWidth=2;lgd.Location='north';
        lgd.FontSize = lgd_font;lgd.FontWeight = 'b';lgd.Orientation='horizontal';
        lgd.NumColumns =3;
        grid on
        %%%%Title
        ax.Title.String =  strcat('Incremental DMD, \sigma_{thr} =', num2str(thresh_vec(thr_idx2)));
        ax.Title.FontWeight = 'bold';ax.Title.FontSize =axis_font;
        %%%%%%%%%%%%%%%%%%%
        rms_thr_fig=strcat(resultdir_pred_eegrms,'\',event_name{event_idx2},'_incDMD_pred_rms_',thresh_str_vec{thr_idx2},'.fig');
        rms_thr_png=strcat(resultdir_pred_eegrms,'\',event_name{event_idx2},'_incDMD_pred_rms_',thresh_str_vec{thr_idx2});
        saveas(H_pred_rms,rms_thr_fig);
        print(rms_thr_png,'-dpdf','-fillpage')
       
    end
    axis_font=30;lgd_font=30;
    fig_num=fig_num+1;
    H_pred_rms_bar=figure(fig_num);
    set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
    bar(pred_rms_mat)
    %%%%Axes
    ax=gca;
    ax.YLabel.String = '1-Norm of Prediction Error';ax.XLabel.String ='\sigma_{thr}';
    ax.FontSize = axis_font;ax.FontWeight = 'b';
    ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
    ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
    ax.XTickLabel=threshold_label;
    ax.YLim=[0,150];
    grid on
    %%%%Legends
    lgd = legend(weight_legends1);
    lgd.Box='on';lgd.LineWidth=2;lgd.Location='north';
    lgd.FontSize = lgd_font;lgd.FontWeight = 'b';lgd.Orientation='horizontal';
    lgd.NumColumns =3;
    %%%%Title
    ax.Title.String = sprintf('%s Event',event_name{event_idx2});
    ax.Title.FontWeight = 'bold';ax.Title.FontSize = axis_font;
    %%%% Saving Image
    eegrms_thr_fig=strcat(resultdir_pred_eegrms,'\',event_name{event_idx2},'_incDMD_eegpred_rms.fig');
    eegrms_thr_png=strcat(resultdir_pred_eegrms,'\',event_name{event_idx2},'_incDMD_eegpred_rms');
    saveas(H_pred_rms_bar,eegrms_thr_fig);
    print(eegrms_thr_png,'-dpdf','-fillpage')
    %print(erprms_thr_png,'-dpng')
    
    
    
    
end
%%

ymax=[3 10];ymin=[-2,-4];
for event_idx2=1:event_idx1
    recon_rms_mat=zeros(thr_idx1,weight_idx1);
    for thr_idx2=1:thr_idx1
        fig_num=fig_num+1;
        H_recon_erp=figure(fig_num);
        set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
        for weight_idx2=1:weight_idx1
            recon_rms_mat(thr_idx2,weight_idx2)=dynamics.(event_name{event_idx2}).(thresh_str_vec{thr_idx2}).(weight_str_vec{weight_idx2}).rms_eeg_recon;
            recon_time= dynamics.(event_name{event_idx2}).(thresh_str_vec{thr_idx2}).(weight_str_vec{weight_idx2}).time_recon;
            recon_eeg= dynamics.(event_name{event_idx2}).(thresh_str_vec{thr_idx2}).(weight_str_vec{weight_idx2}).eeg_recon;
            plot(recon_time,recon_eeg, 'LineWidth',3);
            hold on
            
        end
        
        plot(recon_time,dynamics.(event_name{event_idx2}).eegdata_recon, 'LineWidth',3);
        hold off
        
        %%%%Axes
        axis_font=30;lgd_font=28;
        ax=gca;
        ax.YLabel.String = 'Potential(\muV)';ax.XLabel.String = 'Time(sec)';
        ax.FontSize = axis_font;ax.FontWeight = 'bold';
        ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
        ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
        ax.XLim=[Time(wind+1),Time(recon_ind_f)];
        ax.YLim=[ymin(event_idx2),ymax(event_idx2)];
        grid on
        %%%%Legends
        lgd = legend(weight_legends2);
        lgd.Box='on';lgd.LineWidth=2;lgd.Location='north';
        lgd.FontSize = lgd_font;lgd.FontWeight = 'b';lgd.Orientation='horizontal';
        lgd.NumColumns =3;
        %%%%Title
        ax.Title.String = strcat('Incremental DMD, \sigma_{thr} =', num2str(thresh_vec(thr_idx2)));
        ax.Title.FontWeight = 'bold';ax.Title.FontSize = axis_font;
        %%%%%%%
        rms_thr_fig=strcat(resultdir_recon_erp,'\',event_name{event_idx2},'_incDMD_recon_',thresh_str_vec{thr_idx2},'.fig');
        rms_thr_png=strcat(resultdir_recon_erp,'\',event_name{event_idx2},'_incDMD_recon_',thresh_str_vec{thr_idx2});
        saveas(H_recon_erp,rms_thr_fig);
        print(rms_thr_png,'-dpdf','-fillpage')

    end
    
    axis_font=30;lgd_font=30;
    fig_num=fig_num+1;
    H_recon_rms_bar=figure(fig_num);
    set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
    bar(recon_rms_mat)
    %%%%Axes
    ax=gca;
    ax.YLabel.String = 'Normalized RMS Error';ax.XLabel.String ='\sigma_{thr}';
    ax.FontSize = axis_font;ax.FontWeight = 'b';
    ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
    ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
    ax.XTickLabel=threshold_label;
    ax.YLim=[0,max(max(recon_rms_mat))+0.1];
    grid on
    %%%%Legends
    lgd = legend(weight_legends1);
    lgd.Box='on';lgd.LineWidth=2;lgd.Location='north';
    lgd.FontSize = lgd_font;lgd.FontWeight = 'b';lgd.Orientation='horizontal';
    lgd.NumColumns =3;
    %%%%Title
    ax.Title.String = sprintf('%s Event',event_name{event_idx2});
    ax.Title.FontWeight = 'bold';ax.Title.FontSize = axis_font;
    %%%% Saving Image
    erprms_thr_fig=strcat(resultdir_recon_erprms,'\',event_name{event_idx2},'_incDMD_recon_rms.fig');
    erprms_thr_png=strcat(resultdir_recon_erprms,'\',event_name{event_idx2},'_incDMD_recon_rms');
    saveas(H_recon_rms_bar,erprms_thr_fig);
    print(erprms_thr_png,'-dpdf','-fillpage')
    %print(erprms_thr_png,'-dpng')
    
    
    
    
    
    
    
   
end


