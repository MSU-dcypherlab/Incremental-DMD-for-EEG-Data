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
%%%%%% Threshold Levels

thresh_vec=[0.01 0.001];threshold_label={''};thresh_str_vec={''};
for thr_idx=1:length(thresh_vec)
    thresh_str_vec{thr_idx}=strcat('th',num2str(thr_idx));
    threshold_label{thr_idx}=num2str(thresh_vec(thr_idx));
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

for event_idx1=1:2
eegcarhpfsetfile=strcat(reg_data_dir,data_event_str{event_idx1},'.mat');
    for thr_idx1=1:length(thresh_vec)
        for weight_idx1=1:length(weight_vec)
            thresh=thresh_vec(thr_idx1); wt=(weight_vec(weight_idx1));
            fprintf('loading data for %s events.... \n',event_name{event_idx1});
            EEG=matfile(eegcarhpfsetfile);TTime=zeros(1,1024);
            Data=EEG.ERP;Time=EEG.time;
            test_time_f=0.40;
            [~,recon_ind_f]=min(abs(Time-test_time_f));
            test_time_0=0.20;
            [~,recon_ind_0]=min(abs(Time-test_time_0));
            
            Datax=Data(:,1:end-1);
            Datay=Data(:,2:end);
            [n_data,m_data]=size(Datax);
            Data_x_wind=Datax(:,1:wind);Data_y_wind=Datay(:,1:wind);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%% Initial DMD model
            [U1,Seg1,V1]=svd(Data_x_wind);Seg10=(diag(Seg1))';
            Dataxhat(:,1:wind)=U1*Seg1*V1';
            rtil1 = length(find(diag(Seg1)>=thresh));
            Uo=U1;Vo=V1;Sego=Seg1;rtilo=rtil1;
            Ux=Uo(:,1:rtilo);Segx=Sego(1:rtilo,1:rtilo);Vx= Vo(:,1:rtilo);
            A0=Data_y_wind*Vx*(eye(size(Segx))/Segx)*Ux';
            Ar10=A0;
            err_rms_pred=zeros(length(1:m_data-wind),1);
            err_time_pred=zeros(length(1:m_data-wind),1);
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%% Incremental Weighted DMD 

            for sample_idx=1: m_data-(wind+future_pred_samples)
                fprintf('===========================================\n');
                fprintf('iteration # %d out of %d .... \n',sample_idx,m_data-(wind+future_pred_samples));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%% Weighted Incremental SVD Update 
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
                

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%% Weighted Incremental DMD Model Update 
                Ux=Uo(:,1:rtilo);Segx=Sego(1:rtilo,1:rtilo);vs_telda=Vo(end,1:rtilo);
                Segx_inv=(eye(rtilo)/Segx);
                err1=(yu-Ar10*xu);
                Ar11=(Ar10 +err1*vs_telda* Segx_inv*Ux'*(1/wt));
                Ar10=Ar11;
                [W_Ar10,D_Ar10]=eig(Ar10);
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%% Reduced Dimention DMD model
                A=Ux'*Ar10*Ux;[W_telda,D_telda]=eig(A);lambda_telda=diag(D_telda);
                omega_telda=log(lambda_telda)/dt;
                phi=wt*Ux*W_telda;
               
               
                if (sample_idx+wind==recon_ind_f)
                    test_ind_0=wind+sample_idx+1;test_ind_f=test_ind_0+future_pred_samples-1;
                    x1=Datax(:,wind);b= pinv(phi)*x1;time=(1:recon_ind_f-wind)./FS;
                    time_dynamics=zeros(length(b),length(time));
                    for iter=1:length(time)
                        time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
                    end
                    eeg_hat=real(phi*time_dynamics);
                    eeg_recon=eeg_hat(elec_FCz_num,:);
                    time_recon=time;
                    R_y=max(Datay(elec_FCz_num,recon_ind_0:recon_ind_f))-min(Datay(elec_FCz_num,recon_ind_0:recon_ind_f));
                    err_rms_recon=(rms(eeg_hat(elec_FCz_num,recon_ind_0-wind:recon_ind_f-wind)-Datay(elec_FCz_num,recon_ind_0:recon_ind_f)))/R_y;
                    
                    dynamics_weighted_inc.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).eeg_recon=eeg_hat(elec_FCz_num,:);
                    dynamics_weighted_inc.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).time_recon=time_recon;
                    dynamics_weighted_inc.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).rms_eeg_recon=err_rms_recon;
                    
                    break
                end
                
                
               
            end
            dynamics_weighted_inc.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).rms_future_pred=err_rms_pred(1:sample_idx);
            dynamics_weighted_inc.(event_name{event_idx1}).(thresh_str_vec{thr_idx1}).(weight_str_vec{weight_idx1}).time_future_pred=err_time_pred(1:sample_idx);
            
        end
    end
    
    dynamics_weighted_inc.(event_name{event_idx1}).eegdata_recon=Datay(elec_FCz_num,wind+1:recon_ind_f);
    dynamics_weighted_inc.(event_name{event_idx1}).eegdata_pred=Datay(elec_FCz_num,test_ind_0:test_ind_f);
end


%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%%%% Plotting Predicition Error
close all
fig_num=0;
for event_idx2=1:event_idx1
    pred_rms_mat=zeros(thr_idx1,weight_idx1);
    for thr_idx2=1:thr_idx1
        fig_num=fig_num+1;
        H_pred_rms=figure(fig_num);
         set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
        for weight_idx2=1:weight_idx1
            pred_time= dynamics_weighted_inc.(event_name{event_idx2}).(thresh_str_vec{thr_idx2}).(weight_str_vec{weight_idx2}).time_future_pred;
            pred_rms= dynamics_weighted_inc.(event_name{event_idx2}).(thresh_str_vec{thr_idx2}).(weight_str_vec{weight_idx2}).rms_future_pred;
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
        ax.XLim=[pred_time(1),pred_time(end)];
        %%%%Legends
        lgd = legend(weight_legends);
        lgd.Box='on';lgd.LineWidth=2;lgd.Location='north';
        lgd.FontSize = lgd_font;lgd.FontWeight = 'b';lgd.Orientation='horizontal';
%         lgd.NumColumns =3;
        grid on
        %%%%Title
        ax.Title.String =  strcat('Incremental Weighted DMD ( \sigma_{thr} =',num2str(thresh_vec(thr_idx2)),')');
        ax.Title.FontWeight = 'bold';ax.Title.FontSize =axis_font;
        %%%%%%%%%%%%%%%%%%%
        rms_thr_fig=strcat(EEG_Pred_figures,'\',event_name{event_idx2},'_inc_weighted_DMD_pred_rms_',thresh_str_vec{thr_idx2},'.fig');
        rms_thr_png=strcat(EEG_Pred_figures,'\',event_name{event_idx2},'_inc_weighted_DMD_pred_rms_',thresh_str_vec{thr_idx2});
        saveas(H_pred_rms,rms_thr_fig);
        print(rms_thr_png,'-dpdf','-fillpage')
       
    end
end
DMD_dyn_file=strcat(mat_resultdir_all,'\','inc_weighted_DMD_EEG_Pred.mat');
save(DMD_dyn_file,'dynamics_weighted_inc');

