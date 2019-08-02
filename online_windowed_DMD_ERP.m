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

ERP_Recon_figures=strcat(figures_resultdir_all,'\ERP Reconstruction');
if (~exist(ERP_Recon_figures,'dir'))
    mkdir(ERP_Recon_figures);
end

ERP_Pred_figures=strcat(figures_resultdir_all,'\ERP Prediction');
if (~exist(ERP_Pred_figures,'dir'))
    mkdir(ERP_Pred_figures);
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
        
        test_time_f=0.40;
        [~,erp_ind_f]=min(abs(Time-test_time_f));erp_ind_f=erp_ind_f+wind;
        test_time_0=0.20;
        [~,erp_ind_0]=min(abs(Time-test_time_0));erp_ind_0=erp_ind_0+wind;
        
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
            %%%%%ERP Prediction
            
            if (sample_idx+wind==erp_ind_0)
                x1=Datax(:,erp_ind_0-1);b= pinv(phi)*x1;time=(1:erp_ind_f-erp_ind_0+1)./FS;
                time_dynamics=zeros(length(b),length(time));
                for iter=1:length(time)
                    time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
                end
                erp_hat=real(phi*time_dynamics);
                R_y=max(Datay(elec_FCz_num,erp_ind_0:erp_ind_f))-min(Datay(elec_FCz_num,erp_ind_0:erp_ind_f));
                erp_pred_rms=(rms(erp_hat(elec_FCz_num,:)-Datay(elec_FCz_num,erp_ind_0:erp_ind_f)))/R_y;
                eeg_hat_best=erp_hat(elec_FCz_num,:);
                dynamics_windowed_online.(event_name{event_idx1}).rms_erp_pred=erp_pred_rms;
                
                x1=Datax(:,wind);b= pinv(phi)*x1;time=(1:erp_ind_f-wind)./FS;
                time_dynamics=zeros(length(b),length(time));
                for iter=1:length(time)
                    time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
                end
                erp_hat=real(phi*time_dynamics); erp_hat=erp_hat(elec_FCz_num,:);
                dynamics_windowed_online.(event_name{event_idx1}).erp_pred=erp_hat';
                dynamics_windowed_online.(event_name{event_idx1}).time_pred=time;
            end
            
        end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%ERP Reconstruction
        x1=Datax(:,wind);b= pinv(phi)*x1;time=(1:erp_ind_f-wind)./FS;
        time_dynamics=zeros(length(b),length(time));
        for iter=1:length(time)
            time_dynamics(:,iter)=(b.*exp(omega_telda*time(iter)));
        end
        erp_hat=real(phi*time_dynamics);
        eeg_recon=erp_hat(elec_FCz_num,:);
        time_recon=time;
        
        R_y=max(Datay(elec_FCz_num,erp_ind_0:erp_ind_f))-min(Datay(elec_FCz_num,erp_ind_0:erp_ind_f));
        err_rms_recon=(rms(erp_hat(elec_FCz_num,erp_ind_0-wind:erp_ind_f-wind)-Datay(elec_FCz_num,erp_ind_0:erp_ind_f)))/R_y;
        
        dynamics_windowed_online.(event_name{event_idx1}).erp_recon=erp_hat(elec_FCz_num,:);
        dynamics_windowed_online.(event_name{event_idx1}).time_recon=time_recon;
        dynamics_windowed_online.(event_name{event_idx1}).rms_erp_recon=err_rms_recon;
        
        dynamics_windowed_online.(event_name{event_idx1}).eegdata_rec=Datay(elec_FCz_num,wind+1:erp_ind_f);
    
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Plotting Predicted ERP
close all
fig_num=0;
ymax=[5 10];ymin=[-2,-5];
pred_rms_mat=zeros(1,event_idx1);
for event_idx2=1:event_idx1 
    fig_num=fig_num+1;
    H_pred_erp=figure(fig_num);
    set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
    pred_rms_mat(event_idx2)=dynamics_windowed_online.(event_name{event_idx2}).rms_erp_pred;
    pred_time= dynamics_windowed_online.(event_name{event_idx2}).time_pred;
    pred_erp= dynamics_windowed_online.(event_name{event_idx2}).erp_pred;
    plot(pred_time,pred_erp, 'LineWidth',3);
    hold on
    plot(pred_time,dynamics_windowed_online.(event_name{event_idx2}).eegdata_rec, 'LineWidth',3); 
    hold off
    %%%%Axes
    axis_font=30;lgd_font=28;
    ax=gca;
    ax.YLabel.String = 'Potential(\muV)';ax.XLabel.String = 'Time(sec)';
    ax.FontSize = axis_font;ax.FontWeight = 'bold';
    ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
    ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
    ax.XLim=[pred_time(1),pred_time(end)];
    ax.YLim=[ymin(event_idx2),ymax(event_idx2)];
    grid on
    %%%%Legends
    lgd = legend('windowed','original ERP');
    lgd.Box='on';lgd.LineWidth=2;lgd.Location='north';
    lgd.FontSize = lgd_font;lgd.FontWeight = 'b';lgd.Orientation='horizontal';
    %lgd.NumColumns =3;
    %%%%Title
    ax.Title.String = sprintf('Online Windowed DMD for %s Event',event_name{event_idx2});
    ax.Title.FontWeight = 'bold';ax.Title.FontSize = axis_font;
    %%%%%%%
    rms_thr_fig=strcat(ERP_Pred_figures,'\',event_name{event_idx2},'_online_windowed_DMD_pred.fig');
    rms_thr_pdf=strcat(ERP_Pred_figures,'\',event_name{event_idx2},'_online_windowed_DMD_pred');
    saveas(H_pred_erp,rms_thr_fig);
    print(rms_thr_pdf,'-dpdf','-fillpage')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
end
axis_font=30;
fig_num=fig_num+1;
H_pred_rms_bar=figure(fig_num);
set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
bar(pred_rms_mat)
%%%%Axes
ax=gca;
ax.YLabel.String = 'Normalized RMS Error';
ax.FontSize = axis_font;ax.FontWeight = 'b';
ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
ax.XTickLabel=event_name;
ax.YLim=[0,max(max(pred_rms_mat))+0.2];
grid on
%%%%Title
ax.Title.String = strcat('Online Windowed DMD');
ax.Title.FontWeight = 'bold';ax.Title.FontSize = axis_font;
%%%% Saving Image
erprms_thr_fig=strcat(ERP_Pred_figures,'\','online_windowed_DMD_pred_rms.fig');
erprms_thr_pdf=strcat(ERP_Pred_figures,'\','online_windowed_DMD_pred_rms');
saveas(H_pred_rms_bar,erprms_thr_fig);
print(erprms_thr_pdf,'-dpdf','-fillpage')
    
    





%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Plotting Reconstructed ERP

ymax=[3 10];ymin=[-2,-4];
recon_rms_mat=zeros(1,event_idx1);
for event_idx2=1:event_idx1
    
    fig_num=fig_num+1;
    H_recon_erp=figure(fig_num);
    set(gcf,'PaperPositionMode', 'manual','Position',get(0, 'Screensize'),'PaperOrientation', 'landscape');
    recon_rms_mat(event_idx2)=dynamics_windowed_online.(event_name{event_idx2}).rms_erp_recon;
    recon_time= dynamics_windowed_online.(event_name{event_idx2}).time_recon;
    recon_eeg= dynamics_windowed_online.(event_name{event_idx2}).erp_recon;
    plot(recon_time,recon_eeg, 'LineWidth',3);
    hold on
    plot(recon_time,dynamics_windowed_online.(event_name{event_idx2}).eegdata_rec, 'LineWidth',3);
    hold off
    %%%%Axes
    axis_font=30;lgd_font=28;
    ax=gca;
    ax.YLabel.String = 'Potential(\muV)';ax.XLabel.String = 'Time(sec)';
    ax.FontSize = axis_font;ax.FontWeight = 'bold';
    ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
    ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
    ax.XLim=[recon_time(1),recon_time(end)];
    ax.YLim=[ymin(event_idx2),ymax(event_idx2)];
    grid on
    %%%%Legends
    lgd = legend('windowed','original ERP');
    lgd.Box='on';lgd.LineWidth=2;lgd.Location='north';
    lgd.FontSize = lgd_font;lgd.FontWeight = 'b';lgd.Orientation='horizontal';
    %lgd.NumColumns =3;
    %%%%Title
    ax.Title.String = sprintf('Online Windowed DMD for %s Event',event_name{event_idx2});
    ax.Title.FontWeight = 'bold';ax.Title.FontSize = axis_font;
    %%%%%%%
    rms_thr_fig=strcat(ERP_Recon_figures,'\',event_name{event_idx2},'_online_windowed_DMD_recon');
    rms_thr_png=strcat(ERP_Recon_figures,'\',event_name{event_idx2},'_online_windowed_DMD_recon');
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
ax.YLabel.String = 'Normalized RMS Error';
ax.FontSize = axis_font;ax.FontWeight = 'b';
ax.YLabel.FontSize=axis_font; ax.XLabel.FontSize=axis_font;
ax.YLabel.FontWeight='b'; ax.XLabel.FontWeight='b';
ax.XTickLabel=event_name;
ax.YLim=[0,max(max(recon_rms_mat))+0.1];
grid on
%%%%Title
ax.Title.String = strcat('Online Windowed DMD');
ax.Title.FontWeight = 'bold';ax.Title.FontSize = axis_font;
%%%% Saving Image
erprms_thr_fig=strcat(ERP_Recon_figures,'\','online_windowed_DMD_recon_rms.fig');
erprms_thr_png=strcat(ERP_Recon_figures,'\','online_windowed_DMD_recon_rms');
saveas(H_recon_rms_bar,erprms_thr_fig);
print(erprms_thr_png,'-dpdf','-fillpage')


