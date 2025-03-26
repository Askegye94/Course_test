
clear all; close all;

SUBJ = dir('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_treadmill_study\');

subjects = {SUBJ.name};
subjects(contains({SUBJ.name},'S')==0)=[];
Conditions = {'walk35';'walk45';'walk55'};

for i=1:length(subjects)
load(['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_treadmill_study\',char(subjects(:,i))]);
if i==7
    Conditions={'walk35';'walk45'};

elseif i==12
    Conditions={'walk35';'walk55'};
end
if i >7 && i<12
    Conditions = {'walk35';'walk45';'walk55'};
end

for p= 1:length(Conditions)

P=char(subjects(:,i));
P=P(1:end-9);
    [~,MS_QTM] = findpeaks(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi,"MinPeakHeight",35); % finds midswing peaks
    if Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi(MS_QTM(1))>80
    MS_QTM(1)=[]; % remove synch peak
    end
    MS_QTM(end)=[]; % makes sure that if MS is last frame, there is no error in the following code

    for q= 1:length(MS_QTM)
        TD_QTM(q,1) = MS_QTM(q) + find(sign(diff(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi(MS_QTM(q):end,1)))==1,1,"first"');
    end
    % length(TD_QTM)
    
%     figure;
% % 
%     subplot(2,1,1);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi,'r'); hold on;
% % subplot(3,3,2);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.fron); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.fron,'r'); hold on;
% subplot(3,3,3);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.tran); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.tran,'r'); hold on;
% subplot(3,3,4);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightHip.sagi); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightHip.sagi,'r'); hold on;
% subplot(3,3,5);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightHip.fron); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightHip.fron,'r'); hold on;
% subplot(3,3,6);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightHip.tran); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightHip.tran,'r'); hold on;
% subplot(3,3,7);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightAnkle.sagi); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightAnkle.sagi,'r'); hold on;
% subplot(3,3,8);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightAnkle.fron); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightAnkle.fron,'r'); hold on;
% subplot(3,3,9);
%     plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightAnkle.tran); hold on;
%     plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightAnkle.tran,'r'); hold on;
% title([P,char(Conditions(p,:))])
%     for r=1:length(TD_QTM)
%     xline(TD_QTM(r))
%     end

%% this part applies cross correlation to detect time shifts in the data and then matches the 2 data
% sets according to the best correlation shift
[r,lags]=xcorr(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi,Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi);

[~,delay_temp]=max(r);
delay=lags(delay_temp);
delay_output(i,p)=delay;
fields=fieldnames(Data.(char(P)).(char(Conditions(p,:))).Suit);
fields1=fieldnames(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee);
if delay>0
    for m=1:length(fields)
        for n=1:length(fields1)
    Data.(char(P)).(char(Conditions(p,:))).QTM.(char(fields(m,:))).(char(fields1(n,:)))=Data.(char(P)).(char(Conditions(p,:))).QTM.(char(fields(m,:))).(char(fields1(n,:)))(delay:end,:);
    Data.(char(P)).(char(Conditions(p,:))).Suit.(char(fields(m,:))).(char(fields1(n,:)))=Data.(char(P)).(char(Conditions(p,:))).Suit.(char(fields(m,:))).(char(fields1(n,:)))(1:end-(delay-1),:);

        end
    end
    TD_QTM=TD_QTM-delay;

elseif delay<0   
        for m=1:length(fields)
        for n=1:length(fields1)
    Data.(char(P)).(char(Conditions(p,:))).Suit.(char(fields(m,:))).(char(fields1(n,:)))=Data.(char(P)).(char(Conditions(p,:))).Suit.(char(fields(m,:))).(char(fields1(n,:)))(-delay:end,:);
    Data.(char(P)).(char(Conditions(p,:))).QTM.(char(fields(m,:))).(char(fields1(n,:)))=Data.(char(P)).(char(Conditions(p,:))).QTM.(char(fields(m,:))).(char(fields1(n,:)))(1:end+(delay+1),:);
        end
        end
end
% length(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi)
% length(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi)

 % subplot(2,1,2);
 %    plot(Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi); hold on;
 %    plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi,'r'); hold on;
 %  for r=1:length(TD_QTM)
 %    xline(TD_QTM(r))
 %    end

    [~,MS_suit] = findpeaks(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi,"MinPeakHeight",35); % finds midswing peaks
    if Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi(MS_QTM(1))>80
    MS_suit(1)=[]; % remove synch peak
    end
    MS_suit(end)=[];

    for q= 1:length(MS_suit)
        TD_Suit(q,1) = MS_suit(q) + find(sign(diff(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi(MS_suit(q):end,1)))==1,1,"first"');
    end
    % length(TD_Suit)
    % 
    % subplot(2,1,2);
    % plot(Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.tran); hold on;
    % for r=1:length(TD_Suit)
    % xline(TD_Suit(r))
    % end
    
    
% create matrix for data export
% Labels={'lkneex','lkneey','lkneez','lhipx','lhipy','lhipz','lanklex','lankley','lanklez',...
%     'rkneex','rkneey','rkneez','rhipx','rhipy','rhipz','ranklex','rankley','ranklez'};
Angles_Suit=[Data.(char(P)).(char(Conditions(p,:))).Suit.LeftKnee.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.LeftKnee.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.LeftKnee.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).Suit.LeftHip.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.LeftHip.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.LeftHip.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).Suit.LeftAnkle.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.LeftAnkle.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.LeftAnkle.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.RightKnee.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).Suit.RightHip.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.RightHip.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.RightHip.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).Suit.RightAnkle.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.RightAnkle.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).Suit.RightAnkle.tran(TD_QTM(1):end,1)];

Events_Suit = [TD_Suit(1:end-1)-(TD_Suit(1)-1),TD_Suit(2:end)-(TD_Suit(1)-1)];

Angles_QTM=[Data.(char(P)).(char(Conditions(p,:))).QTM.LeftKnee.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.LeftKnee.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.LeftKnee.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).QTM.LeftHip.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.LeftHip.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.LeftHip.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).QTM.LeftAnkle.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.LeftAnkle.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.LeftAnkle.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.RightKnee.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).QTM.RightHip.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.RightHip.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.RightHip.tran(TD_QTM(1):end,1),...
    Data.(char(P)).(char(Conditions(p,:))).QTM.RightAnkle.sagi(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.RightAnkle.fron(TD_QTM(1):end,1),Data.(char(P)).(char(Conditions(p,:))).QTM.RightAnkle.tran(TD_QTM(1):end,1)];

Events_QTM = [TD_QTM(1:end-1)-(TD_QTM(1)-1),TD_QTM(2:end)-(TD_QTM(1)-1)];

% save events and angles as mat files
filename=['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Angles_QTM\',P,'_',char(Conditions(p,:)),'.mat'];
save(filename,"Angles_QTM");
filename=['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Angles_Suit\',P,'_',char(Conditions(p,:)),'.mat'];
save(filename,"Angles_Suit");
% filename=['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Events_QTM\',P,'_',char(Conditions(p,:)),'.mat'];
% save(filename,"Events_QTM");
% filename=['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Events_Suit\',P,'_',char(Conditions(p,:)),'.mat'];
% save(filename,"Events_Suit");

clear TD_QTM TD_Suit Angles_Suit Angles_QTM Events_Suit Events_QTM
end
pause
end
   