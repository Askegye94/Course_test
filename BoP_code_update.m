%% BOP Project Mirabelle and Boris main function
clear all
close all

%% do you want to see some plots? Switch to 0 if not
test_figure=0;

%% this line makes sure that only participants with QTM files are processed
SUBJ = dir('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\OMC\Data\');

subjects = {SUBJ.name};
subjects(contains({SUBJ.name},'S')==0)=[];
subjects_2 = strrep(subjects,'S','P');

    %% Load data
% Change the filename here to the name of the file you would like to import
% Folder Names
index=1:length(subjects);
% index=setdiff(index, [5,9])
for r=1:length(index)%(9:end)
    Conditions = {'REF'; ['OG_',char(subjects_2(r)),'_S35'];['OG_',char(subjects_2(r)),'_S45'];['OG_',char(subjects_2(r)),'_S55'];...
        ['T_',char(subjects_2(r)),'_S35'];['T_',char(subjects_2(r)),'_S45'];['T_',char(subjects_2(r)),'_S55']};

OpenCap_path = ['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Processed_data\VIDEO\',char(subjects(r)), '\'];    
% Suit_path = ['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\Treadmill\Suit\Processed Files Suit\',char(subjects(r)), '\'];
QTM_path = ['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\OMC\Data\',char(subjects(r)), '\'];

OpenCap_files = dir( [OpenCap_path '*.mot']);
% Suit_files = dir( [Suit_path '*.mat']);
QTM_files = dir([QTM_path '*.mat']);


%% file name detection
    
ref_Q = {}; OG35_Q= {};OG45_Q={};OG55_Q={};T35_Q= {};T45_Q={};T55_Q={};
ref_O = {}; OG35_O= {};OG45_O={};OG55_O={};T35_O= {};T45_O={};T55_O={};

for i=1:length(QTM_files)
        if contains(QTM_files(i).name, Conditions(1,:))
            ref_Q=[ref_Q; QTM_files(i).name];
        elseif contains(QTM_files(i).name, Conditions(2,:))
            OG35_Q = [OG35_Q; QTM_files(i).name];
        elseif contains(QTM_files(i).name, Conditions(3,:))
            OG45_Q =[OG45_Q; QTM_files(i).name];
        elseif contains(QTM_files(i).name, Conditions(4,:))
            OG55_Q =[OG55_Q; QTM_files(i).name]; 
        elseif contains(QTM_files(i).name, Conditions(5,:))
            T35_Q =[T35_Q; QTM_files(i).name]; 
        elseif contains(QTM_files(i).name, Conditions(6,:))
            T45_Q =[T45_Q; QTM_files(i).name];
        elseif contains(QTM_files(i).name, Conditions(7,:))
            T55_Q =[T55_Q; QTM_files(i).name];
        end
    end

    for i=1:length(OpenCap_files)
        if contains(OpenCap_files(i).name, Conditions(1,:))
            ref_O=[ref_O; OpenCap_files(i).name];
        elseif contains(OpenCap_files(i).name, Conditions(2,:))
            OG35_O = [OG35_O ;OpenCap_files(i).name];
        elseif contains(OpenCap_files(i).name, Conditions(3,:))
            OG45_O =[OG45_O; OpenCap_files(i).name];
        elseif contains(OpenCap_files(i).name, Conditions(4,:))
            OG55_O =[OG55_O; OpenCap_files(i).name];
        elseif contains(OpenCap_files(i).name, Conditions(5,:))
            T35_O =[T35_O; OpenCap_files(i).name]; 
        elseif contains(OpenCap_files(i).name, Conditions(6,:))
            T45_O =[T45_O, OpenCap_files(i).name];
        elseif contains(OpenCap_files(i).name, Conditions(7,:))
            T55_O =[T55_O, OpenCap_files(i).name];
        end
    end

    Condition.ref=[ref_Q,ref_O];
    Condition.OG35=[OG35_Q,OG35_O];
    Condition.OG45=[OG45_Q,OG45_O];
    Condition.OG55=[OG55_Q,OG55_O];
    Condition.T35=[T35_Q,T35_O];
    Condition.T45=[T45_Q,T45_O];
    Condition.T55=[T55_Q,T55_O];

    clear ref_Q OG35_Q OG45_Q OG55_Q T35_Q T45_Q T55_Q ref_O OG35_O OG45_O OG55_O T35_O T45_O T55_O


%% Calculate Joint angles for QTM files, first filters markers with 
% a 2nd order low-pass butterworth filter with cutoff frequence of 6 Hz,
% then calculates the joint angles according to the ISB recommendations
Cond = fieldnames(Condition);
for i= 2:length(fieldnames(Condition))
    % there are multiple files for the same condition
    for p = 1:length(Condition.(char(Cond(i))))
    % Filename= eval(Conditions{i,:});
    % Filename1=Filename{1};
[QTM_Angles,freq] = JointAnglesCalc(fullfile(QTM_path, Condition.(char(Cond(i)))(p,1)),fullfile(QTM_path,Condition.(char(Cond(1)))(p,1)));


%% fine until here, next step: load the OpenCap files and extract the Joint angles.

%% Load joint angles from Suit


video=readtable(string(fullfile(OpenCap_path,Condition.(char(Cond(i)))(p,2))), 'FileType','text');
OpenCap_Angles.RightKnee =video.knee_angle_r;
OpenCap_Angles.LeftKnee =video.knee_angle_l;
OpenCap_Angles.RightHip =table2array(video(:,8:10));
OpenCap_Angles.LeftHip =table2array(video(:,15:17));
OpenCap_Angles.RightAnkle =table2array(video(:,12:13));
OpenCap_Angles.LeftAnkle =table2array(video(:,19:20));

frameRate= 60;


% Suit_Angles.RightKnee = [tree.jointData(16).jointAngle(:,3),tree.jointData(16).jointAngle(:,1),tree.jointData(16).jointAngle(:,2)];
% Suit_Angles.RightHip = [tree.jointData(15).jointAngle(:,3),-tree.jointData(15).jointAngle(:,1),-tree.jointData(15).jointAngle(:,2)];
% Suit_Angles.RightAnkle = [tree.jointData(17).jointAngle(:,3),-tree.jointData(17).jointAngle(:,1),tree.jointData(17).jointAngle(:,2)];
% Suit_Angles.LeftKnee = [tree.jointData(20).jointAngle(:,3),tree.jointData(20).jointAngle(:,1),tree.jointData(20).jointAngle(:,2)];
% Suit_Angles.LeftHip = [tree.jointData(19).jointAngle(:,3),-tree.jointData(19).jointAngle(:,1),-tree.jointData(19).jointAngle(:,2)];
% Suit_Angles.LeftAnkle = [tree.jointData(21).jointAngle(:,3),-tree.jointData(21).jointAngle(:,1),tree.jointData(21).jointAngle(:,2)];

% frameRate = tree.metaData.subject_frameRate;
% suitLabel = tree.metaData.subject_label;
% originalFilename = tree.metaData.subject_originalFilename;
% recDate = tree.metaData.subject_recDate;
% segmentCount = tree.metaData.subject_segmentCount;

%% find peak knee joint angle for synch
[~, temp]=findpeaks(QTM_Angles.RightKnee(:,1),"MinPeakHeight",50);
start_QTM =temp(1);
clear temp

[~, temp]=findpeaks(OpenCap_Angles.RightKnee(:,1),"MinPeakHeight",50);
start_OpenCap = temp(1);
% 
% [~, temp]=findpeaks(Suit_Angles.RightKnee(:,1),"MinPeakHeight",50);
% start_Suit = temp(1);

if length(QTM_Angles.RightKnee(start_QTM:end,1))<length(OpenCap_Angles.RightKnee(start_OpenCap:end,1))
    end_file = length(QTM_Angles.RightKnee(start_QTM:end,1));
else
    end_file = length(OpenCap_Angles.RightKnee(start_OpenCap:end,1));
end
% 
% if length(QTM_Angles.RightKnee(start_QTM:end,1))<length(Suit_Angles.RightKnee(start_Suit:end,1))
%     end_file = length(QTM_Angles.RightKnee(start_QTM:end,1));
% else
%     end_file = length(Suit_Angles.RightKnee(start_Suit:end,1));
% end


%%% Here we have to adapt as we have different angles for the knee and
%%% ankle joint (1 for opencap and 3 for the knee
angle_names=fieldnames(QTM_Angles);
for v=1:6
    Synch.QTM_Angles.(char(angle_names(v,:)))=QTM_Angles.(char(angle_names(v,:)))(start_QTM:start_QTM+end_file-1,1:3);
    Synch.OpenCap_Angles.(char(angle_names(v,:)))=OpenCap_Angles.(char(angle_names(v,:)))(start_OpenCap:start_Suit+end_file-1,1:3);
    % Synch.Suit_Angles.(char(angle_names(v,:)))=Suit_Angles.(char(angle_names(v,:)))(start_Suit:start_Suit+end_file-1,1:3);
end

% if test_figure==1
figure
plot(Synch.QTM_Angles.RightKnee(:,1), 'b'); hold on;
plot(Synch.Suit_Angles.RightKnee(:,1), 'r');
figure
plot(Synch.QTM_Angles.LeftKnee(:,1), 'b'); hold on;
plot(Synch.Suit_Angles.LeftKnee(:,1), 'r');
% end
%% find all peaks >50 deg for gait cycle detection

% [~,peaks_QTM_r] = findpeaks(Synch.QTM_Angles.RightKnee(:,1),"MinPeakHeight",30);
% [~,peaks_Suit_r] = findpeaks(Synch.Suit_Angles.RightKnee(:,1),"MinPeakHeight",30);
[~,peaks_QTM_l] = findpeaks(Synch.QTM_Angles.LeftKnee(:,1),"MinPeakHeight",30);
% [~,peaks_Suit_l] = findpeaks(Synch.Suit_Angles.LeftKnee(:,1),"MinPeakHeight",30);

%% find min after peak right leg
%% change here to find the first minimum after the peak!

% 
% [~,TD_QTM_r]= findpeaks(-Synch.QTM_Angles.RightKnee(peaks_QTM_r(11):end,1)-mean(-Synch.QTM_Angles.RightKnee(peaks_QTM_r(11):end,1)),"MinPeakHeight",0,"MinPeakDistance",100);
% TD_QTM_r=TD_QTM_r+peaks_QTM_r(11);

% TD_QTM_r=TD_QTM_r(1:2:length(TD_QTM_r));
% TD_QTM_r=TD_QTM_r+peaks_QTM_r(11);
% [~,TD_Suit_r]= findpeaks(-Synch.Suit_Angles.RightKnee(peaks_Suit_r(11):end,1)-mean(-Synch.Suit_Angles.RightKnee(peaks_QTM_r(11):end,1)),"MinPeakHeight",0,"MinPeakDistance",100);
% TD_Suit_r=TD_Suit_r+peaks_Suit_r(11);

% TD_Suit_r=TD_Suit_r(1:2:length(TD_Suit_r));
% TD_Suit_r=TD_Suit_r+peaks_Suit_r(11);



%% find min after peak left leg
% [~,TD_QTM_l]= findpeaks(-Synch.QTM_Angles.LeftKnee(peaks_QTM_l(11):end,1)-mean(-Synch.QTM_Angles.LeftKnee(peaks_QTM_l(11):end,1)),"MinPeakHeight",0,"MinPeakDistance",100);
% TD_QTM_l=TD_QTM_l+peaks_QTM_l(11);

% TD_QTM_l=TD_QTM_l(1:2:length(TD_QTM_l));
% TD_QTM_l=TD_QTM_l+peaks_QTM_l(11);
%% check for outliers
% if length(rmoutliers(diff(TD_QTM_l),'grubbs'))< length(diff(TD_QTM_l))
% [~,rm_QTM_l] =rmoutliers(diff(TD_QTM_l),'grubbs')
% end
% 
% [~,TD_Suit_l]= findpeaks(-Synch.Suit_Angles.LeftKnee(peaks_Suit_l(11):end,1)-mean(-Synch.Suit_Angles.LeftKnee(peaks_QTM_l(11):end,1)),"MinPeakHeight",0,"MinPeakDistance",100);
% TD_Suit_l=TD_Suit_l+peaks_Suit_l(11);

% TD_Suit_l=TD_Suit_l(1:2:length(TD_Suit_l));
% TD_Suit_l=TD_Suit_l+peaks_Suit_l(11);
% if length(rmoutliers(diff(TD_Suit_l),'grubbs'))< length(diff(TD_Suit_l))
% [~,rm_Suit_l] =rmoutliers(diff(TD_Suit_l),'grubbs')
% TD_Suit_l(rm_Suit_l==1)=[]
% end
%% only use 50 cycles peaks 
% left_names= angle_names(contains(angle_names, 'Left'));
% right_names= angle_names(contains(angle_names, 'Right'));
% 
% for v=1:3
%     for p=1:3
%     for t=1:min([length(TD_QTM_r),length(TD_QTM_l),length(TD_Suit_l),length(TD_Suit_r)])-1
% Norm.QTM_Angles.(char(right_names(v,:)))(1:101,t,p)=normalise(Synch.QTM_Angles.(char(right_names(v,:)))(TD_QTM_r(t):TD_QTM_r(t+1),p),1);
% Norm.Suit_Angles.(char(right_names(v,:)))(1:101,t,p)=normalise(Synch.Suit_Angles.(char(right_names(v,:)))(TD_Suit_r(t):TD_Suit_r(t+1),p),1);
% 
% Norm.QTM_Angles.(char(left_names(v,:)))(1:101,t,p)=normalise(Synch.QTM_Angles.(char(left_names(v,:)))(TD_QTM_l(t):TD_QTM_l(t+1),p),1);
% Norm.Suit_Angles.(char(left_names(v,:)))(1:101,t,p)=normalise(Synch.Suit_Angles.(char(left_names(v,:)))(TD_Suit_l(t):TD_Suit_l(t+1),p),1);
%     end
%     end
% end
% 
% if test_figure==1
% figure;
% plot(Norm.QTM_Angles.RightKnee(:,:,1), 'b'); hold on;
% plot(Norm.Suit_Angles.RightKnee(:,:,1), 'r');
% 
% figure;
% plot(Norm.QTM_Angles.LeftKnee(:,:,1), 'b'); hold on;
% plot(Norm.Suit_Angles.LeftKnee(:,:,1), 'r');
% end

%% change format of Norm

plane=['sagi';'fron';'tran'];
for v=1:6
    for p=1:3
   Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.(char(angle_names(v,:))).(char(plane(p,:)))=Synch.QTM_Angles.(char(angle_names(v,:)))(:,p);
   Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.(char(angle_names(v,:))).(char(plane(p,:)))=Synch.Suit_Angles.(char(angle_names(v,:)))(:,p);

    end
end


figure;
sgtitle('Right');
subplot(3,3,1)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightKnee.sagi,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightKnee.sagi,'r');
legend([x1(1),x2(1)],'QTM', 'Suit');
title('Knee sag');
subplot(3,3,2)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightKnee.fron,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightKnee.fron,'r');
title('Knee front');
subplot(3,3,3)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightKnee.tran,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightKnee.tran,'r');
title('Knee trans');
subplot(3,3,4)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightHip.sagi,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightHip.sagi,'r');
title('Hip sag');
subplot(3,3,5)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightHip.fron,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightHip.fron,'r');
title('Hip front');
subplot(3,3,6)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightHip.tran,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightHip.tran,'r');
title('Hip trans');
subplot(3,3,7)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightAnkle.sagi,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightAnkle.sagi,'r');
title('Ankle sag');
subplot(3,3,8)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightAnkle.fron,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightAnkle.fron,'r');
title('Ankle front');
subplot(3,3,9)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.RightAnkle.tran,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.RightAnkle.tran,'r');
title('Ankle trans');

figure;
sgtitle('Left');
subplot(3,3,1)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftKnee.sagi,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftKnee.sagi,'r');
legend([x1(1),x2(1)],'QTM', 'Suit');
title('Knee sag');
subplot(3,3,2)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftKnee.fron,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftKnee.fron,'r');
title('Knee front');
subplot(3,3,3)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftKnee.tran,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftKnee.tran,'r');
title('Knee trans');
subplot(3,3,4)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftHip.sagi,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftHip.sagi,'r');
title('Hip sag');
subplot(3,3,5)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftHip.fron,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftHip.fron,'r');
title('Hip front');
subplot(3,3,6)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftHip.tran,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftHip.tran,'r');
title('Hip trans');
subplot(3,3,7)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftAnkle.sagi,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftAnkle.sagi,'r');
title('Ankle sag');
subplot(3,3,8)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftAnkle.fron,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftAnkle.fron,'r');
title('Ankle front');
subplot(3,3,9)
x1=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).QTM.LeftAnkle.tran,'b'); hold on
x2=plot(Data.(char(subjects(:,r))).(char(Conditions(i,:))).Suit.LeftAnkle.tran,'r');
title('Ankle trans');
    
    
    end % end loop over all trials of a condition
end %% end loop to loop over all conditions
subjects(r)
save (['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_treadmill_study\',char(subjects(r)),'_Data.mat'], "Data")
pause
end %% end for loop over all subjects

% save ("Data.mat", "Data")
% 
% 
% sub = 'S05'
% 
% plot(Data.(char(sub(1,:))).run9.Suit.RightKnee.fron, 'r'); hold on
% plot(Data.(char(sub(1,:))).run9.QTM.RightKnee.fron, 'b');
% 
% Data.(char(sub(1,:))).run9.Suit.RightKnee.fron=-Data.(char(sub(1,:))).run9.Suit.RightKnee.fron
