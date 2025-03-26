%% BOP Project Mirabelle and Boris main function
clear all
close all

%% do you want to see some plots? Switch to 0 if not
test_figure=0;

%% this line makes sure that only participants with QTM files are processed
% SUBJ = dir('C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Unprocessed_data\OMC\Data\');
SUBJ = dir('C:\Users\sdd380\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\OMC\');

subjects = {SUBJ.name};
subjects(contains({SUBJ.name},'S')==0)=[];
subjects_2 = strrep(subjects,'S','P');
data_all = struct();  % Initialize as an empty structure array
    %% Load data
% Change the filename here to the name of the file you would like to import
% Folder Names
index=1:length(subjects);
% index=setdiff(index, [5,9])
for r=26:length(index)%(9:end)
    Conditions = {'REF'; ['OG_',char(subjects_2(r)),'_S35'];['OG_',char(subjects_2(r)),'_S45'];['OG_',char(subjects_2(r)),'_S55'];...
        ['T_',char(subjects_2(r)),'_S35'];['T_',char(subjects_2(r)),'_S45'];['T_',char(subjects_2(r)),'_S55']};

% OpenCap_path = ['C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\VIDEO\',char(subjects(r)), '\'];    
OpenCap_path = ['C:\Users\sdd380\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\VIDEO\',char(subjects(r)), '\'];    

% % Suit_path = ['C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\Treadmill\Suit\Processed Files Suit\',char(subjects(r)), '\'];
% QTM_path = ['C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\OMC\',char(subjects(r)), '\'];
QTM_path = ['C:\Users\sdd380\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\OMC\',char(subjects(r)), '\'];

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


%% Calculate Joint angles for QTM files

Cond = fieldnames(Condition);

% Initialize global counter
global_counter = 0;

for i = 2:length(Cond)
    num_files = size(Condition.(Cond{i}), 1);
    num_ref_files = size(Condition.(Cond{1}), 1);
    
    for p = 1:num_files
        QTM_Angles = [];
        % Print the global counter and the file we are processing
        fprintf('Processing file %d: %s\n', global_counter, Condition.(Cond{i}){p,1});
        
        % Choose a reference file: if there is a corresponding reference file, use that;
        % otherwise, default to the first reference file.
        if p <= num_ref_files
            refFile = Condition.(Cond{1}){p,1};
        else
            refFile = Condition.(Cond{1}){1,1};
        end
        
        % Call JointAnglesCalc with the QTM file for the current condition and the chosen reference file.
        [QTM_Angles, freq] = JointAnglesCalc(... 
            fullfile(QTM_path, Condition.(Cond{i}){p,1}), ... 
            fullfile(QTM_path, refFile));

        % Determine the index to start considering data (last 70%)
        start_idx = round(0.5 * size(QTM_Angles.LeftKnee, 1)) + 1;
        
        % Adjust knee angles if they are negative in the last 70% of the data
        if min(QTM_Angles.LeftKnee(start_idx:end,1)) < 0
            QTM_Angles.LeftKnee(:,1) = QTM_Angles.LeftKnee(:,1) + abs(min(QTM_Angles.LeftKnee(start_idx:end,1)));
        else
           QTM_Angles.LeftKnee(:,1) = QTM_Angles.LeftKnee(:,1) - abs(min(QTM_Angles.LeftKnee(start_idx:end,1)));

        end

        if min(QTM_Angles.RightKnee(start_idx:end,1)) < 0
            QTM_Angles.RightKnee(:,1) = QTM_Angles.RightKnee(:,1) + abs(min(QTM_Angles.RightKnee(start_idx:end,1)));
        else
            QTM_Angles.RightKnee(:,1) = QTM_Angles.RightKnee(:,1) - abs(min(QTM_Angles.RightKnee(start_idx:end,1)));
        end

        % Increment the global counter after each file is processed
        global_counter = global_counter + 1;
    

%% Load joint angles from OpenCAP

video=readtable(string(fullfile(OpenCap_path,Condition.(char(Cond(i)))(p,2))), 'FileType','text');

OpenCap_Angles.RightKnee =video.knee_angle_r;
OpenCap_Angles.LeftKnee =video.knee_angle_l;

start_idx_OC = round(0.5 * size(OpenCap_Angles.LeftKnee, 1)) + 1;

        if min(OpenCap_Angles.RightKnee(start_idx_OC:end,1)) < 0
            OpenCap_Angles.RightKnee(:,1) = OpenCap_Angles.RightKnee(:,1) + abs(min(OpenCap_Angles.RightKnee(start_idx_OC:end,1)));
        else
            OpenCap_Angles.RightKnee(:,1) = OpenCap_Angles.RightKnee(:,1) - abs(min(OpenCap_Angles.RightKnee(start_idx_OC:end,1)));
        end
        
        if min(OpenCap_Angles.LeftKnee(start_idx_OC:end,1)) < 0
            OpenCap_Angles.LeftKnee(:,1) = OpenCap_Angles.LeftKnee(:,1) + abs(min(OpenCap_Angles.LeftKnee(start_idx_OC:end,1)));
        else
            OpenCap_Angles.LeftKnee(:,1) = OpenCap_Angles.LeftKnee(:,1) - abs(min(OpenCap_Angles.LeftKnee(start_idx_OC:end,1)));
        end


OpenCap_Angles.RightHip =table2array(video(:,8:10));
OpenCap_Angles.LeftHip =table2array(video(:,15:17));
OpenCap_Angles.RightAnkle =table2array(video(:,12:13));
OpenCap_Angles.LeftAnkle =table2array(video(:,19:20));

fields = fieldnames(OpenCap_Angles);
for f = 1:length(fields)
    fieldData = OpenCap_Angles.(fields{f});
    if isnumeric(fieldData)
        % Process each column in the field
        for col = 1:size(fieldData,2)
            fieldData(:,col) = fill_missing(fieldData(:,col));
        end
        OpenCap_Angles.(fields{f}) = fieldData;
    end
end


% Assume frameRate_QTM and frameRate_OpenCap are the respective sampling rates
frameRate_QTM = freq;  % This is the output from JointAnglesCalc
frameRate_OpenCap = 60; % Defined in your code

% Compute resampling ratio
[p1, q1] = rat(frameRate_QTM / frameRate_OpenCap);  % p/q gives the ratio for resampling

% Resample OpenCap angles to match QTM
OpenCap_Angles.RightKnee = resample(OpenCap_Angles.RightKnee, p1, q1);
OpenCap_Angles.LeftKnee = resample(OpenCap_Angles.LeftKnee, p1, q1);
OpenCap_Angles.RightHip = resample(OpenCap_Angles.RightHip, p1, q1);
OpenCap_Angles.LeftHip = resample(OpenCap_Angles.LeftHip, p1, q1);
OpenCap_Angles.RightAnkle = resample(OpenCap_Angles.RightAnkle, p1, q1);
OpenCap_Angles.LeftAnkle = resample(OpenCap_Angles.LeftAnkle, p1, q1);

%% find peak knee joint angle for synch
% Create temporary copies of the data
temp_QTM = QTM_Angles.RightKnee(:,1) - min(QTM_Angles.RightKnee(:,1));
temp_OpenCap = OpenCap_Angles.RightKnee(:,1) - min(OpenCap_Angles.RightKnee(:,1));

% Find peaks in the adjusted temporary data
[~, temp] = findpeaks(temp_QTM, "MinPeakHeight", 50);
start_QTM = temp(1);

[~, temp] = findpeaks(temp_OpenCap, "MinPeakHeight", 50);
start_OpenCap = temp(1);

% Determine the shortest length from the starting points
end_file = min(length(QTM_Angles.RightKnee(start_QTM:end,1)), ...
               length(OpenCap_Angles.RightKnee(start_OpenCap:end,1)));

angle_names = fieldnames(QTM_Angles);
for v = 1:6
    % Extract the current angle name as a string
    angle_name = char(angle_names(v,:));

    % Synchronizing QTM_Angles
    Synch.QTM_Angles.(angle_name) = QTM_Angles.(angle_name)(start_QTM:start_QTM+end_file-1, 1:3);

    % Synchronizing OpenCap_Angles, ensuring indexing does not exceed available data
    if isfield(OpenCap_Angles, angle_name) && ~isempty(OpenCap_Angles.(angle_name))
        % Get the actual size of the second dimension
        numCols = min(3, size(OpenCap_Angles.(angle_name), 2));

        % Assign only the available number of columns
        Synch.OpenCap_Angles.(angle_name) = OpenCap_Angles.(angle_name)(start_OpenCap:start_OpenCap+end_file-1, 1:numCols);
    else
        warning(['Field ', angle_name, ' is missing or empty in OpenCap_Angles']);
        Synch.OpenCap_Angles.(angle_name) = []; % Assign an empty array if the field does not exist
    end
end


% if test_figure==1
% figure
% plot(Synch.QTM_Angles.RightKnee(:,1), 'b'); hold on;
% plot(Synch.OpenCap_Angles.RightKnee(:,1), 'r');
% figure
% plot(Synch.QTM_Angles.LeftKnee(:,1), 'b'); hold on;
% plot(Synch.OpenCap_Angles.LeftKnee(:,1), 'r');
% end

%%

% Step 1: Find midswing peaks (peaks over 35)
[~, MS_QTM] = findpeaks(Synch.QTM_Angles.RightKnee(:,1), "MinPeakHeight", 25);

% Step 2: Remove synchronization peak (last peak over 80)
MS_QTM(Synch.QTM_Angles.RightKnee(MS_QTM) > 70)=[];

% Step 3: Remove peaks if they are in the last two data points
last_idx = length(Synch.QTM_Angles.RightKnee);
MS_QTM(MS_QTM >= last_idx - 1) = [];

% Step 4: Find the first valley (local min) before each peak
TD_QTM = zeros(length(MS_QTM), 1); % Pre-allocate

for q = 1:length(MS_QTM)
    % Search for the valley **before** the peak
    search_range = MS_QTM(q):length(Synch.QTM_Angles.RightKnee); % Only look before the peak
    [~, valley_idx] = findpeaks(-Synch.QTM_Angles.RightKnee(search_range,1)); % Find valleys

    if ~isempty(valley_idx)
        TD_QTM(q) = valley_idx(1)+MS_QTM(q); % Select the last valley before the peak
    else
        TD_QTM(q) = NaN; % In case no valley is found
    end

    if isnan(TD_QTM(end))
        TD_QTM(end)=[];
    end
end

%%
figure;
plot(Synch.QTM_Angles.RightKnee(:,1), 'b', 'LineWidth', 1.5); % Plot Right Knee Angle in blue
hold on;
plot(TD_QTM, Synch.QTM_Angles.RightKnee(TD_QTM,1), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Plot TD_QTM markers in red
xlabel('Frame Number');
ylabel('Knee Angle (degrees)');
title('Right Knee Angle with TD_QTM Events');
legend('Right Knee Angle', 'TD_QTM Events');
grid on;
hold off;

% pause
%% this part applies cross correlation to detect time shifts in the data and then matches the 2 data
% sets according to the best correlation shift
[v,lags]=xcorr(Synch.QTM_Angles.RightKnee(:,1),Synch.OpenCap_Angles.RightKnee(:,1));

[~,delay_temp]=max(v);
delay=lags(delay_temp);
delay_output(i,p)=delay;
fields=fieldnames(Synch.QTM_Angles);
if delay>0
    for m=1:length(fields)
    Synch.QTM_Angles.(char(fields(m,:)))=Synch.QTM_Angles.(char(fields(m,:)))(delay:end,:);
    Synch.OpenCap_Angles.(char(fields(m,:)))=Synch.OpenCap_Angles.(char(fields(m,:)))(1:end-(delay-1),:);

    end
    TD_QTM=TD_QTM-delay;

elseif delay<0   
        for m=1:length(fields)
    Synch.QTM_Angles.(char(fields(m,:)))=Synch.QTM_Angles.(char(fields(m,:)))(-delay:end,:);
    Synch.OpenCap_Angles.(char(fields(m,:)))=Synch.OpenCap_Angles.(char(fields(m,:)))(1:end+(delay+1),:);
        end
        
end

% Adapt to 3 planes with new data 
Angles_OpenCap_Angles=[Synch.OpenCap_Angles.LeftKnee(TD_QTM(1):end,1),...
    Synch.OpenCap_Angles.LeftHip(TD_QTM(1):end,1:3),...
    Synch.OpenCap_Angles.LeftAnkle(TD_QTM(1):end,1:2),...
    Synch.OpenCap_Angles.RightKnee(TD_QTM(1):end,1),...
    Synch.OpenCap_Angles.RightHip(TD_QTM(1):end,1:3),...
    Synch.OpenCap_Angles.RightAnkle(TD_QTM(1):end,1:2)];

% Events_OpenCap = [TD_OpenCap(1:end-1)-(TD_OpenCap(1)-1),TD_OpenCap(2:end)-(TD_OpenCap(1)-1)];

Angles_QTM_Angles=[Synch.QTM_Angles.LeftKnee(TD_QTM(1):end,1),...
    Synch.QTM_Angles.LeftHip(TD_QTM(1):end,1:3),...
    Synch.QTM_Angles.LeftAnkle(TD_QTM(1):end,1:2),...
    Synch.QTM_Angles.RightKnee(TD_QTM(1):end,1),...
    Synch.QTM_Angles.RightHip(TD_QTM(1):end,1:3),...
    Synch.QTM_Angles.RightAnkle(TD_QTM(1):end,1:2)];


Events_QTM = [TD_QTM(1:end-1)-(TD_QTM(1)-1),TD_QTM(2:end)-(TD_QTM(1)-1)];


% 
% for t = 1:size(Events_QTM, 1)  % Loop through each event row in Events_QTM
%     % Ensure the indices in Events_QTM are valid
%     start_idx = Events_QTM(t, 1);
%     end_idx = Events_QTM(t, 2);
% 
%     % Check that start and end indices are within the bounds of the data
%     if start_idx > 0 && end_idx <= size(Angles_QTM_Angles, 1)
%         % Safeguard in case there are fewer rows of data than expected
%         num_rows = min(101, end_idx - start_idx + 1);  % Ensure we're not going out of bounds
% 
%         % Store normalized angles data
%         Data.QTM_Angles(1:num_rows, :, t) = normalise2(Angles_QTM_Angles(start_idx:end_idx, :), 1);
%         Data.OpenCap_Angles(1:num_rows, :, t) = normalise2(Angles_OpenCap_Angles(start_idx:end_idx, :), 1);
% 
%         Data.Filename = QTM_files(global_counter).name;  % Assign the correct filename
%         data_all(global_counter).QTM_Angles= Data.QTM_Angles;
%         data_all(global_counter).OpenCap_Angles = Data.OpenCap_Angles;
%         data_all(global_counter).Filename = Data.Filename;    
%     end
% end

for t = 1:size(Events_QTM, 1)  % Loop through each event row in Events_QTM
    % Ensure the indices in Events_QTM are valid
    start_idx = Events_QTM(t, 1);
    end_idx = Events_QTM(t, 2);

    % Check that start and end indices are within the bounds of the data
    if start_idx > 0 && end_idx <= size(Angles_QTM_Angles, 1)
        % Calculate number of rows safely
        actual_rows = end_idx - start_idx + 1;  
        num_rows = min(101, actual_rows);  % Ensure we do not exceed 101

        % Normalize and extract only the required number of rows
        norm_QTM = normalise2(Angles_QTM_Angles(start_idx:end_idx, :), 1);
        norm_OpenCap = normalise2(Angles_OpenCap_Angles(start_idx:end_idx, :), 1);

        % Ensure correct dimensions for assignment
        Data.QTM_Angles(1:num_rows, :, t) = norm_QTM(1:num_rows, :);
        Data.OpenCap_Angles(1:num_rows, :, t) = norm_OpenCap(1:num_rows, :);

        % Store filename and save data
        Data.Filename = QTM_files(global_counter).name;
        data_all(global_counter).QTM_Angles = Data.QTM_Angles;
        data_all(global_counter).OpenCap_Angles = Data.OpenCap_Angles;
        data_all(global_counter).Filename = Data.Filename;    
    end
end


    end % end loop over all trials of a condition
end %% end loop to loop over all conditions
subjects(r)
% save (['C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\Data_for_students\',char(subjects(r)),'_Data.mat'], "data_all")
save (['C:\Users\sdd380\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\Data_for_students\',char(subjects(r)),'_Data.mat'], "data_all")

pause
end %% end for loop over all subjects
