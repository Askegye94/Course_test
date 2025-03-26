clear all

SUBJ = dir('C:\Users\sdd380\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\OMC\');

subjects = {SUBJ.name};
subjects(contains({SUBJ.name},'S')==0)=[];
subjects_2 = strrep(subjects,'S','P');
data_all = struct();  % Initialize as an empty structure array
    %% Load data
% Change the filename here to the name of the file you would like to import
% Folder Names
index=1:length(subjects);
for r=10%:length(index)%(9:end)
    Conditions = {'REF'; ['OG_',char(subjects_2(r)),'_S35'];['OG_',char(subjects_2(r)),'_S45'];['OG_',char(subjects_2(r)),'_S55'];...
        ['T_',char(subjects_2(r)),'_S35'];['T_',char(subjects_2(r)),'_S45'];['T_',char(subjects_2(r)),'_S55']};

OpenCap_path = ['C:\Users\sdd380\OneDrive - Vrije Universiteit Amsterdam\PHD Aske\Papers\OMC_IMU_VIDEO\Data_2025\Processed_data\VIDEO\',char(subjects(r)), '\'];    

OpenCap_files = dir( [OpenCap_path '*.mot']);
for z=1:length(OpenCap_files)
    number = extractBetween(OpenCap_files(z).name,'_T','.mot');
    if str2double(number)<10 & ~contains(number, '0')
        OpenCap_files_temp= fullfile([char(extractBefore(OpenCap_files(z).name, '_T')), char('_T0'),char(number),char('.mot')]);
        movefile(fullfile(OpenCap_path,OpenCap_files(z).name),fullfile(OpenCap_path,OpenCap_files_temp))
    end


end

end