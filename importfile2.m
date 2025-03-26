function data = importfile2(filename, dataLines)
%IMPORTFILE2 Import data from a text file
%  RIGHTTHIGHD422CD0084E020250219135915 = IMPORTFILE2(FILENAME) reads
%  data from text file FILENAME for the default selection.  Returns the
%  data as a table.
%
%  RIGHTTHIGHD422CD0084E020250219135915 = IMPORTFILE2(FILE, DATALINES)
%  reads data for the specified row interval(s) of text file FILENAME.
%  Specify DATALINES as a positive scalar integer or a N-by-2 array of
%  positive scalar integers for dis-contiguous row intervals.
%
%  Example:
%  rightthighD422CD0084E020250219135915 = importfile2("C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\right_thigh_D422CD0084E0_20250219_135915.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 14-Mar-2025 18:28:47

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 24);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["PacketCounter", "SampleTimeFine", "Quat_W", "Quat_X", "Quat_Y", "Quat_Z", "dq_W", "dq_X", "dq_Y", "dq_Z", "dv1", "dv2", "dv3", "Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z", "Status", "VarName24"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "VarName24", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "VarName24", "EmptyFieldRule", "auto");

% Import the data
data = readtable(filename, opts);

end