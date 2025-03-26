function [filtered_signal] = filterfunc(signal, fs, type, order, cu1, cu2)
%FILTERFUNC filters a signal using a butterworth filter. 
%   [filtered_signal] = FILTERFUNC(signal, fs, type, order, cu1, cu2)
%   filters a signal using a butterworth filter. The function can be used
%   for low-pass, high-pass bandpass and bandstop filters. 
%
%   Inputs: 
%       signal  =   the unfiltered signal
%       fs      =   sample frequency of the unfiltered signal
%       type    =   the type of filter, the following filters are possible:
%           'low'       -   a lowpass filter
%           'high'      -   a highpass filter
%           'bandpass'  -   a bandpass filter
%           'stop'      -   a bandstop filter
%       order   =   the filter order
%       cu1     =   cutoff frequency of the filter
%       cu2     =   a second cutoff frequency (only specified in case of a 
%                   bandstop filter)  
%
%   Outputs:
%       filtered_signal     =   the filtered signal 

% Function made for the course:
% Movement Analysis @ Vrije Universiteit Amsterdam

%% Computations
if nargin > 5
    if ~strcmp(type,'stop') && ~strcmp(type,'bandpass')
        error('Filter type does not correspond to number of input arguments')
    end
    [b,a] = butter(order,[(cu1/(fs/2)) (cu2/(fs/2))], type);  
    filtered_signal = filtfilt(b,a,signal);
else 
    if ~strcmp(type,'low') && ~strcmp(type,'high')
        error('Filter type does not correspond to number of input arguments')
    end
    [b,a] = butter(order,(cu1/(fs/2)), type);  
    filtered_signal = filtfilt(b,a,signal);  
end
end

