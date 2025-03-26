function [Angles, freq] = JointAnglesCalc(File,Ref)



% seq = {'xyx', 'xyz', 'xzy', 'xzx', 'yxy', 'yxz', 'yzy', 'yzx', 'zxy', 'zxz', 'zyx', 'zyz'};
% for i=1:length(seq)
%     eval(['R',seq{i},'=RotFormula(seq{i});'])
% end
% clear i seq %clear unwanted variables
% save EulerR.mat %save .mat file

acq = load(string(File));
name= fieldnames(acq);
M1 = acq.(char(name)).Trajectories.Labeled.Data;
Mnames = acq.(char(name)).Trajectories.Labeled.Labels;
m_names= Mnames;
for p= 1:length(Mnames)
M.(char(m_names(p)))= reshape(M1(p,1:3,:),[3,length(M1)])';
end
% acq= btkReadAcquisition(File);
% M=btkGetMarkers(acq);
%% Fill missing values in each marker field before filtering
marker_fields = fieldnames(M);
for i = 1:length(marker_fields)
    marker = M.(char(marker_fields{i}));
    % Process each coordinate column separately
    for col = 1:size(marker,2)
         marker(:,col) = fill_missing(marker(:,col));
    end
    M.(char(marker_fields{i})) = marker;
end

% for p = 1:length(Mnames)
%     tempData = reshape(M1(p,1:3,:), [3, length(M1)])';
%     tempData(isnan(tempData) | isinf(tempData)) = 0; % Replace NaN and Inf with 0
%     M.(char(m_names(p))) = tempData;
% end

%% Check and print any remaining NaNs or Infs per marker
for i = 1:length(marker_fields)
    marker = M.(char(marker_fields{i}));
    countNaN = sum(isnan(marker), 'all');
    countInf = sum(isinf(marker), 'all');
    if countNaN > 0 || countInf > 0
        fprintf('Marker %s has %d NaNs and %d Infs.\n', marker_fields{i}, countNaN, countInf);
    end
end
%% filter markers

% m_names=fieldnames(M);
for i=1:length(m_names)
M.(char(m_names(i))) = filterfunc(M.(char(m_names(i))),240,'low',2,6);
end
freq = acq.(char(name)).FrameRate;
% freq = btkGetPointFrequency(acq);
% DSHS markerset
% [lRg2p, lVg2p, lPcoord_local] = CoordPelvis(cat(3, M.LASI, M.LPSI, M.RASI), 'l', {'LASI', 'LPSI', 'RASI'});
% [rRg2p, rVg2p, rPcoord_local] = CoordPelvis(cat(3, M.LASI, M.RPSI, M.RASI), 'r', {'LASI', 'RPSI', 'RASI'});
% [lRg2t, lVg2t, lTcoord_local] = CoordThigh(cat(3, M.LKNE, M.LTRO, M.LKNEM), 'l', {'LLFC', 'LTRO', 'LMFC'});
% [rRg2t, rVg2t, rTcoord_local] = CoordThigh(cat(3, M.RTRO, M.RKNEM, M.RKNE), 'r', {'RTRO', 'RMFC', 'RLFC'});
% [lRg2s, lVg2s, lScoord_local] = CoordShank(cat(3, M.LTIB, M.LANK, M.LANKM, M.LKNE), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
% [rRg2s, rVg2s, rScoord_local] = CoordShank(cat(3, M.RTIB, M.RKNE, M.RANK, M.RANKM), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
% [lRg2f, lVg2f, lFcoord_local] = CoordFoot(cat(3, M.LHEE, M.LTOE, M.LLMT1), 'l', {'LHEE', 'LTOE', 'LFOO'});
% [rRg2f, rVg2f, rFcoord_local] = CoordFoot(cat(3, M.RLMT1, M.RHEE, M.RTOE), 'r', {'RFOO', 'RHEE', 'RTOE'});

% VU markerset old
% [lRg2p, lVg2p, lPcoord_local] = CoordPelvis(cat(3, M.L_ASIS, M.L_PSIS, M.R_ASIS), 'l', {'LASI', 'LPSI', 'RASI'});
% [rRg2p, rVg2p, rPcoord_local] = CoordPelvis(cat(3, M.L_ASIS, M.R_PSIS, M.R_ASIS), 'r', {'LASI', 'RPSI', 'RASI'});
% [lRg2t, lVg2t, lTcoord_local] = CoordThigh(cat(3, M.L_Knee, M.L_GTR, M.L_Knee_Medial), 'l', {'LLFC', 'LTRO', 'LMFC'});
% [rRg2t, rVg2t, rTcoord_local] = CoordThigh(cat(3, M.R_GTR, M.R_Knee_Medial, M.R_Knee), 'r', {'RTRO', 'RMFC', 'RLFC'});
% [lRg2s, lVg2s, lScoord_local] = CoordShank(cat(3, M.L_TT, M.L_Ankle, M.L_Ankle_Medial, M.L_HF), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
% [rRg2s, rVg2s, rScoord_local] = CoordShank(cat(3, M.R_TT, M.R_HF, M.R_Ankle, M.R_Ankle_Medial), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
% [lRg2f, lVg2f, lFcoord_local] = CoordFoot(cat(3, M.L_Heel_Top, M.L_MT2, M.L_LMT1), 'l', {'LHEE', 'LTOE', 'LFOO'});
% [rRg2f, rVg2f, rFcoord_local] = CoordFoot(cat(3, M.R_MT1, M.R_Heel_Top, M.R_MT2), 'r', {'RFOO', 'RHEE', 'RTOE'});

% VU markerset old
[lRg2p, lVg2p, lPcoord_local] = CoordPelvis(cat(3, M.LASI, M.LPSI, M.RASI), 'l', {'LASI', 'LPSI', 'RASI'});
[rRg2p, rVg2p, rPcoord_local] = CoordPelvis(cat(3, M.LASI, M.RPSI, M.RASI), 'r', {'LASI', 'RPSI', 'RASI'});
[lRg2t, lVg2t, lTcoord_local] = CoordThigh(cat(3, M.LKNE, M.LTRO, M.LKNEM), 'l', {'LLFC', 'LTRO', 'LMFC'});
[rRg2t, rVg2t, rTcoord_local] = CoordThigh(cat(3, M.RTRO, M.RKNEM, M.RKNE), 'r', {'RTRO', 'RMFC', 'RLFC'});
[lRg2s, lVg2s, lScoord_local] = CoordShank(cat(3, M.LTT, M.LANK, M.LANKM, M.LHF), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
[rRg2s, rVg2s, rScoord_local] = CoordShank(cat(3, M.RTT, M.RHF, M.RANK, M.RANKM), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
[lRg2f, lVg2f, lFcoord_local] = CoordFoot(cat(3, M.LHEE, M.LTOE, M.LMT1), 'l', {'LHEE', 'LTOE', 'LFOO'});
[rRg2f, rVg2f, rFcoord_local] = CoordFoot(cat(3, M.RMT1, M.RHEE, M.RTOE), 'r', {'RFOO', 'RHEE', 'RTOE'});

n=size(lRg2p,3);

lRp2t = pagemtimes(lRg2p, 'ctranspose', lRg2t, 'none'); lRt2p =pagemtimes(lRg2t, 'ctranspose', lRg2p, 'none');
rRp2t = pagemtimes(rRg2p, 'ctranspose', rRg2t, 'none'); rRt2p = pagemtimes(rRg2t, 'ctranspose', rRg2p, 'none');
lRt2s = pagemtimes(lRg2t, 'ctranspose', lRg2s, 'none'); lRs2t = pagemtimes(lRg2s, 'ctranspose', lRg2t, 'none');
rRt2s = pagemtimes(rRg2t, 'ctranspose', rRg2s, 'none'); rRs2t = pagemtimes(rRg2s, 'ctranspose', rRg2t, 'none');
lRs2f = pagemtimes(lRg2s, 'ctranspose', lRg2f, 'none'); lRf2s = pagemtimes(lRg2f, 'ctranspose', lRg2s, 'none');
rRs2f = pagemtimes(rRg2s, 'ctranspose', rRg2f, 'none'); rRf2s = pagemtimes(rRg2f, 'ctranspose', rRg2s, 'none');

therpt = RotAngConvert(rRp2t,'zxy');
thelpt = RotAngConvert(lRp2t,'zxy');
thelts = RotAngConvert(lRt2s,'zxy');
therts = RotAngConvert(rRt2s,'zxy');
thelsf = RotAngConvert(lRs2f,'zxy');
thersf = RotAngConvert(rRs2f,'zxy');
% 
% % (i)
% dif1=abs(RotAngConvert(therpt,'zxy')-rRp2t);
% if ~isempty(dif1(dif1>10^-10))
%     disp('(i) not equal!');
% else
%     disp('(i) equal!'); 
% end
% 
% % (ii)
% dif2=abs(therpt+fliplr(RotAngConvert(rRt2p,'yxz')));
% if ~isempty(dif2(dif2>10^-10)) 
%     disp('(ii) not equal!');
% else
%     disp('(ii) equal!'); 
% end
% 
% % (iii)
% dif3=abs(RotAngConvert(rRg2t,'zxy')-fliplr(RotAngConvFix(rRg2t,'yxz')));
% if ~isempty(dif3(dif3>10^-10)) 
%     disp('(iii) not equal!');
% else
%     disp('(iii) equal!'); 
% end
% 

%% calibrate with ref file
clear acq M

acq = load(string(File));
name= fieldnames(acq);
M1 = acq.(char(name)).Trajectories.Labeled.Data;
Mnames = acq.(char(name)).Trajectories.Labeled.Labels;
m_names= Mnames;
for p= 1:length(Mnames)
M.(char(m_names(p)))= reshape(M1(p,1:3,:),[3,length(M1)])';
end

%% Fill missing values in each marker field before filtering
marker_fields = fieldnames(M);
for i = 1:length(marker_fields)
    marker = M.(char(marker_fields{i}));
    % Process each coordinate column separately
    for col = 1:size(marker,2)
         marker(:,col) = fill_missing(marker(:,col));
    end
    M.(char(marker_fields{i})) = marker;
end
% 
% for p = 1:length(Mnames)
%     tempData = reshape(M1(p,1:3,:), [3, length(M1)])';
%     tempData(isnan(tempData) | isinf(tempData)) = 0; % Replace NaN and Inf with 0
%     M.(char(m_names(p))) = tempData;
% end

%% filter markers
% m_names=fieldnames(M);
for i=1:length(m_names)
M.(char(m_names(i))) = filterfunc(M.(char(m_names(i))),240,'low',2,6);
end
% Create Rs

% DSHS markerset
% lRg2p = CoordPelvis(cat(3, M.LASI(1,:), M.LPSI(1,:), M.RASI(1,:)), 'l', {'LASI', 'LPSI', 'RASI'});
% rRg2p = CoordPelvis(cat(3, M.LASI(1,:), M.RPSI(1,:), M.RASI(1,:)), 'r', {'LASI', 'RPSI', 'RASI'});
% lRg2t = CoordThigh(cat(3, M.LKNE(1,:), M.LTRO(1,:), M.LKNEM(1,:)), 'l', {'LLFC', 'LTRO', 'LMFC'});
% rRg2t = CoordThigh(cat(3, M.RTRO(1,:), M.RKNEM(1,:), M.RKNE(1,:)), 'r', {'RTRO', 'RMFC', 'RLFC'});
% lRg2s = CoordShank(cat(3, M.LTIB(1,:), M.LANK(1,:), M.LANKM(1,:), M.LKNE(1,:)), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
% rRg2s = CoordShank(cat(3, M.RTIB(1,:), M.RKNE(1,:), M.RANK(1,:), M.RANKM(1,:)), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
% lRg2f = CoordFoot(cat(3, M.LHEE(1,:), M.LTOE(1,:), M.LLMT1(1,:)), 'l', {'LHEE', 'LTOE', 'LFOO'});
% rRg2f =CoordFoot(cat(3, M.RLMT1(1,:), M.RHEE(1,:), M.RTOE(1,:)), 'r', {'RFOO', 'RHEE', 'RTOE'});

% VU markerset old
% lRg2p = CoordPelvis(cat(3, M.L_ASIS(1,:), M.L_PSIS(1,:), M.R_ASIS(1,:)), 'l', {'LASI', 'LPSI', 'RASI'});
% rRg2p = CoordPelvis(cat(3, M.L_ASIS(1,:), M.R_PSIS(1,:), M.R_ASIS(1,:)), 'r', {'LASI', 'RPSI', 'RASI'});
% lRg2t = CoordThigh(cat(3, M.L_Knee(1,:), M.L_GTR(1,:), M.L_Knee_Medial(1,:)), 'l', {'LLFC', 'LTRO', 'LMFC'});
% rRg2t = CoordThigh(cat(3, M.R_GTR(1,:), M.R_Knee_Medial(1,:), M.R_Knee(1,:)), 'r', {'RTRO', 'RMFC', 'RLFC'});
% lRg2s = CoordShank(cat(3, M.L_TT(1,:), M.L_Ankle(1,:), M.L_Ankle_Medial(1,:), M.L_HF(1,:)), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
% rRg2s = CoordShank(cat(3, M.R_TT(1,:), M.R_HF(1,:), M.R_Ankle(1,:), M.R_Ankle_Medial(1,:)), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
% lRg2f = CoordFoot(cat(3, M.L_Heel_Top(1,:), M.L_MT2(1,:), M.L_LMT1(1,:)), 'l', {'LHEE', 'LTOE', 'LFOO'});
% rRg2f =CoordFoot(cat(3, M.R_MT1(1,:), M.R_Heel_Top(1,:), M.R_MT2(1,:)), 'r', {'RFOO', 'RHEE', 'RTOE'});

% VU markerset
lRg2p = CoordPelvis(cat(3, M.LASI(1,:), M.LPSI(1,:), M.RASI(1,:)), 'l', {'LASI', 'LPSI', 'RASI'});
rRg2p = CoordPelvis(cat(3, M.LASI(1,:), M.RPSI(1,:), M.RASI(1,:)), 'r', {'LASI', 'RPSI', 'RASI'});
lRg2t = CoordThigh(cat(3, M.LKNE(1,:), M.LTRO(1,:), M.LKNEM(1,:)), 'l', {'LLFC', 'LTRO', 'LMFC'});
rRg2t = CoordThigh(cat(3, M.RTRO(1,:), M.RKNEM(1,:), M.RKNE(1,:)), 'r', {'RTRO', 'RMFC', 'RLFC'});
lRg2s = CoordShank(cat(3, M.LTT(1,:), M.LANK(1,:), M.LANKM(1,:), M.LHF(1,:)), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
rRg2s = CoordShank(cat(3, M.RTT(1,:), M.RHF(1,:), M.RANK(1,:), M.RANKM(1,:)), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
lRg2f = CoordFoot(cat(3, M.LHEE(1,:), M.LTOE(1,:), M.LMT1(1,:)), 'l', {'LHEE', 'LTOE', 'LFOO'});
rRg2f =CoordFoot(cat(3, M.RMT1(1,:), M.RHEE(1,:), M.RTOE(1,:)), 'r', {'RFOO', 'RHEE', 'RTOE'});
lRspt = lRg2p\lRg2t; rRspt = rRg2p\rRg2t;
lRsts = lRg2t\lRg2s; rRsts = rRg2t\rRg2s;
lRssf = lRg2s\lRg2f; rRssf = rRg2s\rRg2f;

% Create Rc
lRcpt = JointAngOffset( lRspt, lRp2t ); 
rRcpt = JointAngOffset( rRspt, rRp2t );
lRcts = JointAngOffset( lRsts, lRt2s ); rRcts = JointAngOffset( rRsts, rRt2s );
lRcsf = JointAngOffset( lRssf, lRs2f ); rRcsf = JointAngOffset( rRssf, rRs2f );

% Euler angle of Rc
rcpt = RotAngConvert(rRcpt, 'zxy'); lcpt = RotAngConvert(lRcpt, 'zxy'); 
rcts = RotAngConvert(rRcts, 'zxy'); lcts = RotAngConvert(lRcts, 'zxy'); 
rcsf = RotAngConvert(rRcsf, 'zxy'); lcsf = RotAngConvert(lRcsf, 'zxy'); 


Angles.RightKnee = [-rcts(:,1),-rcts(:,2),rcts(:,3)];
Angles.RightHip = [rcpt(:,1),rcpt(:,2),-rcpt(:,3)];
Angles.RightAnkle = [rcsf(:,1),rcsf(:,2),rcsf(:,3)];
Angles.LeftKnee = [-lcts(:,1),lcts(:,2),-lcts(:,3)];
Angles.LeftHip = [lcpt(:,1),-lcpt(:,2),lcpt(:,3)];
Angles.LeftAnkle = [lcsf(:,1),-lcsf(:,2),-lcsf(:,3)];
