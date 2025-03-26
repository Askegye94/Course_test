%%
% Computer Methods in Human Motion Analysis 2017 -- HW3
% Matlab Version: MATLAB R2017a
% Student: ¾÷±ñºÓ¤@ ¤ý«Âµ¾ R05522625

% addpath(genpath(fileparts(cd))) % adding all hw directory to PATH.

%% Initialization
clc;
clearvars;
close all;


%% Practice 1
% tic
% disp('Practice 1')
seq = {'xyx', 'xyz', 'xzy', 'xzx', 'yxy', 'yxz', 'yzy', 'yzx', 'zxy', 'zxz', 'zyx', 'zyz'};
for i=1:length(seq)
    eval(['R',seq{i},'=RotFormula(seq{i});'])
end
clear i seq %clear unwanted variables
save EulerR.mat %save .mat file
% disp('Done Saving.')
% toc
% disp(' ')
%% Practice 2
% tic
% disp('Practice 2')
% load('DataQ1.mat')

%%
% figure;
% plot3(LASI(1,1),LASI(1,2),LASI(1,3), 'ro'); hold on;
% plot3(RASI(1,1),RASI(1,2),RASI(1,3), 'ro');
% plot3(LPSI(1,1),LPSI(1,2),LPSI(1,3), 'ro');
% plot3(RPSI(1,1),RPSI(1,2),RPSI(1,3), 'ro');
% plot3(LHEE(1,1),LHEE(1,2),LHEE(1,3), 'ro');
% plot3(RHEE(1,1),RHEE(1,2),RHEE(1,3), 'ro');
% plot3(LTHI(1,1),LTHI(1,2),LTHI(1,3), 'ro');
% plot3(RTHI(1,1),RTHI(1,2),RTHI(1,3), 'ro');
% plot3(LTOE(1,1),LTOE(1,2),LTOE(1,3), 'ro');
% plot3(RTOE(1,1),RTOE(1,2),RTOE(1,3), 'ro');
% plot3(LTRO(1,1),LTRO(1,2),LTRO(1,3), 'ko');
% plot3(RTRO(1,1),RTRO(1,2),RTRO(1,3), 'ko');
% 
% 
% plot3(LBTO(1,1),LBTO(1,2),LBTO(1,3), 'go');
% plot3(RBTO(1,1),RBTO(1,2),RBTO(1,3), 'go');
% 
% plot3(LFOO(1,1),LFOO(1,2),LFOO(1,3), 'bo');
% plot3(RFOO(1,1),RFOO(1,2),RFOO(1,3), 'bo');
% 
% plot3(LLFC(1,1),LLFC(1,2),LLFC(1,3), 'bx');
% plot3(RLFC(1,1),RLFC(1,2),RLFC(1,3), 'bx');
% 
% plot3(LLMA(1,1),LLMA(1,2),LLMA(1,3), 'gx');
% plot3(RLMA(1,1),RLMA(1,2),RLMA(1,3), 'gx');
% 
% plot3(LMFC(1,1),LMFC(1,2),LMFC(1,3), 'rx');
% plot3(RMFC(1,1),RMFC(1,2),RMFC(1,3), 'rx');
% 
% plot3(LMMA(1,1),LMMA(1,2),LMMA(1,3), 'mo');
% plot3(RMMA(1,1),RMMA(1,2),RMMA(1,3), 'mo');
% 
% plot3(LSHA(1,1),LSHA(1,2),LSHA(1,3), 'mx');
% plot3(RSHA(1,1),RSHA(1,2),RSHA(1,3), 'mx');
% 
% plot3(LTT(1,1),LTT(1,2),LTT(1,3), 'kx');
% plot3(RTT(1,1),RTT(1,2),RTT(1,3), 'kx');
% axis equal




acq= btkReadAcquisition('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\Treadmill\Processed Data\S19\run90002.c3d')
M=btkGetMarkers(acq)

[lRg2p, lVg2p, lPcoord_local] = CoordPelvis(cat(3, M.L_ASIS, M.L_PSIS, M.R_ASIS), 'l', {'LASI', 'LPSI', 'RASI'});
[rRg2p, rVg2p, rPcoord_local] = CoordPelvis(cat(3, M.L_ASIS, M.R_PSIS, M.R_ASIS), 'r', {'LASI', 'RPSI', 'RASI'});
[lRg2t, lVg2t, lTcoord_local] = CoordThigh(cat(3, M.L_Knee, M.L_GTR, M.L_Knee_Medial), 'l', {'LLFC', 'LTRO', 'LMFC'});
% [lRg2t, lVg2t, lTcoord_local] = CoordThigh(cat(3, LLFC, LTRO, LMFC), 'l', {'LLFC', 'LTRO', 'LMFC'});

[rRg2t, rVg2t, rTcoord_local] = CoordThigh(cat(3, M.R_GTR, M.R_Knee_Medial, M.R_Knee), 'r', {'RTRO', 'RMFC', 'RLFC'});
% [rRg2t, rVg2t, rTcoord_local] = CoordThigh(cat(3, RTRO, RMFC, RLFC), 'r', {'RTRO', 'RMFC', 'RLFC'});

[lRg2s, lVg2s, lScoord_local] = CoordShank(cat(3, M.L_TT, M.L_Ankle, M.L_Ankle_Medial, M.L_HF), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
% [lRg2s, lVg2s, lScoord_local] = CoordShank(cat(3, L_TT, LLMA, LMMA, LSHA), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});

[rRg2s, rVg2s, rScoord_local] = CoordShank(cat(3, M.R_TT, M.R_HF, M.R_Ankle, M.R_Ankle_Medial), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
% [rRg2s, rVg2s, rScoord_local] = CoordShank(cat(3, RTT, RSHA, RLMA, RMMA), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});

[lRg2f, lVg2f, lFcoord_local] = CoordFoot(cat(3, M.L_Heel_Top, M.L_MT2, M.L_LMT1), 'l', {'LHEE', 'LTOE', 'LFOO'});
% [lRg2f, lVg2f, lFcoord_local] = CoordFoot(cat(3, LHEE, LTOE, LFOO), 'l', {'LHEE', 'LTOE', 'LFOO'});
[rRg2f, rVg2f, rFcoord_local] = CoordFoot(cat(3, M.R_MT1, M.R_Heel_Top, M.R_MT2), 'r', {'RFOO', 'RHEE', 'RTOE'});
% [rRg2f, rVg2f, rFcoord_local] = CoordFoot(cat(3, RFOO, RHEE, RTOE), 'r', {'RFOO', 'RHEE', 'RTOE'});

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

% (i)
dif1=abs(RotAngConvert(therpt,'zxy')-rRp2t);
if ~isempty(dif1(dif1>10^-10))
    disp('(i) not equal!');
else
    disp('(i) equal!'); 
end
    
% (ii)
dif2=abs(therpt+fliplr(RotAngConvert(rRt2p,'yxz')));
if ~isempty(dif2(dif2>10^-10)) 
    disp('(ii) not equal!');
else
    disp('(ii) equal!'); 
end

% (iii)
dif3=abs(RotAngConvert(rRg2t,'zxy')-fliplr(RotAngConvFix(rRg2t,'yxz')));
if ~isempty(dif3(dif3>10^-10)) 
    disp('(iii) not equal!');
else
    disp('(iii) equal!'); 
end

toc
disp(' ')
%% Practice 3
tic
disp('Practice 3')

clear acq M
% Load data
% load('subcali.mat');
acq= btkReadAcquisition('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\Treadmill\Processed Data\S19\ref0001.c3d')
M=btkGetMarkers(acq)

% Create Rs


lRg2p = CoordPelvis(cat(3, M.L_ASIS(1,:), M.L_PSIS(1,:), M.R_ASIS(1,:)), 'l', {'LASI', 'LPSI', 'RASI'});
rRg2p = CoordPelvis(cat(3, M.L_ASIS(1,:), M.R_PSIS(1,:), M.R_ASIS(1,:)), 'r', {'LASI', 'RPSI', 'RASI'});
lRg2t = CoordThigh(cat(3, M.L_Knee(1,:), M.L_GTR(1,:), M.L_Knee_Medial(1,:)), 'l', {'LLFC', 'LTRO', 'LMFC'});
rRg2t = CoordThigh(cat(3, M.R_GTR(1,:), M.R_Knee_Medial(1,:), M.R_Knee(1,:)), 'r', {'RTRO', 'RMFC', 'RLFC'});
lRg2s = CoordShank(cat(3, M.L_TT(1,:), M.L_Ankle(1,:), M.L_Ankle_Medial(1,:), M.L_HF(1,:)), 'l', {'LTT', 'LLMA', 'LMMA', 'LSHA'});
rRg2s = CoordShank(cat(3, M.R_TT(1,:), M.R_HF(1,:), M.R_Ankle(1,:), M.R_Ankle_Medial(1,:)), 'r', {'RTT', 'RSHA', 'RLMA', 'RMMA'});
lRg2f = CoordFoot(cat(3, M.L_Heel_Top(1,:), M.L_MT2(1,:), M.L_LMT1(1,:)), 'l', {'LHEE', 'LTOE', 'LFOO'});
rRg2f =CoordFoot(cat(3, M.R_MT1(1,:), M.R_Heel_Top(1,:), M.R_MT2(1,:)), 'r', {'RFOO', 'RHEE', 'RTOE'});
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

% Plot
figname = {'Z', 'X', 'Y'};
artiname = {'lHip','rHip','lKnee','rKnee','lAnkle','rAnkle'};
dataRo = cat(3, thelpt, therpt, thelts, therts, thelsf, thersf);
dataRc = cat(3, lcpt, rcpt, lcts, rcts, lcsf, rcsf);

QTM_Angles.LeftKnee = [-lcts(:,1),-lcts(:,3),-lcts(:,2)];
QTM_Angles.LeftHip = [lcpt(:,1),lcpt(:,3),lcpt(:,2)];
QTM_Angles.LeftAnkle = [lcsf(:,1),lcsf(:,3),lcsf(:,2)];
QTM_Angles.RightKnee = [-rcts(:,1),rcts(:,3),rcts(:,2)];
QTM_Angles.RightHip = [rcpt(:,1),rcpt(:,3),rcpt(:,2)];
QTM_Angles.RightFoot = [rcsf(:,1),rcsf(:,3),rcsf(:,2)];

difRoc = abs(dataRc - dataRo);
frame = 1:n;
for i = 1:3
    figure('Name',char(figname(i)),'NumberTitle','off','position',[400*i-300 50 600 700]);
    hold on
    % suptitle(['Euler Angle : ',char(figname(i)),' -axis']);
    for j = 1:6
        eval(['ax',num2str(j)]) = subplot(3,2,j);
        hold on
        title(char(artiname(j)))
        a=plot(frame, dataRo(:,i,j));
        b=plot(frame, dataRc(:,i,j));
        mi = difRoc(:,i,j)==max(difRoc(:,i,j));
        scatter([frame(mi) frame(mi)],[dataRo(mi, i, j) dataRc(mi, i, j)],'k')
        ha = annotation('arrow');  % store the arrow information in ha
        ha.Parent = gca;           % associate the arrow the the current axes
        ha.Y = [dataRo(mi, i, j) dataRc(mi, i, j)];          % the location in data units
        ha.X = [frame(mi) frame(mi)];  
        ylabel('Angle (deg)')
        xlim([0 n]) 
        ylim auto 
        if j == 2
            Leg = legend([a; b], {'Raw', 'Offseted'});
            Pos = get(Leg, 'Position'); 
            set(Leg, 'Position', [Pos(1)+0.1, Pos(2)+0.07, Pos(3), Pos(4)])
        end
    end   
end

disp('Done Plotting.')
toc
save('rotang.mat', 'lRg2t', 'rRg2t', 'lRg2s', 'rRg2s', 'lRg2f', 'rRg2f')


