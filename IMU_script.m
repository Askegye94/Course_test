
clear all; clc
%% IMU script, try getting rotation matrixes from raw IMU data


%% load data:
% IMU.Left_foot = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\Left_foot_D422CD0084FA_20250219_135915.csv');
% IMU.Left_shank = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\left_shank_D422CD00851E_20250219_135915.csv');
% IMU.Left_thigh = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\left_thigh_D422CD0084F1_20250219_135915.csv');
% IMU.pelvis = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\pelvis_D422CD0084FC_20250219_135915.csv');
% IMU.Right_foot = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\right_foot_D422CD00855C_20250219_135915.csv');
% IMU.Right_shank = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\right_shank_D422CD00854D_20250219_135915.csv');
% IMU.Right_thigh = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\right_thigh_D422CD0084E0_20250219_135915.csv');


IMU.Left_foot = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\Left_foot_D422CD0084FA_20250219_134736.csv');
IMU.Left_shank = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\left_shank_D422CD00851E_20250219_134736.csv');
IMU.Left_thigh = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\left_thigh_D422CD0084F1_20250219_134736.csv');
IMU.pelvis = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\pelvis_D422CD0084FC_20250219_134736.csv');
IMU.Right_foot = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\right_foot_D422CD00855C_20250219_134736.csv');
IMU.Right_shank = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\right_shank_D422CD00854D_20250219_134736.csv');
IMU.Right_thigh = importfile2('C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\right_thigh_D422CD0084E0_20250219_134736.csv');

sensors=fieldnames(IMU);
% fuse = imufilter('SampleRate',60, 'OrientationFormat','Rotation matrix');
% fuse2 = imufilter('SampleRate',60);
% fuse3 = ahrsfilter("SampleRate",60);
% 
% B = 1/10*ones(10,1);
% out = filter(B,1,input);
% [pxx,x]=pwelch(ACC.pelvis(:,1),[],[],[],60);
% figure;
% plot(x,pxx)
% 
% filterfunc()

% acc_filt=filterfunc(ACC.pelvis(2:end,:),60,'low',2,1)
for i=1:length(sensors)
% ACC.(char(sensors(i)))=filterfunc(table2array(IMU.(char(sensors(i)))(2:200,14:16)),60,'low',2,1);
% GYR.(char(sensors(i)))=filterfunc(table2array(IMU.(char(sensors(i)))(2:200,17:19)),60,'low',2,1);
%% ref
% ACC.(char(sensors(i)))=(table2array(IMU.(char(sensors(i)))(10:200,14:16)));
% GYR.(char(sensors(i)))=(table2array(IMU.(char(sensors(i)))(10:200,17:19)))-mean(table2array(IMU.(char(sensors(i)))(10:200,17:19)),1);
% MAG.(char(sensors(i)))=table2array(IMU.(char(sensors(i)))(10:200,20:22));

%% dyn
ACC.(char(sensors(i)))=filterfunc(table2array(IMU.(char(sensors(i)))(10:end,14:16)),60,'low',2,1);
GYR.(char(sensors(i)))=(table2array(IMU.(char(sensors(i)))(10:end,17:19)))-mean(table2array(IMU.(char(sensors(i)))(10:100,17:19)),1);
MAG.(char(sensors(i)))=filterfunc(table2array(IMU.(char(sensors(i)))(10:end,20:22)),60,'low',2,1);

quat_old.(char(sensors(i)))=table2array(IMU.(char(sensors(i)))(10:end,3:6));
% rotMat.(char(sensors(i)))= fuse(ACC.(char(sensors(i))),GYR.(char(sensors(i))));
% eul.(char(sensors(i)))=rotm2eul(rotMat.(char(sensors(i))),"YXZ");
% quat.(char(sensors(i)))= fuse3(ACC.(char(sensors(i))),GYR.(char(sensors(i))),MAG.(char(sensors(i))));
end


%% EKF 

% Number of IMUs
numIMUs = 7;

% State Vector for each IMU [q0, q1, q2, q3, bgx, bgy, bgz]
X = zeros(7, numIMUs);

P = repmat(eye(7), 1, 1, numIMUs); % Covariance matrices
% P = repmat(eye(7) * 1e-3, 1, 1, numIMUs);
% P(1:4, 1:4, :) = P(1:4, 1:4, :) * 1e-2; % Start with a higher uncertainty for orientation


if any(isnan(P(:))) || any(isinf(P(:)))
    error('P is NaN or Inf at initialization!');
end
% Process noise
% Q = diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]);
Q = diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]);

% Measurement noise (accelerometer & magnetometer)
% R = diag([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]);
R = diag([5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2]);

dt = 1/60; % Assume 60Hz IMU data

MIN=min([length(ACC.Left_foot),length(ACC.Left_shank),length(ACC.Left_thigh),length(ACC.pelvis),length(ACC.Right_foot),length(ACC.Right_shank),length(ACC.Right_thigh)]);

% **Improved Initial Quaternion Estimation**
for imu = 1:numIMUs
    accel_avg = mean(ACC.(char(sensors(imu)))(1:5, :), 1);  % Average first 5 readings
    mag_avg = mean(MAG.(char(sensors(imu)))(1:5, :), 1);

    X(1:4, imu) = estimateInitialQuaternion(accel_avg, mag_avg); % Compute initial quaternion
    X(5:7, imu) = [0; 0; 0];  % Initial bias as zero
end

% **Main Loop**
for t = 1:MIN
    for imu = 1:numIMUs
        % Extract sensor readings
        gyro = GYR.(char(sensors(imu)));  % [gx, gy, gz]
        accel = ACC.(char(sensors(imu))); % [ax, ay, az]
        mag = MAG.(char(sensors(imu)));   % [mx, my, mz]

        % **Prediction Step using Gyroscope**
        F = eye(7); % Jacobian of the state transition
        X(:, imu) = predictEKF(X(:, imu), gyro(t,:)', dt, t);
        if any(isnan(X))
            error('NaN detected in predictEKF output for IMU %d at time step %d', imu, t);
        end
        P(:, :, imu) = F * P(:, :, imu) * F' + Q;

        % **Measurement Update using Accelerometer & Magnetometer**
        [X(:, t,imu), P(:, :,t, imu)] = updateEKF(X(:, imu), P(:, :, imu), accel(t,:)', mag(t,:)', R);
        % disp(norm(X(1:4, imu))); % Quaternion should remain normalized

    end
end

for r=1:7
QUAT.(char(sensors(r)))=X(1:4,:,r)';
end
figure
plot(QUAT.pelvis,'DisplayName','QUAT.pelvis')
%% Plots

figure;
poseplot(quaternion(QUAT.pelvis(10,:)));
hold on
poseplot(quaternion(QUAT.Left_thigh(10,:)));
poseplot(quaternion(QUAT.Right_thigh(10,:)));

poseplot(quaternion(QUAT.Left_shank(10,:)));
poseplot(quaternion(QUAT.Right_shank(10,:)));

poseplot(quaternion(QUAT.Left_foot(10,:)));
poseplot(quaternion(QUAT.Right_foot(10,:)));


nLines = [size(QUAT.pelvis(10:end,:),1), size(QUAT.Left_thigh(10:end,:),1), size(QUAT.Left_shank(10:end,:),1),size(QUAT.Left_foot(10:end,:),1), ...
    size(QUAT.Right_thigh(10:end,:),1), size(QUAT.Right_shank(10:end,:),1), size(QUAT.Right_foot(10:end,:),1)];

time = IMU.pelvis.SampleTimeFine * 10^-6;
time = time - time(1);
time = string(time);

outputFilename = 'C:\Users\sdd380\surfdrive - David, S. (Sina)@surfdrive.surf.nl\Projects\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Data_2025\Unprocessed_data\IMU\S13\Dyn.sto';

header = "DataRate=60.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.4-2022-07-23-0e9fedc\nendheader";
fid=fopen(outputFilename,'w');
fprintf(fid, header + '\n');
line = "time\tpelvis_imu\tfemur_l_imu\ttibia_l_imu\tcalcn_l_imu\tfemur_r_imu\ttibia_r_imu\tcalcn_r_imu";

fprintf(fid, line + '\n');
i = 1;
while i <= size(QUAT.pelvis(10:end,:),1) && i<= size(QUAT.Left_thigh(10:end,:),1) && i<= size(QUAT.Left_shank(10:end,:),1) && i<= size(QUAT.Left_foot(10:end,:),1) && i<= size(QUAT.Right_thigh(10:end,:),1) && i<= size(QUAT.Right_shank(10:end,:),1) && i<= size(QUAT.Right_foot(10:end,:),1)

    t = time(i);
    
   
    pe = string(sprintf('%.4f, %.4f, %.4f, %.4f',QUAT.pelvis(i+9,:)));
    l_th = string(sprintf('%.4f, %.4f, %.4f, %.4f',QUAT.Left_thigh(i+9,:)));
    l_s = string(sprintf('%.4f, %.4f, %.4f, %.4f',QUAT.Left_shank(i+9,:)));
    l_f = string(sprintf('%.4f, %.4f, %.4f, %.4f',QUAT.Left_foot(i+9,:)));
    r_th = string(sprintf('%.4f, %.4f, %.4f, %.4f',QUAT.Right_thigh(i+9,:)));
    r_s = string(sprintf('%.4f, %.4f, %.4f, %.4f',QUAT.Right_shank(i+9,:)));
    r_f = string(sprintf('%.4f, %.4f, %.4f, %.4f',QUAT.Right_foot(i+9,:)));
   

    line = t + "\t" + pe + "\t" + r_th + "\t" + r_s + "\t" + r_f + "\t" + l_th + "\t" + l_s + "\t" + l_f + "\t";

    fprintf(fid, line + "\n");
    i = i + 1;
end

fclose(fid);