clear all
close all

D=1;  % diameter m
U=1;  % flow velocity m/s
ruo=1;  % water density kg/m3
g=1;    % gravity kg/m2

data = load('C:\Users\jwu\SINTEF\KSP - MAPLES - Documents\02 Prosjektgjennomføring\Collaboration\Collaboration_Brown_MIT_WestLake\NekRS_RigidCylinder\CFD_data_Re60K\comb_Ur4\Hydro.dog');

t=data(:,1);   % time sec
xd=data(:,2);  % x displaement m
yd=data(:,3);  % y displaement m
xv=data(:,4);  % x velocity m/s
yv=data(:,5);  % y velocity m/s
xa=data(:,6);  % x acceleration m/s2
ya=data(:,7);  % y acceleration m/s2
Xf=data(:,8);  % x-force /drag (N)
Yf=data(:,9);  % y-force /lift (N)

dt=t(2)-t(1);
fs=1/dt;

figure
plot(t, yd)
grid on

figure
plot(t,yv,t,ya,t,Yf)
grid on

[pxx, f] = pwelch(yv, [], [], [], fs);

figure
plot(f, pxx, 'LineWidth',1.5)
xlabel('Frequency (Hz)')
ylabel('Power Spectral Density')
title('PSD (Linear Scale)')
grid on
xlim([0 1])