%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function jkrichma_ocular_dominance
%
% Jeff Krichmar - UC Irvine
% 
% Demonstrate how BCM learning rule adjusts to different levels of input.
% 
% This script is based on the experiments from Rittenhouse et al., 
% "Monocular deprivation induces homosynaptic long-term depression 
% in visual cortex, Nature, 397:347-350, 1999. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function jkrichma_ocular_dominance

% Valid wavelength 400 < wavelength < 699
train_rgby = wavelength_rgb(580)
test_rgby = wavelength_rgb(580)

[th_ctl,od_ctl, m_ctl_1]=ocular_dominance(train_rgby(1), train_rgby(2), 'rg'); % run model with both eyes stimulated
od_ctl = od_ctl / max(abs(od_ctl)); % normalize the ocular dominance metric between -1 and 1

[th_dep,od_dep, m_ctl_2]=ocular_dominance(0, 0, 'rg'); % run model with only right eye stimulated
od_dep = od_dep / max(abs(od_dep));

[th_ctl,od_ctl, m_ctl_3]=ocular_dominance(train_rgby(3), train_rgby(4), 'by'); % run model with both eyes stimulated
od_ctl = od_ctl / max(abs(od_ctl)); % normalize the ocular dominance metric between -1 and 1

[th_dep,od_dep, m_ctl_4]=ocular_dominance(0, 0, 'by'); % run model with only right eye stimulated
od_dep = od_dep / max(abs(od_dep));
subplot(2,1,1)
title('Color Opponency Spectrum Adapt')
plot((1:299),m_ctl_1,"Color",'b','LineStyle','--')
hold on
title('Color Opponency Spectrum nAdapt')
plot((1:299),m_ctl_2,"Color",'r','LineStyle','--')
hold off
subplot(2,1,2)
title('Color Opponency Spectrum Adapt')
plot((1:299),m_ctl_3,"Color",'b','LineStyle','--')
hold on
title('Color Opponency Spectrum nAdapt')
plot((1:299),m_ctl_4,"Color",'r','LineStyle','--')

figure
scatter(-m_ctl_1, m_ctl_3, 'blue', 'filled')
hold on
scatter(-m_ctl_2, m_ctl_4, 'red','filled')
figure


title('Color Opponency Spectrum')
plot((1:299),m_ctl_1,"Color",'b','LineStyle','--')
hold on
plot((1:299),m_ctl_2,"Color",'g','LineStyle','--','LineWidth',3)
plot((1:299),m_ctl_3,"Color",'r','LineWidth',3)
plot((1:299),m_ctl_4,"Color",'y','LineWidth',3)
box off
% % plot the distribution of ocular dominance values
% subplot(2,1,1);histogram(od_ctl);title('Control');axis([-1 1 0 35])
% xlabel('Left Dominant --------------------------------------------------------------- Right Dominant')
% subplot(2,1,2);histogram(od_dep);title('Deprived');axis([-1 1 0 35])
% xlabel('Left Dominant --------------------------------------------------------------- Right Dominant')
% [h,p]=kstest2(th_ctl,th_dep);

% % plot the BCM learning function with the median thresholds
% figure
% subplot(2,1,1)
% inx = 0;
% x = 0.01:0.01:1;
% dw_ctl = zeros(1,size(x,2));
% dw_dep = zeros(1,size(x,2));
% no_lrn = zeros(1,size(x,2));
% for i=1:size(x,2)
%     inx = inx + 1;
%     dw_ctl(inx) = bcm(1, x(i), median(th_ctl));
%     dw_dep(inx) = bcm(1, x(i), median(th_dep));
% end
% plot(x, dw_ctl, 'LineWidth',2)
% hold on
% plot(x, dw_dep, 'LineWidth',2)
% plot(x,no_lrn,'.-')
% title('BCM Learning Curve', 'FontSize', 16)
% legend('Control', 'Deprived', 'FontSize', 14, 'Location','northwest')
% xlabel('Postsynaptic Activity', 'FontSize', 14)
% ylabel('dW', 'FontSize', 14)
% 
% % plot the population of BCM thresholds with boxplots
% subplot(2,1,2)
% h=boxplot([th_ctl;th_dep]','Notch','off','Labels',{'Control','Deprived'},'Whisker',1, 'widths', 0.75);
% set(h,{'linew'},{2})
% title(['BCM Thresholds: p < ', num2str(p)], 'FontSize', 16)

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function jkrichma_ocular_dominance
%
% Input: mon_depriv - 0 if both eyes stimulated; 1 if only right eye
% stimulated.
% Outputs: - BCM thresholds of V1 neurons (v1_thr) and the ocular dominance 
% metric (OD) of the eye responses.
function[v1_thr, OD, mean_cl] = ocular_dominance(leye_stim, reye_stim, color_opp)

if color_opp == 'rg'
    l_stim = 1
    r_stim = 2
else
    l_stim = 3
    r_stim = 4
end
TRAINING_TRIALS = 50;
num_eye = 20;
num_v1 = 100;
eye = zeros(1,num_eye);
v1 = zeros(1,num_v1);

% random uniform weight distribution between eye and V1
w = 0.1*rand(num_eye,num_v1);

gain = 0.5; % gain of sigmoid function
eye_stim = 1.0;

% Spontaneous activity.
% The output of the sigmoid function with this value and a gain of 0.5 equals 0.377
spont_act = -1.0; 
v1_thr = 0.5*ones(1,num_v1);

% Train the network with spontaneous activity and the eye stimulation
for t = 1:TRAINING_TRIALS

    i_eye = spont_act*ones(1,num_eye)+(-0.25+0.5*rand(1,num_eye)); % generate some randomness with spontaneous activity


    i_eye(1:num_eye/2) = i_eye(1:num_eye/2)+leye_stim;
    i_eye(num_eye/2+1:num_eye) = i_eye(num_eye/2+1:num_eye)+reye_stim;

    % get the eye activity based on spontaneous activity and the sigmoid
    % function
    for n = 1:num_eye
        eye(n) = sigmoid(i_eye(n),gain);
    end

    % get the V1 activity based on the eye activity and the weights between
    % the eye and V1
    i_v1 = zeros(1,num_v1);
    for n = 1:num_v1
        i_v1(n) = sum(eye .* w(:,n)');
        v1(n) = sigmoid(i_v1(n),gain);
    end

    % updated the weights using the BCM rule
    for i=1:num_eye
        for j = 1:num_v1
            w(i,j) = w(i,j) + bcm(eye(i),v1(j),v1_thr(j));
        end
    end

    % update the thresholds using the BCM rule
    for i=1:num_v1
        v1_thr(i) = v1_thr(i) + bcm_thr(v1(i),v1_thr(i));
    end
end

% Stimulate each eye. Calculate the ocular dominance as described in
% Rittenhouse Nature 1999.
ler = zeros(1,num_v1);
rer = zeros(1,num_v1);
OD = zeros(1,num_v1);
for wv = 1:299
    rgby = wavelength_rgb(wv+400);
    % Stimulate left eye. Record the response.
    i_eye = spont_act*ones(1,num_eye);
    i_eye(1:num_eye/2) = i_eye(1:num_eye/2)+rgby(l_stim);
    for n = 1:num_eye
        eye(n) = sigmoid(i_eye(n),gain);
    end
    i_v1 = zeros(1,num_v1);
    for n = 1:num_v1
        i_v1(n) = sum(eye .* w(:,n)');
        ler(n) = sigmoid(i_v1(n),gain);
    end
    
    % Stimulate right eye. Record the response.
    i_eye = spont_act*ones(1,num_eye);
    i_eye(num_eye/2+1:num_eye) = i_eye(num_eye/2+1:num_eye)+rgby(r_stim);
    for n = 1:num_eye
        eye(n) = sigmoid(i_eye(n),gain);
    end
    i_v1 = zeros(1,num_v1);
    for n = 1:num_v1
        i_v1(n) = sum(eye .* w(:,n)');
        rer(n) = sigmoid(i_v1(n),gain);
    end
    
    % calculate the ocular dominance metric
    for n = 1:num_v1
        OD(n) = ((rer(n)-spont_act)-(ler(n)-spont_act)) / ((rer(n)-spont_act)+(ler(n)-spont_act));
    end
    mean_cl(wv) = mean(OD);
end
end


function y = sigmoid (x, g)
y = 1/(1+exp(-x*g));
end

function [dw] = bcm(pre, post, post_thr)
lr = 0.1;
dw = lr*pre*post*(post-post_thr);
end

function [dw_thr] = bcm_thr(post, post_thr)
lr = 0.01;
dw_thr = lr*(post^2 - post_thr);
end

function rgby_out = wavelength_rgb(target_length)
wavelength_400 = readmatrix('rgby_norm.csv');
rgby_out = wavelength_400(target_length - 399, :);
rgby_out = rgby_out/max(rgby_out);
end
