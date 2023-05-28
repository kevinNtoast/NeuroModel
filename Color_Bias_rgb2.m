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

function Color_Bias_rgb2

% Valid wavelength 351 < wavelength < 699
train_rgb = wavelength_rgb(680)
test_rgb = wavelength_rgb(380)
[th_ctl,od_ctl]=Color_under_habituation(train_rgb, test_rgb); % run model stimulating three cones with relative strengths

% plot the distribution of ocular dominance values
subplot(2,2,1);histogram(od_ctl);title('Control');axis([-1 1 0 35])
% xlabel('Green --------------------------------------------------------Red')
% subplot(2,2,2);histogram(od_red);title('Red');axis([-1 1 0 35])
% xlabel('Green --------------------------------------------------------Red')
% subplot(2,2,3);histogram(od_grn);title('Green');axis([-1 1 0 35])
% xlabel('? --------------------------------------------------------?')
% subplot(2,2,4);histogram(od_blu);title('Blue');axis([-1 1 0 35])
% xlabel('Blue? --------------------------------------------------------Green?')
% [h,p]=kstest2(th_ctl,th_red);

rgby = [od_ctl(1, :)-od_ctl(2, :); (od_ctl(1, :) + od_ctl(2, :))/2 - od_ctl(3,:)]
subplot(2,2,2);
scatter(rgby(:,1), rgby(:,2))
hold on
subplot(2,2,3);
hist3([rgby(:,1), rgby(:,2)])
% plot3(od_ctl(:,1), od_ctl(:,2), od_ctl(:,3))

% plot the BCM learning function with the median thresholds
% figure
% subplot(2,1,1)
% inx = 0;
% x = 0.01:0.01:1;
% dw_ctl = zeros(1,size(x,2));
% dw_red = zeros(1,size(x,2));
% no_lrn = zeros(1,size(x,2));
% for i=1:size(x,2)
%     inx = inx + 1;
%     dw_ctl(inx) = bcm(1, x(i), median(th_ctl));
%     dw_red(inx) = bcm(1, x(i), median(th_red));
end
% plot(x, dw_ctl, 'LineWidth',2)
% hold on
% plot(x, dw_red, 'LineWidth',2)
% plot(x,no_lrn,'.-')
% title('BCM Learning Curve', 'FontSize', 16)
% legend('Control', 'Deprived', 'FontSize', 14, 'Location','northwest')
% xlabel('Postsynaptic Activity', 'FontSize', 14)
% ylabel('dW', 'FontSize', 14)

% plot the population of BCM thresholds with boxplots
% subplot(2,1,2)
% h=boxplot([th_ctl;th_red]','Notch','off','Labels',{'Control','Deprived'},'Whisker',1, 'widths', 0.75);
% set(h,{'linew'},{2})
% title(['BCM Thresholds: p < ', num2str(p)], 'FontSize', 16)

% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Opponent(rgb4train, rgb4test)
rg4train = [rgb4train(1) rgb4train(2)]
rg4test = [rgb4test(1) rgb4test(2)]
by4train = [rgb4train(3) (rgb4train(1) + rgb4train(1))/2]
by4test = [rgb4test(3) (rgb4test(1) + rgb4test(1))/2]

Color_under_habituation(rg4train, rg4test)
Color_under_habituation(by4train, by4test)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function for color training
%
% Input: 
% col_bias -
% 0 = no bias, all stimulated
% 1 = red bias, only red wavelength exposure
% 2 = green bias, only green wavelength exposure
% 3 = blue bias, only blue wavelength exposure
% Outputs: - BCM thresholds of V1 neurons (v1_thr) and the ocular dominance 
% metric (OD) of the eye responses.
function[v1_thr, Col] = Color_under_habituation(rgb4train, rgb4test)

TRAINING_TRIALS = 50;
num_con = 30;
num_rgb = 100;
cone = zeros(1,num_con);
v1 = zeros(1,num_rgb);

% random uniform weight distribution between eye and V1
w = 0.1*rand(num_con,num_rgb);

gain = 0.5; % gain of sigmoid function
eye_stim = 1.0; %stim level 을 색에 관해서 바꿔야함.

% Spontaneous activity.
% The output of the sigmoid function with this value and a gain of 0.5 equals 0.377
spont_act = -1.0; 
v1_thr = 0.5*ones(1,num_rgb);

% Train the network with spontaneous activity and the eye stimulation
for t = 1:TRAINING_TRIALS

    i_cone = spont_act*ones(1,num_con)+(-0.25+0.5*rand(1,num_con)); % generate some randomness with spontaneous activity


    i_cone(1:num_con*(1/3)) = i_cone(1:num_con*(1/3)) + rgb4train(1);                             %Red
    i_cone(num_con*(1/3)+1:num_con*(2/3)) = i_cone(num_con*(1/3)+1:num_con*(2/3)) + rgb4train(2); %Green
    i_cone(num_con*(2/3)+1:num_con) = i_cone(num_con*(2/3)+1:num_con) + rgb4train(3);             %Blue


    % get the eye activity based on spontaneous activity and the sigmoid
    % function
    for n = 1:num_con
        cone(n) = sigmoid(i_cone(n),gain);
    end

    % get the V1 activity based on the eye activity and the weights between
    % the eye and V1
    i_v1 = zeros(1,num_rgb);
    for n = 1:num_rgb
        i_v1(n) = sum(cone .* w(:,n)');
        v1(n) = sigmoid(i_v1(n),gain);
    end

    % updated the weights using the BCM rule
    for i=1:num_con
        for j = 1:num_rgb
            w(i,j) = w(i,j) + bcm(cone(i),v1(j),v1_thr(j));
        end
    end

    % update the thresholds using the BCM rule
    for i=1:num_rgb
        v1_thr(i) = v1_thr(i) + bcm_thr(v1(i),v1_thr(i));
    end
end

% Stimulate each eye. Calculate the ocular dominance as described in
% Rittenhouse Nature 1999.
k = num_con*(1/3)
red = zeros(1, 10);
blu = zeros(1, 10);
grn = zeros(1, 10);
Col = zeros(3,num_rgb);

% Stimulate red receptors. Record the response.
i_cone = spont_act*ones(1,num_con);
i_cone(1:num_con*(1/3)) = i_cone(1:num_con*(1/3)) + rgb4test(1);                  %Red
i_cone(num_con*(1/3)+1:num_con*(2/3)) = i_cone(num_con*(1/3)+1:num_con*(2/3)) +  + rgb4test(2);  %Green
i_cone(num_con*(2/3)+1:num_con) = i_cone(num_con*(2/3)+1:num_con)  + rgb4test(2);              %Blue
for n = 1:num_con
    cone(n) = sigmoid(i_cone(n),gain);
end
i_v1 = zeros(1,num_rgb);
for n = 1:num_con*(1/3)
    i_v1(n) = sum(cone .* w(:,n)');
    red(n) = sigmoid(i_v1(n),gain);
end


for n = num_con*(1/3)+1 : num_con*(2/3)
    i_v1(n) = sum(cone .* w(:,n)');
    grn(n - num_con*(1/3)) = sigmoid(i_v1(n),gain);
end


for n = num_con*(2/3)+1:num_rgb
    i_v1(n) = sum(cone .* w(:,n)');
    blu(n - num_con*(2/3)) = sigmoid(i_v1(n),gain);
end
% % Stimulate green receptors. Record the response.
% i_cone = spont_act*ones(1,num_con);
% i_cone(1:num_con*(1/3)) = 0 %i_cone(1:num_con*(1/3));                                          %Red
% i_cone(num_con*(1/3)+1:num_con*(2/3)) = i_cone(num_con*(1/3)+1:num_con*(2/3)) + rgb4test(2); %Green
% i_cone(num_con*(2/3)+1:num_con) = 0%i_cone(num_con*(2/3)+1:num_con);                          %Blue
% for n = 1:num_con
%     cone(n) = sigmoid(i_cone(n),gain);
% end
% i_v1 = zeros(1,num_v1);
% for n = 1:num_v1
%     i_v1(n) = sum(cone .* w(:,n)');
%     grn(n) = sigmoid(i_v1(n),gain);
% end
% 
% % Stimulate blue receptors. Record the response.
% i_cone = spont_act*ones(1,num_con);
% i_cone(1:num_con*(1/3)) = 0%i_cone(1:num_con*(1/3));                                  %Red
% i_cone(num_con*(1/3)+1:num_con*(2/3)) = 0%i_cone(num_con*(1/3)+1:num_con*(2/3));      %Green
% i_cone(num_con*(2/3)+1:num_con) = i_cone(num_con*(2/3)+1:num_con) + rgb4test(3);    %Blue
% for n = 1:num_con
%     cone(n) = sigmoid(i_cone(n),gain);
% end
% i_v1 = zeros(1,num_v1);
% for n = 1:num_v1
%     i_v1(n) = sum(cone .* w(:,n)');
%     blu(n) = sigmoid(i_v1(n),gain);
% end
% 
% % calculate the ocular dominance metric
for n = 1:k
%     Col(n) = ((red(n)-spont_act)-(grn(n)-spont_act)) / ((red(n)-spont_act)+(grn(n)-spont_act));
    Col(:,n) = [red(n) grn(n) blu(n)];
end
Col=Col(:, 1:10)

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

function rgb_out = wavelength_rgb(target_length)
wavelength_350 = readmatrix('rgb_stim.csv');
rgb_out = wavelength_350(target_length - 350, :);
end
