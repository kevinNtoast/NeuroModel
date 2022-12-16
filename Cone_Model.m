%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Kevin Nam
%
% A Neural model looking into Prevalence-Induced Concept change through the
% implementation of the BCM rule.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%The Test and Training wavelengths can be changed in the following two
%variables.
%Valid wavelength 400 < wavelength < 699
train_wavelength = 680;


train = wavelength_rgb(train_wavelength);



[th_rg_col,w_rg_col, m_rg_col]=cone_model(train(1), train(2), 'rg'); % run model with both eyes stimulated
w_rg_col = w_rg_col / max(abs(w_rg_col)); % normalize the ocular dominance metric between -1 and 1

[th_rg_non,w_rg_non, m_rg_non]=cone_model(0, 0, 'rg'); % run model with only right eye stimulated
w_rg_non = w_rg_non / max(abs(w_rg_non));

[th_by_col,w_by_col, m_by_col]=cone_model(train(3), train(4), 'by'); % run model with both eyes stimulated
w_by_col = w_by_col / max(abs(w_by_col)); % normalize the ocular dominance metric between -1 and 1

[th_by_non,w_by_non, m_by_non]=cone_model(0, 0, 'by'); % run model with only right eye stimulated
w_by_non = w_by_non / max(abs(w_by_non));



figure
col_space_a = [-m_rg_col; m_by_col];
% col_space_a = [cos(-pi/6) -sin(-pi/6); sin(-pi/6) cos(-pi/6)]*col_space_a; 
% col_space_a = col_space_a;
scatter(col_space_a(1,:), col_space_a(2,:),'filled', 'blue');
hold on
col_space_b = [-m_rg_non; m_by_non];
% col_space_b = [cos(-pi/6) -sin(-pi/6); sin(-pi/6) cos(-pi/6)]*col_space_b;
% col_space_b = col_space_b;
scatter(col_space_b(1,:), col_space_b(2,:),'filled', 'red');
legend('Post-training','Pre-Training')

col_a = col_space_a';
col_b = col_space_b';
cols_ab = col_a - col_b;
[theta rho] = cart2pol(cols_ab(:,1), cols_ab(:,2));
kk = [rho.*cos(theta) rho.*sin(theta)];
quiver(col_b(:,1), col_b(:,2), kk(:,1), kk(:,2), 2.5,'HandleVisibility','off');
set(gca,'xtick',[])
set(gca,'ytick',[])
title("Train : " + num2str(train_wavelength) + " nm")

figure
inx = 0;
x = 0.01:0.01:1;
col_train = zeros(1,size(x,2));
non_train = zeros(1,size(x,2));
no_lrn = zeros(1,size(x,2));
for i=1:size(x,2)
    inx = inx + 1;
    col_train(inx) = bcm(1, x(i), median(th_rg_col));
    non_train(inx) = bcm(1, x(i), median(th_rg_non));
end
plot(x, col_train, 'LineWidth',2)
hold on
plot(x, non_train, 'LineWidth',2)
plot(x,no_lrn,'.-')
title('BCM Learning Curve')
legend('Post-Training', 'Pre-Training', 'Location','northwest')
xlabel('Postsynaptic Activity')
ylabel('dW')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%|   %%%%|    %%%%%%|     %%%%%%%|       %%%%%%%|      %%%%|  %%%%%%%%%%%%%
%%%|  | %%| %|  %%%%|  %%%%|  %%%%%|  %%%%|  %%%%%|  %%%%%%%%|  %%%%%%%%%%%%%
%%%|  %%|  %%|  %%%%|  %%%%|  %%%%%|  %%%%|  %%%%%|      %%%%|  %%%%%%%%%%%%%
%%%|  %%%%%%%|  %%%%|  %%%%|  %%%%%|  %%%%|  %%%%%|  %%%%%%%%|  %%%%%%%%%%%%%
%%%|  %%%%%%%|  %%%%%|      %%%%%%%|       %%%%%%%|      %%%%|      %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function cone_model
%
% Input: 
% red_blu - The first color to be put in the model, represents red for red-green axis and blue for blue-yellow axis.
% green_yellow - The second color to be put in the model, represents green
% for red-green axis and yellow for blue-yellow axis.
% color_opp - the color opponency being used:
%           - 'rg' - red-green axis
%           - 'by' - blue-yellow axis
% Outputs:
% v1_thr - the threshold at v1 for the model for number of v1 neurons. (In
% this case, 100)
% v1_w   - List of weights at V1 for the model
% mean_cl- The mean weights obtained per iteration of the testing phase.


function[v1_thr, v1_w, mean_cl] = cone_model(red_blu, green_yellow, color_opp)

if color_opp == 'rg'
    stim1 = 1;
    stim2 = 2;
else
    stim1 = 3;
    stim2 = 4;
end

Training = 50;
num_cones = 20;
num_v1 = 100;
cones = zeros(1,num_cones);
v1 = zeros(1,num_v1);

% random uniform weight distribution between eye and V1
w = 0.1*rand(num_cones,num_v1);
gain = 0.5; % gain of sigmoid function
spont_act = -1.0; 
v1_thr = 0.5*ones(1,num_v1);

% Network Training
for t = 1:Training

    i_cone = spont_act*ones(1,num_cones)+(-0.25+0.5*rand(1,num_cones));
    i_cone(1:num_cones/2) = i_cone(1:num_cones/2)+red_blu;
    i_cone(num_cones/2+1:num_cones) = i_cone(num_cones/2+1:num_cones)+green_yellow;

    for n = 1:num_cones
        cones(n) = sigmoid(i_cone(n),gain);
    end

    i_v1 = zeros(1,num_v1);
    for n = 1:num_v1
        i_v1(n) = sum(cones .* w(:,n)');
        v1(n) = sigmoid(i_v1(n),gain);
    end

    for i=1:num_cones
        for j = 1:num_v1
            w(i,j) = w(i,j) + bcm(cones(i),v1(j),v1_thr(j));
        end
    end

    for i=1:num_v1
        v1_thr(i) = v1_thr(i) + bcm_thr(v1(i),v1_thr(i));
    end
end

c1 = zeros(1,num_v1);
c2 = zeros(1,num_v1);
v1_w = zeros(1,num_v1);
for wv = 1:299
    rgby = wavelength_rgb(wv+400);

    i_cone = spont_act*ones(1,num_cones);
    i_cone(1:num_cones/2) = i_cone(1:num_cones/2)+rgby(stim1);
    for n = 1:num_cones
        cones(n) = sigmoid(i_cone(n),gain);
    end
    i_v1 = zeros(1,num_v1);
    for n = 1:num_v1
        i_v1(n) = sum(cones .* w(:,n)');
        c1(n) = sigmoid(i_v1(n),gain);
    end
    
    i_cone = spont_act*ones(1,num_cones);
    i_cone(num_cones/2+1:num_cones) = i_cone(num_cones/2+1:num_cones)+rgby(stim2);
    for n = 1:num_cones
        cones(n) = sigmoid(i_cone(n),gain);
    end
    i_v1 = zeros(1,num_v1);
    for n = 1:num_v1
        i_v1(n) = sum(cones .* w(:,n)');
        c2(n) = sigmoid(i_v1(n),gain);
    end
    
    for n = 1:num_v1
        v1_w(n) = ((c2(n)-spont_act)-(c1(n)-spont_act)) / ((c2(n)-spont_act)+(c1(n)-spont_act));
    end
    mean_cl(wv) = mean(v1_w);
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
wavelength_400 = readmatrix('rgby.csv');
rgby_out = wavelength_400(target_length - 399, :);
rgby_out = rgby_out/max(rgby_out);
end
