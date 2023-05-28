rgbstim = readmatrix("rgby_norm.csv");
% rgby = [rgbstim(:,1)'-rgbstim(:,2)'; (rgbstim(:,1)' + rgbstim(:,2)')-rgbstim(:,3)']
% rgby = [((rgbstim(:,1)-rgbstim(:,2))/(rgbstim(:,1)+rgbstim(:,2))) (((rgbstim(:,1) + rgbstim(:,2))/2-rgbstim(:,3))/((rgbstim(:,1) + rgbstim(:,2))/2+rgbstim(:,3)))]
rgby = [(rgbstim(:,1)-rgbstim(:,2)) ((rgbstim(:,1) + rgbstim(:,2))/2-rgbstim(:,3))];
% scatter(rgby(:,1), rgby(:,2))
plot(rgby(2:300,1), Color='r', LineWidth=2)
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('Wavelength (Hz)')
hold on 
plot(rgby(2:300,2),Color='b', LineWidth=2)
legend('Red-Green', 'Blue-Yellow','Location','northwest')
title('Color Opponency Spectrum')
box off

figure
plot((1:1:299),rgbstim(2:300,1),"Color",'R','LineWidth',2)
set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('Wavelength (Hz)')
hold on
plot((1:1:299),rgbstim(2:300,2),"Color",'G','LineWidth',2)
plot((1:1:299),rgbstim(2:300,3),"Color",'B','LineWidth',2)
plot((1:1:299),rgbstim(2:300,4),"Color",'Y','LineWidth',2)
title('Wavelength to V1 Color Activity')
box off
