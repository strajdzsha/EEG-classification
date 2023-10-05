im1 = importdata('C:/Users/Zephyrus/Desktop/data_bp1.txt')
im2 = importdata('C:/Users/Zephyrus/Desktop/data_bp2.txt')
im3 = importdata('C:/Users/Zephyrus/Desktop/data_bp3.txt')
im4 = importdata('C:/Users/Zephyrus/Desktop/data_bp4.txt')
im5 = importdata('C:/Users/Zephyrus/Desktop/data_bp5.txt')

figure
load carsmall
subplot(2,3,1)
boxplot(im1(:, 1:2), 'Notch','on')
set(gca,'XTick',1:2,'XTickLabel',{"Alzheimer's Disease",'Control Group'})
ylim([-0.4 80])
title('hjorth activity alpha 9')
set(gca,'FontSize', 12,'FontName', 'Times')
h = findobj(gca,'Tag','Box');
patch(get(h(1),'XData'),get(h(1),'YData'),[77,187,213]/255,'FaceAlpha',.5);
patch(get(h(2),'XData'),get(h(2),'YData'),[230/255,75/255,53/254],'FaceAlpha',.5);

subplot(2,3,2)
boxplot(im2(:, 1:2), 'Notch','on')
set(gca,'XTick',1:2,'XTickLabel',{'CN','AD'})
ylim([-0.4 83])
title('hjorth activity beta 9')
set(gca,'FontSize', 12,'FontName', 'Times')
h = findobj(gca,'Tag','Box');
patch(get(h(1),'XData'),get(h(1),'YData'),[77,187,213]/255,'FaceAlpha',.5);
patch(get(h(2),'XData'),get(h(2),'YData'),[230/255,75/255,53/254],'FaceAlpha',.5);

subplot(2,3,3)
boxplot(im3(:, 1:2), 'Notch','on')
set(gca,'XTick',1:2,'XTickLabel',{'CN','AD'})
ylim([1.01 1.041])
title('hjorth complexity alpha 8')
set(gca,'FontSize', 12,'FontName', 'Times')
h = findobj(gca,'Tag','Box');
patch(get(h(1),'XData'),get(h(1),'YData'),[77,187,213]/255,'FaceAlpha',.5);
patch(get(h(2),'XData'),get(h(2),'YData'),[230/255,75/255,53/254],'FaceAlpha',.5);

subplot(2,3,4)
boxplot(im4(:, 1:2), 'Notch','on')
set(gca,'XTick',1:2,'XTickLabel',{'CN','AD'})
ylim([-0.4 83])
title('hjorth activity beta 13')
set(gca,'FontSize', 12,'FontName', 'Times')
h = findobj(gca,'Tag','Box');
patch(get(h(1),'XData'),get(h(1),'YData'),[77,187,213]/255,'FaceAlpha',.5);
patch(get(h(2),'XData'),get(h(2),'YData'),[230/255,75/255,53/254],'FaceAlpha',.5);

subplot(2,3,5)
boxplot(im5(:, 1:2), 'Notch','on')
set(gca,'XTick',1:2,'XTickLabel',{'CN','AD'})
title('hjorth activity gamma 11')
set(gca,'FontSize', 12,'FontName', 'Times')
h = findobj(gca,'Tag','Box');
patch(get(h(1),'XData'),get(h(1),'YData'),[77,187,213]/255,'FaceAlpha',.5);
patch(get(h(2),'XData'),get(h(2),'YData'),[230/255,75/255,53/254],'FaceAlpha',.5);
