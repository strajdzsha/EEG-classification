im1 = importdata('C:/Users/Zephyrus/Desktop/sig1.txt')
im2 = importdata('C:/Users/Zephyrus/Desktop/sig2.txt')+ 200
im3 = importdata('C:/Users/Zephyrus/Desktop/sig4.txt')+ 400
im4 = importdata('C:/Users/Zephyrus/Desktop/sig5.txt')+ 600
im5 = importdata('C:/Users/Zephyrus/Desktop/sig6.txt')+ 800
im6 = importdata('C:/Users/Zephyrus/Desktop/sig7.txt')+ 1000
im7 = importdata('C:/Users/Zephyrus/Desktop/sig8.txt')+ 1200
im8 = importdata('C:/Users/Zephyrus/Desktop/sig9.txt')+ 1400

figure
plot(im1, 'Color', [230/255,75/255,53/254], 'LineWidth',2)
hold on
plot(im2, 'Color', [77,187,213]/255, 'LineWidth',2)
hold on
plot(im3, 'Color', [0,160,135]/255, 'LineWidth',2)
hold on
plot(im4, 'Color', [60,84,136]/255, 'LineWidth',2)
hold on
plot(im5, 'Color', [243,155,127]/255, 'LineWidth',2)
yticks([])

figure
plot(im1(1:4000), 'Color', [60,84,136]/255, 'LineWidth',2)


