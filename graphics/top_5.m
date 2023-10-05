% bar plot for top 5 features 

figure

names = {"hjorth\newlineactivity\newlinealpha 9      ", "hjorth\newlineactivity\newlinebeta 9        ", "hjorth\newlinecomplexity\newlinealpha 9", "hjorth\newlineactivity\newlinebeta 13      ", "hjorth\newlineactivity\newlinegamma 11 "};

igs = [0.144656 0.129898 0.129025 0.046154 0.044443];
names = fliplr(names)
igs = fliplr(igs)

x = [1:5]; 
b = barh(x,igs)
set(gca,'yticklabel',names)
b.FaceColor = "#3c5488";
        
for i=1:5
    y = b.YData(i) + 0.005;
    s = sprintf('%.3f', b.YData(i));
    text(y, b.XData(i), s);
end

xlim([0,0.175])

xlabel('Information Gain (IG)')
set(gca,'FontSize', 12,'FontName', 'Times')
