function [t,AUC] = plotMRvsAccuracy(MRsize,predictedGrade,realGrade,CancerPercentage)

cnt = 0;
t = [0 : 0.3 : 1,1.6,1.8, 2:0.3:2.7];

ACC = zeros(1,size(t,2));
AUC = zeros(1,size(t,2));
SEN = zeros(1,size(t,2));
SPEC = zeros(1,size(t,2));
S = zeros(1,size(t,2));
Cmat = zeros(size(t,2),3);
Cmat(1,:) = [1.0 0.5 0.5];
Cmat(2,:) = [0.5 0.5 1.0];
Cmat(3,:) = [0.4 1.0 0.4];
Cmat(4,:) = [0.5 0.4 0.6];


figure1 = figure('Color','None');
axes1 = axes('Parent',figure1,'FontSize',13,'FontName','Times');
box(axes1,'on');
hold on
for j = t 
    
    filter_Core = find(MRsize >= j & MRsize ~=100);
    cnt = cnt + 1;
    S(cnt) = size(filter_Core,1);
    L1 = predictedGrade(filter_Core,1);
    L2 = realGrade(filter_Core,1);
    L3 = CancerPercentage(filter_Core,1);
    [~, ~, ~, ~,SEN(cnt),SPEC(cnt)] = findStatResult(L1,L2);
    ACC(cnt) = (1 - sum(L1~=L2) / size(filter_Core,1));
    [X_ROC,Y_ROC,~,AUC(cnt)] = perfcurve(L2,L3,1);
    if( mod(cnt,2) == 0 )
        plot(X_ROC,Y_ROC,'Color',Cmat(cnt/2,:),'LineWidth',2.0,'LineStyle','--',...
            'DisplayName',sprintf('Larger than %2.2g cm',j))
        xlabel('False positive rate (1-Specificity)','Interpreter','latex','FontSize',13); 
        ylabel('True positive rate (Specificity)','Interpreter','latex','FontSize',13);
    end
            
end
legend(axes1,'show');
hold off

% Create figure
figure2 = figure('Color',[1 1 1]);
axes2 = axes('Parent',figure2,'FontSize',13,'FontName','Times');
box(axes2,'on');
hold(axes2,'all');

% Create scatter
% h1=scatter(t,ACC,'MarkerEdgeColor',[1 0.5 0.5],'DisplayName','Accuracy'); plot(t,ACC,'LineStyle',':','Color',[1 0 0]);
% h2=scatter(t,SPEC,'MarkerEdgeColor',[0.5 0.5 1],'DisplayName','Specificty'); plot(t,SPEC,'LineStyle',':','Color',[0 0 1]);
% h3=scatter(t,SEN,'MarkerEdgeColor',[0.5 1 0.5],'DisplayName','Sensitivity'); plot(t,SEN,'LineStyle',':','Color',[0 1 0]);
h4=scatter(t,AUC,'MarkerEdgeColor',[0.5 0.5 1],'DisplayName','AUC','LineWidth',1.5); plot(t,AUC,'LineStyle','--','Color',[0.5 0.5 1],'LineWidth',1.5);
xlabel('Greatest Tumor Length in MRI','Interpreter','latex','FontSize',13);
ylabel('Area Under the Curve (AUC)','Interpreter','latex','FontSize',13);
% legend([h1,h4]);
