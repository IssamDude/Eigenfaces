clear all;
close all;

load eigenfaces_masque;

h = figure('Position',[0,0,0.67*L,0.67*H]);
figure('Name','RMSE en fonction du nombre de composantes principales','Position',[0.67*L,0,0.33*L,0.3*L]);

% Calcul de la RMSE entre images originales et images reconstruites :
RMSE_masque_max = 0;

% Composantes principales des données d'apprentissage
%la matrice U representant la base des vecteurs propres de sigma
U = vecteurs_propres_masque;
C = X_centre_masque*U;

for q = 0:n_masque-1
    Uq = U(:,1:q);  % q premières eigenfaces
    Cq = C(:,1:q);		% q premières composantes principales	
    X_reconstruit_masque = (Uq*Cq' + individu_moyen_masque')';
    figure(1);
    set(h,'Name',['Utilisation des ' num2str(q) ' premieres composantes principales']);
    colormap gray;
    hold off;
    for k = 1:n_masque
        subplot(nb_personnes_base, nb_postures_base,k);
        img = reshape(X_reconstruit_masque(k,:), nb_lignes, nb_colonnes);
        imagesc(img);
        hold on;
        axis image;
        axis off;
    end
    
    figure(2);
    hold on;

    RMSE_masque = sqrt(mean2((X_masque-X_reconstruit_masque).^2));
    RMSE_masque_max = max(RMSE_masque,RMSE_masque_max);

    plot(q,RMSE_masque,'r+','MarkerSize',8,'LineWidth',2);
    axis([0, n_masque-1, 0, 1.1*RMSE_masque_max]);
    set(gca,'FontSize',20);
    hx = xlabel('$q$','FontSize',30);
    set(hx,'Interpreter','Latex');
    ylabel('RMSE','FontSize',30);
    
    pause(0.01);
end

save projection_masque;
