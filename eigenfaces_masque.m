%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AVEC MASQUE
%%%%%%%%%%%%%
clear all;
close all;

% Dimensions du masque
ligne_min = 200;
ligne_max = 350;
colonne_min = 60;
colonne_max = 290;

%%%%%%%% LECTURE DES DONNES ET AJOUT DU MASQUE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% liste des differentes personnes
liste_personnes = {
 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'm01', 'm02', 'm03', 'm04', 'm05', 'm06', 'm07', 'm08', 'm09', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15', 'm16'
                   };
nb_personnes = length(liste_personnes);
% liste des differentes postures 
liste_postures = {'v1e1','v3e1','v1e2','v3e2','v1e3','v3e3'};
nb_postures = length(liste_postures);

nb_lignes = 400;
nb_colonnes = 300;
% personnes constituant la base d'apprentissage (A FAIRE EVOLUER)
liste_personnes_base = {'f01', 'f10', 'm01', 'm08'};
%       personnes          1     10     17     24     
%liste_personnes_base = {'f01', 'f10', 'm10', 'm08'} clusters mieux séparés
nb_personnes_base = length(liste_personnes_base); 

% postures de la base d'apprentissage (A FAIRE EVOLUER)
liste_postures_base = [1 2 3 4];
nb_postures_base = length(liste_postures_base);


X_masque = [];
taille_ecran = get(0,'ScreenSize');
L = taille_ecran(3);
H = taille_ecran(4); 
figure('Name','Personnes avec Masque','Position',[0,0,0.67*L,0.67*H]);
colormap(gray(256));
 
for j = 1:nb_personnes_base,
    no_posture = 0;
    for k = liste_postures_base,
        no_posture = no_posture + 1;
        ficF = strcat('./Data/', liste_personnes_base{j}, liste_postures{k}, '-300x400.gif');
        img = imread(ficF);
         
        % Degradation de l'image
        img(ligne_min:ligne_max,colonne_min:colonne_max) = 0;
        % Remplissage de la matrice X_masque :
        X_masque= [X_masque; double(transpose(img(:)))];
        
        % Affichage
 	    subplot(nb_personnes_base, nb_postures_base, (j-1)*nb_postures_base + no_posture);
 	    imagesc(img);
	    hold on;
        axis image;
        title(['Personne ' liste_personnes_base{j} ', posture ' num2str(k)]);         
    end
end

%%%%%% REFAIRE LE CALCUL ET L'AFFICHAGE DES EIGENFACES AVEC MASQUE

% Calcul de l'individu moyen :
n_masque = size(X_masque,1);
individu_moyen_masque = mean(X_masque);

% Centrage de la matrice X_masque :
X_centre_masque = X_masque - individu_moyen_masque;

% Calcul de la matrice de covariance (impossible a calculer ainsi a cause de sa taille) :
%Sigma_masque = transpose(X_centre_masque)*X_centre_masque/n;
%impossible a calculer Requested 120000x120000 (107.3GB) array exceeds maximum array size preference (15.6GB)

% Calcul de la matrice resultant du calcul commuté (voir annexe du sujet) :
Sigma2_masque = X_centre_masque*transpose(X_centre_masque)/n_masque;

% Calcul des vecteurs/valeurs propres de la matrice Sigma2 :
[vecteurs_propres_sigma2_masque, valeurs_propres_sigma2_masque] = eig(Sigma2_masque);

% Les vecteurs propres de Sigma (les eigenfaces) se deduisent de ceux de Sigma2 :
vecteurs_propres_masque = X_centre_masque'*vecteurs_propres_sigma2_masque;

% Tri par ordre decroissant des valeurs propres de Sigma_barre :
[valeurs_propres_masque , indices_masque] = sort(diag(valeurs_propres_sigma2_masque),'descend');


% Tri des eigenfaces dans le meme ordre 
vecteurs_propres_masque = vecteurs_propres_masque(:, indices_masque);
% (on enleve la derniere eigenface, qui appartient au noyau de Sigma) :
vecteurs_propres_masque = vecteurs_propres_masque( : , 1 : size(vecteurs_propres_masque,2)-1 );

% Normalisation des eigenfaces :
vecteurs_propres_masque = normc(vecteurs_propres_masque);

% Affichage de l'individu moyen avec masque et des eigenfaces sous la forme de "pseudo-images" 
% (leurs coordonnees sont interpretees comme des niveaux de gris) :
figure('Name','Individu moyen et eigenfaces avec masque', 'Position', [0,0,0.67*L,0.67*H]);
colormap(gray(256)); 
img = reshape(individu_moyen_masque, nb_lignes, nb_colonnes);
subplot(nb_personnes_base, nb_postures_base, 1)
imagesc(img); 
hold on; 
axis image; 
title(['Individu moyen avec masque']);
for k = 1:n_masque-1,
	img = reshape(vecteurs_propres_masque(:,k), nb_lignes,nb_colonnes);
	subplot(nb_personnes_base, nb_postures_base,k+1);
	imagesc(img); 
	hold on; 
	axis image; 
	title(['Eigenface_masqué ', num2str(k)]);
end

save eigenfaces_masque;