clear
close all

%%

% load("./best_beams_original/Best_beams_BS3_new.mat")
% load("./best_beams_omni/Best_beams.mat")
% load('Best_beams_BS3_new.mat')
% load('Best_beams_BS4_new.mat')
load('Learned_codebook.mat')

F = (1/4)*exp(1j*beams.');

% BS3 middle beam: 16
% BS4 middle beam: 12

%%

serv_dir = zeros(2, size(F, 2));

% plot_codebook_db(F(:,5))

fig_count = 1;
for ii = 1:size(F, 2)
    
    h = figure(fig_count);
    if ii ~= 16
        plot_codebook_db_with_directions_v2(F(:, ii), 1)
        fig_count = fig_count + 1;
    elseif ii == 16
        plot_codebook_db_with_directions_v2(F(:, ii), 2)
        fig_count = fig_count + 1;
    else
        
    end
    
    % savefig(h, ['./BS3_beam_patterns_v2/beam_', num2str(ii), '.fig'])
    % pause(3)
    
end
