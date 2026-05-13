% =========================================================================
% visualizations.m
% Generates publication-quality figures comparing FCS-DEMPC and CCS-DEMPC.
%
% Assumes the workspace contains:
%   - results_dempc (from FCS-DEMPC simulation)
%   - results_gwo   (from CCS-DEMPC simulation)
% =========================================================================

% Verify that data is loaded
if ~exist('results_dempc', 'var') || ~exist('results_gwo', 'var')
    error('Results not found in workspace. Please run test_comparison.m or the individual simulations first.');
end

close all;
clc;

t = results_dempc.t;

set(0,'DefaultFigureColor','w')

% Recreate disturbance profiles for plotting
G_profile = @(t) 1000 .* (t < 0.2) + ...
                 (1000 + 100*(t-0.2)/0.2) .* (t >= 0.2 & t < 0.4) + ...
                 1100 .* (t >= 0.4 & t < 0.6) + ...
                 1000 .* (t >= 0.6);

Pload_profile = @(t) 20000 .* (t < 0.2) + ...
                     (20000 + 20000*(t-0.2)/0.2) .* (t >= 0.2 & t < 0.4) + ...
                     40000 .* (t >= 0.4 & t < 0.6) + ...
                     50000 .* (t >= 0.6);

G_t     = arrayfun(G_profile, t);
Pload_t = arrayfun(Pload_profile, t) / 1000; % kW
Pmax_t  = results_dempc.Pmax / 1000;         % kW
Pnet_t  = Pmax_t - Pload_t;                  % kW

stages = [0.2, 0.4, 0.6]; % Stage boundaries

% =========================================================================
% % COMBINED FIGURE: Figure 1 + Figure 2 as subplot(4,1,4)
% % =========================================================================
% f_combined = figure('Name', 'Combined Figure 1 and 2', ...
%     'Position', [100, 50, 900, 1000], 'Color', 'w');
% 
% %% ------------------------------------------------------------------------
% % (a) Irradiance and Load
% % -------------------------------------------------------------------------
% ax1 = subplot(4,1,1);
% yyaxis left;
% plot(t, G_t, 'b-', 'LineWidth', 2);
% ylabel('Irradiance G (W/m^2)');
% ylim([900 1200]);
% 
% yyaxis right;
% plot(t, Pload_t, 'r-', 'LineWidth', 2);
% ylabel('Load P_{load} (kW)');
% ylim([15 55]);
% 
% xline(stages, 'k--');
% legend('Solar irradiance', 'Load power', 'Location', 'west');
% text(ax1, 0.5, -0.25, '(a)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% 
% grid on;
% xlim([0 0.8]);
% 
% %% ------------------------------------------------------------------------
% % (b) EMS Flags
% % -------------------------------------------------------------------------
% ax2 = subplot(4,1,2);
% stairs(t, results_dempc.zeta(:,1), 'b-', 'LineWidth', 2); hold on;
% stairs(t, results_dempc.zeta(:,2), 'r--', 'LineWidth', 2);
% 
% xline(stages, 'k--');
% ylabel('EMS Flags');
% yticks([0 1]);
% ylim([-0.2 1.4]);
% 
% legend('\zeta_a', '\zeta_p', 'Location', 'best');
% 
% text(ax2, 0.5, -0.25, '(b)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% 
% grid on;
% xlim([0 0.8]);
% 
% %% ------------------------------------------------------------------------
% % (c) Net Power
% % -------------------------------------------------------------------------
% ax3 = subplot(4,1,3);
% plot(t, Pnet_t, 'm-', 'LineWidth', 1.5);
% 
% xline(stages, 'k--');
% xlabel('Time (s)');
% ylabel('P_{net} (kW)');
% 
% legend('Net power balance', 'Location', 'best');
% 
% text(ax3, 0.5, -0.35, '(c)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% 
% grid on;
% xlim([0 0.8]);
% 
% %% ------------------------------------------------------------------------
% % (d) DC Bus Voltage Regulation
% % -------------------------------------------------------------------------
% ax4 = subplot(4,1,4);
% plot(t, results_dempc.X(:,5), 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.X(:,5), 'r--', 'LineWidth', 1.5);
% plot(t, results_dempc.Vdc_ref, 'k:', 'LineWidth', 1.5);
% 
% xline(stages, 'k--');
% 
% xlabel('Time (s)');
% ylabel('V_{dc} (V)');
% 
% legend('DMPC', 'Proposed GWO-DMPC', 'V_{dc,ref}', ...
%     'Location', 'northwest');
% 
% text(ax4, 0.5, -0.30, '(d)', 'Units', 'normalized', ...
%     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% 
% grid on;
% xlim([0 0.8]);
% ylim([285 315]);
% 
% %% ------------------------------------------------------------------------
% % OPTIONAL: Add inset zooms inside subplot (d)
% % -------------------------------------------------------------------------
% 
% % Zoom 1
% xlim1 = [0.185 0.210];
% idx1 = (t >= xlim1(1) & t <= xlim1(2));
% ymin1 = min([results_dempc.X(idx1,5); results_gwo.X(idx1,5); results_dempc.Vdc_ref(idx1)]);
% ymax1 = max([results_dempc.X(idx1,5); results_gwo.X(idx1,5); results_dempc.Vdc_ref(idx1)]);
% 
% inset1 = axes('Position', [0.25 0.13 0.12 0.08]);
% box on; hold on; grid on;
% plot(t, results_dempc.X(:,5), 'b-');
% plot(t, results_gwo.X(:,5), 'r--');
% plot(t, results_dempc.Vdc_ref, 'k:');
% xlim(xlim1);
% ylim([ymin1-0.5 ymax1+0.5]);
% 
% % Zoom 2
% xlim2 = [0.56 0.65];
% idx2 = (t >= xlim2(1) & t <= xlim2(2));
% ymin2 = min([results_dempc.X(idx2,5); results_gwo.X(idx2,5); results_dempc.Vdc_ref(idx2)]);
% ymax2 = max([results_dempc.X(idx2,5); results_gwo.X(idx2,5); results_dempc.Vdc_ref(idx2)]);
% 
% inset2 = axes('Position', [0.70 0.13 0.12 0.08]);
% box on; hold on; grid on;
% plot(t, results_dempc.X(:,5), 'b-');
% plot(t, results_gwo.X(:,5), 'r--');
% plot(t, results_dempc.Vdc_ref, 'k:');
% xlim(xlim2);
% ylim([ymin2-0.5 ymax2+0.5]);

% %% ------------------------------------------------------------------------
% sgtitle('Simulation Scenario, EMS Response, and DC Bus Voltage Regulation');

% =========================================================================
% FIGURE 1: Simulation Scenario and EMS Response
% =========================================================================
f1 = figure('Name', 'Figure 1: Simulation Scenario and EMS Response', 'Position', [100, 20, 1000, 1000], 'Color', 'w');

% Top Panel: Irradiance and Load
ax1_1 = subplot(4, 1, 1);
yyaxis left;
plot(t, G_t, 'b-', 'LineWidth', 2);
ylabel('Irradiance G (W/m^2)');
ylim([900 1200]);
yyaxis right;
plot(t, Pload_t, 'r-', 'LineWidth', 2);
ylabel('Load P_{load} (kW)');
ylim([15 55]);
xline(stages, 'k--');
legend('Solar irradiance', 'Load power', 'Location', 'West');
text(0.1, 52, 'Stage 1', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(0.3, 52, 'Stage 2', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(0.5, 52, 'Stage 3', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
text(0.7, 52, 'Stage 4', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
%title('Disturbance Profiles');
% text(ax1_1, 0.5, -0.2, '(a) Irradiance and Load profile w.r.t. time', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
text(ax1_1, 0.5, -0.2, '(a)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
grid on;
xlim([0, 0.8]);

% Middle Panel: EMS Activation Flags
ax1_2 =subplot(4, 1, 2);
stairs(t, results_dempc.zeta(:, 1), 'b-', 'LineWidth', 2); hold on;
stairs(t, results_dempc.zeta(:, 2), 'r--', 'LineWidth', 2);
xline(stages, 'k--');
ylabel('EMS Flags');
yticks([0 1]);
ylim([-0.2 1.4]);
legend('\zeta_a (Electrolyzer)', '\zeta_p (Fuel Cell)', 'Location', 'best');
%title('EMS Activation Response');
% text(ax1_2, 0.5, -0.15, '(b) EMS Activation Response', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
text(ax1_2, 0.5, -0.25, '(b)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
grid on;
xlim([0, 0.8]);

% Bottom Panel: Net Power (Surplus/Deficit)
ax1_3 = subplot(4, 1, 3);
plot(t, Pnet_t, 'm-', 'LineWidth', 1.5); hold on;
% yline(0, 'g-', 'LineWidth', 1.5);
xline(stages, 'k--');
% xlabel('Time (s)');
ylabel('Net power balance P_{net} (kW)');
legend('Net power balance', 'Location', 'best');
%title('Power Surplus and Deficit (P_{pv,max} - P_{load})');
% text(ax1_3, 0.5, -0.35, '(c) Net Power (Surplus/Deficit)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
text(ax1_3, 0.5, -0.25, '(c)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
grid on;
xlim([0, 0.8]);

ax1_4 = subplot(4,1,4);
plot(ax1_4, t, results_dempc.X(:,5), 'b-', 'LineWidth', 1.5); hold on;
plot(ax1_4, t, results_gwo.X(:,5), 'r--', 'LineWidth', 1.5);
plot(ax1_4, t, results_dempc.Vdc_ref, 'k:', 'LineWidth', 1.5);
xline(ax1_4, stages, 'k--');

xlabel('Time (s)');
ylabel('DC Bus Voltage (V)');
legend('DMPC', 'Proposed GWO-DMPC', 'V_{dc,ref}', 'Location', 'south');
% text(ax_main2, 0.5, -0.15, 'DC Bus Voltage regulation', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
grid on;
text(ax1_4, 0.5, -0.30, '(d)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
xlim([0, 0.8]);
ylim([285, 305]);

% --- Zoom 1 (t = 0.2 s) ---
xlim1 = [0.195, 0.212];
idx1 = (t >= xlim1(1) & t <= xlim1(2));
% ymin1 = min([min(results_dempc.X(idx1,5)), min(results_gwo.X(idx1,5)), min(results_dempc.Vdc_ref(idx1))]);
% ymax1 = max([max(results_dempc.X(idx1,5)), max(results_gwo.X(idx1,5)), max(results_dempc.Vdc_ref(idx1))]);
ymin1 = 299.8;
ymax1 = 300.2;
% margin1 = max(0.5, (ymax1 - ymin1) * 0.2);
margin1 = ((ymax1 - ymin1) * 0.2);
% Draw bounding box on main axes
rectangle(ax1_4, 'Position', [xlim1(1), ymin1-margin1, xlim1(2)-xlim1(1), (ymax1+margin1)-(ymin1-margin1)], 'EdgeColor', 'k', 'LineStyle', '-.');

% Create Inset Axes
ax_ins1 = axes('Position', [0.2, 0.135, 0.08, 0.08]); % Bottom-leftish
box on; grid on; hold on;
plot(ax_ins1, t, results_dempc.X(:,5), 'b-', 'LineWidth', 1.5);
plot(ax_ins1, t, results_gwo.X(:,5), 'r--', 'LineWidth', 1.5);
plot(ax_ins1, t, results_dempc.Vdc_ref, 'k:', 'LineWidth', 1.5);
xlim(ax_ins1, xlim1);
ylim(ax_ins1, [ymin1-margin1, ymax1+margin1]);
%title(ax_ins1, 'Zoom: 0.18s - 0.28s', 'FontSize', 9);

% --- Zoom 2 (t = 0.6 s) ---
xlim2 = [0.56, 0.65];
idx2 = (t >= xlim2(1) & t <= xlim2(2));
ymin2 = min([min(results_dempc.X(idx2,5)), min(results_gwo.X(idx2,5)), min(results_dempc.Vdc_ref(idx2))]);
ymax2 = max([max(results_dempc.X(idx2,5)), max(results_gwo.X(idx2,5)), max(results_dempc.Vdc_ref(idx2))]);
margin2 = max(0.5, (ymax2 - ymin2) * 0.2);

% Draw bounding box on main axes
rectangle(ax1_4, 'Position', [xlim2(1), ymin2-margin2, xlim2(2)-xlim2(1), (ymax2+margin2)-(ymin2-margin2)], 'EdgeColor', 'k', 'LineStyle', '-.');

% Create Inset Axes
ax_ins2 = axes('Position', [0.8, 0.135, 0.08, 0.08]); % Bottom-rightish
box on; grid on; hold on;
plot(ax_ins2, t, results_dempc.X(:,5), 'b-', 'LineWidth', 1.5);
plot(ax_ins2, t, results_gwo.X(:,5), 'r--', 'LineWidth', 1.5);
plot(ax_ins2, t, results_dempc.Vdc_ref, 'k:', 'LineWidth', 1.5);
xlim(ax_ins2, xlim2);
ylim(ax_ins2, [ymin2-margin2, ymax2+margin2]);
%title(ax_ins2, 'Zoom: 0.58s - 0.68s', 'FontSize', 9);


% % =========================================================================
% % FIGURE 3: Duty Cycle Trajectories
% % =========================================================================
% f3 = figure('Name', 'Figure 3: Duty Cycle Trajectories', 'Position', [200, 100, 800, 800], 'Color', 'w');

% % PV Duty Cycle
% ax3_1 = subplot(2, 2, 1);
% stairs(t, results_dempc.U(:,1), 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.U(:,1), 'r--', 'LineWidth', 1.5);
% ylabel('d_s');
% %title('PV Duty Cycle');
% % text(ax3_1, 0.5, -0.15, '(a). PV Duty Cycle', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% text(ax3_1, 0.5, -0.15, '(a)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% grid on; xlim([0, 0.8]); ylim([-0.1 1.1]);
% % lgd1 =legend('DMPC', 'Proposed GWO-DMPC', 'Location', 'northwest');
% % lgd1.FontSize = 6;
% lgd = legend(ax3_1, 'DMPC', 'Proposed GWO-DMPC', 'P_{max}');
% set(lgd, 'Position', [0.18, 0.85, 0.1, 0.05]); % [left, bottom, width, height]
% lgd.FontSize = 6;
% %lgd1.ItemTokenSize = [10, 8];
% %legend boxoff;

% % AE Duty Cycle
% ax3_2 = subplot(2, 2, 2);
% stairs(t, results_dempc.U(:,2), 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.U(:,2), 'r--', 'LineWidth', 1.5);
% ylabel('d_{ae}');
% lgd = legend(ax3_2, 'DMPC', 'Proposed GWO-DMPC', 'P_{max}');
% set(lgd, 'Position', [0.62, 0.85, 0.1, 0.05]); % [left, bottom, width, height]
% lgd.FontSize = 6;
% % lgd.ItemTokenSize = [8, 8];
% %title('Electrolyzer Duty Cycle');
% % text(ax3_2, 0.5, -0.15, '(b). Electrolyzer Duty Cycle', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% text(ax3_2, 0.5, -0.15, '(b)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% grid on; xlim([0, 0.8]); ylim([-0.1 1.1]);

% % FC Duty Cycle
% ax3_3 = subplot(2, 2, 3);
% stairs(t, results_dempc.U(:,3), 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.U(:,3), 'r--', 'LineWidth', 1.5);
% xlabel('Time (s)');
% ylabel('d_{pe}');
% lgd = legend(ax3_3, 'DMPC', 'Proposed GWO-DMPC', 'P_{max}');
% set(lgd, 'Position', [0.22, 0.15, 0.1, 0.05]); % [left, bottom, width, height]
% lgd.FontSize = 6;
% %title('Fuel Cell Duty Cycle');
% % text(ax3_3, 0.5, -0.15, '(c). Fuel Cell Duty Cycle', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% text(ax3_3, 0.5, -0.20, '(c)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% grid on; xlim([0, 0.8]); ylim([-0.1 1.1]);

% % Battery Duty Cycle
% ax3_4 = subplot(2, 2, 4);
% stairs(t, results_dempc.U(:,4), 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.U(:,4), 'r--', 'LineWidth', 1.5);
% xlabel('Time (s)');
% ylabel('d_b');
% lgd = legend(ax3_4, 'DMPC', 'Proposed GWO-DMPC', 'P_{max}');
% set(lgd, 'Position', [0.62, 0.115, 0.1, 0.04]); % [left, bottom, width, height]
% lgd.FontSize = 6;
% %title('Battery Duty Cycle');
% % text(ax3_4, 0.5, -0.15, '(d). Battery Duty Cycle', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% text(ax3_4, 0.5, -0.20, '(d)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold')
% grid on; xlim([0, 0.8]); ylim([-0.1 1.1]);

% % --- Zoom on d_s ---
% xlim_ds = [0.19, 0.23];
% idx_ds = (t >= xlim_ds(1) & t <= xlim_ds(2));
% ymin_ds = min([min(results_dempc.U(idx_ds,1)), min(results_gwo.U(idx_ds,1))]);
% ymax_ds = max([max(results_dempc.U(idx_ds,1)), max(results_gwo.U(idx_ds,1))]);
% margin_ds = max(0.05, (ymax_ds - ymin_ds) * 0.1);

% rectangle(ax3_1, 'Position', [xlim_ds(1), ymin_ds-margin_ds, xlim_ds(2)-xlim_ds(1), (ymax_ds+margin_ds)-(ymin_ds-margin_ds)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos1 = get(ax3_1, 'Position');
% ax_ins3_1 = axes('Position', [pos1(1)+pos1(3)*0.65, pos1(2)+pos1(4)*0.55, pos1(3)*0.3, pos1(4)*0.35]);
% box on; grid on; hold on;
% stairs(ax_ins3_1, t, results_dempc.U(:,1), 'b-', 'LineWidth', 1.5);
% plot(ax_ins3_1, t, results_gwo.U(:,1), 'r--', 'LineWidth', 1.5);
% xlim(ax_ins3_1, xlim_ds);
% ylim(ax_ins3_1, [ymin_ds-margin_ds, ymax_ds+margin_ds]);
% %title(ax_ins3_1, 'Zoom: 0.19 - 0.23 s', 'FontSize', 8);

% % --- Zoom on d_ae ---
% xlim_dae = [0.19, 0.23];
% idx_dae = (t >= xlim_dae(1) & t <= xlim_dae(2));
% ymin_dae = min([min(results_dempc.U(idx_dae,2)), min(results_gwo.U(idx_dae,2))]);
% ymax_dae = max([max(results_dempc.U(idx_dae,2)), max(results_gwo.U(idx_dae,2))]);
% margin_dae = max(0.05, (ymax_dae - ymin_dae) * 0.1);

% rectangle(ax3_2, 'Position', [xlim_dae(1), ymin_dae-margin_dae, xlim_dae(2)-xlim_dae(1), (ymax_dae+margin_dae)-(ymin_dae-margin_dae)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos2 = get(ax3_2, 'Position');
% ax_ins3_2 = axes('Position', [pos2(1)+pos2(3)*0.65, pos2(2)+pos2(4)*0.55, pos2(3)*0.3, pos2(4)*0.35]);
% box on; grid on; hold on;
% stairs(ax_ins3_2, t, results_dempc.U(:,2), 'b-', 'LineWidth', 1.5);
% plot(ax_ins3_2, t, results_gwo.U(:,2), 'r--', 'LineWidth', 1.5);
% xlim(ax_ins3_2, xlim_dae);
% ylim(ax_ins3_2, [ymin_dae-margin_dae, ymax_dae+margin_dae]);
% %title(ax_ins3_2, 'Zoom: 0.19 - 0.23 s', 'FontSize', 8);

% % --- Zoom on d_pe ---
% xlim_dpe = [0.58, 0.65];
% idx_dpe = (t >= xlim_dpe(1) & t <= xlim_dpe(2));
% ymin_dpe = min([min(results_dempc.U(idx_dpe,3)), min(results_gwo.U(idx_dpe,3))]);
% ymax_dpe = max([max(results_dempc.U(idx_dpe,3)), max(results_gwo.U(idx_dpe,3))]);
% margin_dpe = max(0.05, (ymax_dpe - ymin_dpe) * 0.1);

% rectangle(ax3_3, 'Position', [xlim_dpe(1), ymin_dpe-margin_dpe, xlim_dpe(2)-xlim_dpe(1), (ymax_dpe+margin_dpe)-(ymin_dpe-margin_dpe)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos3 = get(ax3_3, 'Position');
% ax_ins3_3 = axes('Position', [pos3(1)+pos3(3)*0.1, pos3(2)+pos3(4)*0.55, pos3(3)*0.3, pos3(4)*0.35]);
% box on; grid on; hold on;
% stairs(ax_ins3_3, t, results_dempc.U(:,3), 'b-', 'LineWidth', 1.5);
% plot(ax_ins3_3, t, results_gwo.U(:,3), 'r--', 'LineWidth', 1.5);
% xlim(ax_ins3_3, xlim_dpe);
% ylim(ax_ins3_3, [ymin_dpe-margin_dpe, ymax_dpe+margin_dpe]);
% %title(ax_ins3_3, 'Zoom: 0.58 - 0.65 s', 'FontSize', 8);

% % --- Zoom on d_b ---
% xlim_db = [0.58, 0.65];
% idx_db = (t >= xlim_db(1) & t <= xlim_db(2));
% ymin_db = min([min(results_dempc.U(idx_db,4)), min(results_gwo.U(idx_db,4))]);
% ymax_db = max([max(results_dempc.U(idx_db,4)), max(results_gwo.U(idx_db,4))]);
% margin_db = max(0.05, (ymax_db - ymin_db) * 0.1);

% rectangle(ax3_4, 'Position', [xlim_db(1), ymin_db-margin_db, xlim_db(2)-xlim_db(1), (ymax_db+margin_db)-(ymin_db-margin_db)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos4 = get(ax3_4, 'Position');
% ax_ins3_4 = axes('Position', [pos4(1)+pos4(3)*0.1, pos4(2)+pos4(4)*0.55, pos4(3)*0.3, pos4(4)*0.35]);
% box on; grid on; hold on;
% stairs(ax_ins3_4, t, results_dempc.U(:,4), 'b-', 'LineWidth', 1.5);
% plot(ax_ins3_4, t, results_gwo.U(:,4), 'r--', 'LineWidth', 1.5);
% xlim(ax_ins3_4, xlim_db);
% ylim(ax_ins3_4, [ymin_db-margin_db, ymax_db+margin_db]);
% %title(ax_ins3_4, 'Zoom: 0.58 - 0.65 s', 'FontSize', 8);


% % =========================================================================
% % FIGURE 4: Power Dispatch Comparison
% % =========================================================================
% f4 = figure('Name', 'Figure 4: Power Dispatch Comparison', 'Position', [250, 100, 800, 800], 'Color', 'w');

% % PV Power
% ax4_1 = subplot(2, 2, 1);
% plot(t, results_dempc.Ppv / 1000, 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.Ppv / 1000, 'r--', 'LineWidth', 1.5);
% plot(t, Pmax_t, 'k:', 'LineWidth', 1.5);
% ylabel('P_{pv} (kW)');
% % lgd = legend('DMPC', 'Proposed GWO-DMPC', 'P_{max}', 'Location', 'NorthWest');
% % lgd.FontSize = 5;
% % lgd.ItemTokenSize = [8, 8];
% % legend boxoff;
% lgd = legend(ax4_1, 'DMPC', 'Proposed GWO-DMPC', 'P_{max}');
% set(lgd, 'Position', [0.15, 0.85, 0.1, 0.05]); % [left, bottom, width, height]
% lgd.FontSize = 5;
% lgd.ItemTokenSize = [8, 8];
% %title('PV Power and MPPT');
% % text(ax4_1, 0.5, -0.15, '(a). PV Power and MPPT', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% text(ax4_1, 0.5, -0.15, '(a)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% grid on; xlim([0, 0.8]);

% % --- Zoom on P_pv ---
% xlim_ppv = [0.19, 0.23];
% idx_ppv = (t >= xlim_ppv(1) & t <= xlim_ppv(2));
% ymin_pv = min([min(results_dempc.Ppv(idx_ppv))/1000, min(results_gwo.Ppv(idx_ppv))/1000, min(Pmax_t(idx_ppv))]);
% ymax_pv = max([max(results_dempc.Ppv(idx_ppv))/1000, max(results_gwo.Ppv(idx_ppv))/1000, max(Pmax_t(idx_ppv))]);
% margin_pv = max(0.5, (ymax_pv - ymin_pv) * 0.1);

% rectangle(ax4_1, 'Position', [xlim_ppv(1), ymin_pv-margin_pv, xlim_ppv(2)-xlim_ppv(1), (ymax_pv+margin_pv)-(ymin_pv-margin_pv)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos_ppv = get(ax4_1, 'Position');
% ax_ins4_1 = axes('Position', [pos_ppv(1)+pos_ppv(3)*0.35, pos_ppv(2)+pos_ppv(4)*0.05, pos_ppv(3)*0.32, pos_ppv(4)*0.28]);
% box on; grid on; hold on;
% plot(ax_ins4_1, t, results_dempc.Ppv / 1000, 'b-', 'LineWidth', 1.5);
% plot(ax_ins4_1, t, results_gwo.Ppv / 1000, 'r--', 'LineWidth', 1.5);
% plot(ax_ins4_1, t, Pmax_t, 'k:', 'LineWidth', 1.5);
% xlim(ax_ins4_1, xlim_ppv);
% ylim(ax_ins4_1, [ymin_pv - margin_pv, ymax_pv + margin_pv]);
% %title(ax_ins4_1, 'Zoom: 0.19 - 0.23 s', 'FontSize', 8);

% % AE Power
% ax4_2 = subplot(2, 2, 2);
% plot(t, results_dempc.Pae / 1000, 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.Pae / 1000, 'r--', 'LineWidth', 1.5);
% ylabel('P_{ae} (kW)');
% lgd = legend('DMPC', 'Proposed GWO-DMPC', 'P_{max}', 'Location', 'NorthWest');
% lgd.FontSize = 6;
% lgd.ItemTokenSize = [10, 8];
% % legend boxoff;
% %title('Electrolyzer Power');
% % text(ax4_2, 0.5, -0.15, '(b). Electrolyzer Power', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% text(ax4_2, 0.5, -0.15, '(b)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% grid on; xlim([0, 0.8]);

% % --- Zoom on P_ae ---
% xlim_pae = [0.19, 0.21];
% idx_pae = (t >= xlim_pae(1) & t <= xlim_pae(2));
% ymin_pae = min([min(results_dempc.Pae(idx_pae)), min(results_gwo.Pae(idx_pae))]) / 1000;
% ymax_pae = max([max(results_dempc.Pae(idx_pae)), max(results_gwo.Pae(idx_pae))]) / 1000;
% margin_pae = max(0.5, (ymax_pae - ymin_pae) * 0.1);

% rectangle(ax4_2, 'Position', [xlim_pae(1), ymin_pae-margin_pae, xlim_pae(2)-xlim_pae(1), (ymax_pae+margin_pae)-(ymin_pae-margin_pae)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos_pae = get(ax4_2, 'Position');
% ax_ins4_2 = axes('Position', [pos_pae(1)+pos_pae(3)*0.55, pos_pae(2)+pos_pae(4)*0.55, pos_pae(3)*0.4, pos_pae(4)*0.4]);
% box on; grid on; hold on;
% plot(ax_ins4_2, t, results_dempc.Pae / 1000, 'b-', 'LineWidth', 1.5);
% plot(ax_ins4_2, t, results_gwo.Pae / 1000, 'r--', 'LineWidth', 1.5);
% xlim(ax_ins4_2, xlim_pae);
% ylim(ax_ins4_2, [ymin_pae - margin_pae, ymax_pae + margin_pae]);
% %title(ax_ins4_2, 'Zoom: 0.19 - 0.23 s', 'FontSize', 8);

% % FC Power
% ax4_3 = subplot(2, 2, 3);
% plot(t, results_dempc.Ppe / 1000, 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.Ppe / 1000, 'r--', 'LineWidth', 1.5);
% xlabel('Time (s)');
% ylabel('P_{pe} (kW)');
% % lgd = legend('DMPC', 'Proposed GWO-DMPC', 'P_{max}', 'Location', 'SouthWest');
% % lgd.FontSize = 5;
% % lgd.ItemTokenSize = [10, 8];
% % legend boxoff;
% lgd = legend(ax4_3, 'DMPC', 'Proposed GWO-DMPC');
% set(lgd, 'Position', [0.15, 0.20, 0.1, 0.03]); % [left, bottom, width, height]
% lgd.FontSize = 5;
% lgd.ItemTokenSize = [8, 8];
% %title('Fuel Cell Power');
% % text(ax4_3, 0.5, -0.15, '(c). Fuel Cell Power', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% text(ax4_3, 0.5, -0.20, '(c)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% grid on; xlim([0, 0.8]);

% % --- Zoom on P_fc ---
% xlim_pfc = [0.58, 0.65];
% idx_pfc = (t >= xlim_pfc(1) & t <= xlim_pfc(2));
% ymin_p = min([min(results_dempc.Ppe(idx_pfc)), min(results_gwo.Ppe(idx_pfc))]) / 1000;
% ymax_p = max([max(results_dempc.Ppe(idx_pfc)), max(results_gwo.Ppe(idx_pfc))]) / 1000;
% margin_p = max(0.5, (ymax_p - ymin_p) * 0.1);

% rectangle(ax4_3, 'Position', [xlim_pfc(1), ymin_p-margin_p, xlim_pfc(2)-xlim_pfc(1), (ymax_p+margin_p)-(ymin_p-margin_p)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos_pfc = get(ax4_3, 'Position');
% ax_ins4_3 = axes('Position', [pos_pfc(1)+pos_pfc(3)*0.1, pos_pfc(2)+pos_pfc(4)*0.55, pos_pfc(3)*0.4, pos_pfc(4)*0.4]);
% box on; grid on; hold on;
% plot(ax_ins4_3, t, results_dempc.Ppe / 1000, 'b-', 'LineWidth', 1.5);
% plot(ax_ins4_3, t, results_gwo.Ppe / 1000, 'r--', 'LineWidth', 1.5);
% xlim(ax_ins4_3, xlim_pfc);
% ylim(ax_ins4_3, [ymin_p - margin_p, ymax_p + margin_p]);
% %title(ax_ins4_3, 'Zoom: 0.58 - 0.65 s', 'FontSize', 8);

% % Battery Power
% ax4_4 = subplot(2, 2, 4);
% plot(t, results_dempc.Pbat / 1000, 'b-', 'LineWidth', 1.5); hold on;
% plot(t, results_gwo.Pbat / 1000, 'r--', 'LineWidth', 1.5);
% xlabel('Time (s)');
% ylabel('P_{bat} (kW)');
% % lgd = legend('DMPC', 'Proposed GWO-DMPC', 'P_{max}', 'Location', 'SouthWest');
% % lgd.FontSize = 5;
% % lgd.ItemTokenSize = [10, 8];
% % legend boxoff;
% lgd4 = legend(ax4_4, 'DMPC', 'Proposed GWO-DMPC');
% set(lgd4, 'Position', [0.59, 0.2, 0.1, 0.03]); % [left, bottom, width, height]
% lgd4.FontSize = 5;
% lgd4.ItemTokenSize = [8, 8];
% %title('Battery Power');
% % text(ax4_4, 0.5, -0.15, '(d). Battery Power', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% text(ax4_4, 0.5, -0.20, '(d)', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
% grid on; xlim([0, 0.8]);

% % --- Zoom on P_bat ---
% xlim_pbat = [0.58, 0.65];
% idx_pbat = (t >= xlim_pbat(1) & t <= xlim_pbat(2));
% ymin_p = min([min(results_dempc.Pbat(idx_pbat)), min(results_gwo.Pbat(idx_pbat))]) / 1000;
% ymax_p = max([max(results_dempc.Pbat(idx_pbat)), max(results_gwo.Pbat(idx_pbat))]) / 1000;
% margin_p = max(0.5, (ymax_p - ymin_p) * 0.1);

% rectangle(ax4_4, 'Position', [xlim_pbat(1), ymin_p-margin_p, xlim_pbat(2)-xlim_pbat(1), (ymax_p+margin_p)-(ymin_p-margin_p)], 'EdgeColor', 'k', 'LineStyle', '-.');

% pos_pbat = get(ax4_4, 'Position');
% ax_ins4_4 = axes('Position', [pos_pbat(1)+pos_pbat(3)*0.1, pos_pbat(2)+pos_pbat(4)*0.55, pos_pbat(3)*0.4, pos_pbat(4)*0.4]);
% box on; grid on; hold on;
% plot(ax_ins4_4, t, results_dempc.Pbat / 1000, 'b-', 'LineWidth', 1.5);
% plot(ax_ins4_4, t, results_gwo.Pbat / 1000, 'r--', 'LineWidth', 1.5);
% xlim(ax_ins4_4, xlim_pbat);
% ylim(ax_ins4_4, [ymin_p - margin_p, ymax_p + margin_p]);
% %title(ax_ins4_4, 'Zoom: 0.58 - 0.65 s', 'FontSize', 8);


% % =========================================================================
% % FIGURE 5: Battery SOC and Hydrogen Tank Pressure
% % =========================================================================
% f5 = figure('Name', 'Figure 5: Energy Storage States', 'Position', [300, 200, 800, 500], 'Color', 'w');

% % Battery SOC
% subplot(2, 1, 1);
% plot(t, results_dempc.SOC * 100, 'b-', 'LineWidth', 2); hold on;
% plot(t, results_gwo.SOC * 100, 'r--', 'LineWidth', 2);
% yline(90, 'k--', 'Upper Limit (90%)', 'LabelHorizontalAlignment', 'left', 'LineWidth', 1.2);
% yline(20, 'k--', 'Lower Limit (20%)', 'LabelHorizontalAlignment', 'left', 'LineWidth', 1.2);
% xline(stages, 'k:');
% ylabel('SOC (%)');
% %title('Battery State of Charge');
% legend('DMPC', 'Proposed GWO-DMPC', 'Location', 'best');
% grid on; xlim([0, 0.8]);

% % Hydrogen Tank Pressure
% subplot(2, 1, 2);
% plot(t, results_dempc.Ptank / 1e5, 'b-', 'LineWidth', 2); hold on;
% plot(t, results_gwo.Ptank / 1e5, 'r--', 'LineWidth', 2);
% xline(stages, 'k:');
% xlabel('Time (s)');
% ylabel('Pressure (bar)');
% %title('Hydrogen Tank Pressure');
% grid on; xlim([0, 0.8]);


% =========================================================================
% Export Figures
% =========================================================================
% fprintf('Saving figures to PDF (500 DPI)...\n');
% exportgraphics(f1, 'Figure_1_Scenario_EMS.jpeg', 'ContentType', 'vector', 'Resolution', 500);
% exportgraphics(f2, 'Figure_2_DC_Bus_Voltage.jpeg', 'ContentType', 'vector', 'Resolution', 500);
% exportgraphics(f3, 'Figure_3_Duty_Cycles.jpeg', 'ContentType', 'vector', 'Resolution', 500);
% exportgraphics(f4, 'Figure_4_Power_Dispatch.jpeg', 'ContentType', 'vector', 'Resolution', 500);
% exportgraphics(f5, 'Figure_5_Energy_Storage.jpeg', 'ContentType', 'vector', 'Resolution', 500);


% =========================================================================
% Console Notification
% =========================================================================
% fprintf('All visualizations generated successfully.\n');
% fprintf(' - Figure 1: Scenario & EMS\n');
% fprintf(' - Figure 2: DC Bus Voltage Regulation\n');
% fprintf(' - Figure 3: Duty Cycle Trajectories\n');
% fprintf(' - Figure 4: Power Dispatch\n');
% fprintf(' - Figure 5: Energy Storage States\n');