%% Demonstrating reading the output files
output_csv = 'SINGLE-PERSON\VID_20220830_172442.csv';
%VID_20220830_172128.csv';
%VID_20220830_171920.csv';
%VID_20220830_161618.csv';
%VID_20220830_171739.csv';
%

% First read in the column names, to know which columns to read for
% particular features
tab = readtable(output_csv);
column_names = tab.Properties.VariableNames;

% Read all of the data
all_params  = dlmread(output_csv, ',', 1, 0);

% This indicates which frames were succesfully tracked

% Find which column contains success of tracking data and timestamp data
valid_ind = cellfun(@(x) ~isempty(x) && x==1, strfind(column_names, 'success'));
time_stamp_ind = cellfun(@(x) ~isempty(x) && x==1, strfind(column_names, 'timestamp'));

% Extract tracking success data and only read those frame
valid_frames = logical(all_params(:,valid_ind));

% Get the timestamp data
time_stamps = all_params(valid_frames, time_stamp_ind);

% Recupero i face_id
face_id_ind = cellfun(@(x) ~isempty(x) && x==1, strfind(column_names, 'face_id'));
face_ids = all_params(valid_frames, face_id_ind);

%% Demo gaze
gaze_inds = cellfun(@(x) ~isempty(x) && x==1, strfind(column_names, 'gaze_angle'));

% Read gaze (x,y,z) for one eye and (x,y,z) for another
gaze = all_params(valid_frames, gaze_inds);

plot(time_stamps, gaze(:,1), 'DisplayName', 'Left - right');
hold on;
plot(time_stamps, gaze(:,2), 'DisplayName', 'Up - down');
xlabel('Time(s)') % x-axis label
ylabel('Angle radians') % y-axis label
legend('show');
hold off;

%% Gaze per ogni persona
gaze_inds = cellfun(@(x) ~isempty(x) && x==1, strfind(column_names, 'gaze_angle'));

% Read gaze (x,y,z) for one eye and (x,y,z) for another
gaze = all_params(valid_frames, gaze_inds);

% Concateno face_ids con gaze
M = [time_stamps face_ids gaze];

% Filtro per face_id
[~,~,X] = unique(M(:,2));
C = accumarray(X,1:size(M,1),[],@(r){M(r,:)});

figure
for i=1:length(C)
    plot(C{i,1}(:,1), C{i,1}(:,3), 'DisplayName', sprintf('%d Left - right', i-1));
    hold on;
    plot(C{i,1}(:,1), C{i,1}(:,4), 'DisplayName', sprintf('%d Up - down', i-1));
    hold on;
end
xlabel('Time(s)') % x-axis label
ylabel('Angle radians') % y-axis label
legend('show');
hold off;
