import logging
import multiprocessing
import os
import pickle
from pathlib import Path
import tqdm

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    import tensorflow as tf

except ImportError as e:
    logger.info(e)

from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from domain_expansion.utils.data_utils import interpolate_track, create_highd_lanelet_network
from collections import defaultdict
from domain_expansion.dataset.data_process import *
from commonroad.common.file_reader import CommonRoadFileReader
from domain_expansion.utils import data_utils
from scenarionet.converter.waymo.waymo_protos import scenario_pb2

SPLIT_KEY = "|"


def get_scenarios(data_directory, start_index, num):
    logger.info("\nReading raw data")
    pattern = "*_tracks.csv"
    file_list = list(Path(data_directory).glob(pattern))
    if num is None:
        logger.warning(
            "You haven't specified the number of raw files! It is set to {} now.".format(len(file_list) - start_index)
        )
        num = len(file_list) - start_index
    assert len(file_list) >= start_index + num and start_index >= 0, \
        "No sufficient files ({}) in raw_data_directory. need: {}, start: {}".format(len(file_list), num, start_index)
    file_list = file_list[start_index:start_index + num]
    num_files = len(file_list)
    all_result = [os.path.join(data_directory, f) for f in file_list]
    logger.info("\nFind {} files".format(num_files))
    return all_result

def get_scenarios_sinD(data_directory, start_index, num):
    logger.info("\nReading raw data")
    file_list = list(Path(data_directory).iterdir())
    if num is None:
        logger.warning(
            "You haven't specified the number of raw files! It is set to {} now.".format(len(file_list) - start_index)
        )
        num = len(file_list) - start_index
    assert len(file_list) >= start_index + num and start_index >= 0, \
        "No sufficient files ({}) in raw_data_directory. need: {}, start: {}".format(len(file_list), num, start_index)
    file_list = file_list[start_index:start_index + num]
    num_files = len(file_list)
    all_result = [os.path.join(data_directory, f) for f in file_list]
    logger.info("\nFind {} files".format(num_files))
    return all_result

def preprocess_rounD_scenarios(files, worker_index):
    """
    Convert the rounD files into scenarios. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenarios
    """
    delta_frames = 5
    history_length = 105
    future_length = 300
    lane_discretization = 0.5
    map_file = '/home/boqi/CoDriving/data/rounD/rounD_location2.lanelet.xml'
    preprocessor = DataPreprocess(history_length, future_length, delta_frames, lane_discretization, num_points_each_polyline=21)
    lanelet_network = CommonRoadFileReader(map_file).open_lanelet_network()
    for file_path in tqdm.tqdm(files, leave=False, position=0, desc="Worker {} Number of raw file".format(worker_index)):
        file_path = Path(file_path)
        filename = file_path.name
        logger.info(f"Worker {worker_index} is reading raw file: {filename}")
        scenario_id = filename.split('_')[-2]
        static_filename = file_path.parent / (scenario_id + "_tracksMeta.csv")
        meta_filename = file_path.parent / (scenario_id + "_recordingMeta.csv")
        if ("tracks.csv" not in filename):
            logger.info(f"Worker {worker_index} skip this file: {filename}")
            continue
        static_info = pd.read_csv(static_filename).to_dict(orient="records")
        static_info = {x['trackId']:x for x in static_info}
        raw_tracks = pd.read_csv(file_path).groupby("trackId", sort=False)
        tracks = {}
        
        for track_id, track_rows in raw_tracks:
            track = track_rows.to_dict(orient="list")
            for key, value in track.items():
                if key in ["trackId", "recordingId"]:
                    track[key] = value[0]
                else:
                    track[key] = np.array(value)
            track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
            tracks[track_id] = track
        
        center_vehids, center_vehids_start_frame = [], []
        for track_id in tracks:
            tracks[track_id] = interpolate_track(tracks[track_id])
            vehid = tracks[track_id]['trackId']
            veh_type = static_info[track_id]['class']
            if veh_type != 'bus' and veh_type != 'trailer' and veh_type != 'pedestrian' and veh_type != 'bicycle' and veh_type != 'motorcycle' and len(tracks[track_id]['frame']) > history_length+future_length:
                center_vehids.append(vehid)
                center_vehids_start_frame.append(tracks[track_id]['frame'][0])
        all_scenarios = defaultdict(lambda:dict())
        for track in tracks.values():
            vehid = track['trackId']
            veh_length = static_info[vehid]['length']
            veh_width = static_info[vehid]['width']
            veh_type = static_info[vehid]['class']
            
            for frame_ind in range(len(track['frame'])):
                position_x = track['xCenter'][frame_ind]
                position_y = track['yCenter'][frame_ind]
                orientation = np.deg2rad(track['heading'][frame_ind])
                velocity_x = track['xVelocity'][frame_ind]
                velocity_y = track['yVelocity'][frame_ind]
                all_scenarios[track['frame'][frame_ind]][vehid] = [position_x, position_y, orientation, velocity_x, velocity_y, veh_length, veh_width, veh_type]
        
        # sort the vehids by start frame
        center_vehids = [x for _, x in sorted(zip(center_vehids_start_frame, center_vehids))]
        current_frame = -1
        for vehid in center_vehids:
            start_frame = max(static_info[vehid]['initialFrame']*2+history_length, current_frame+1)
            if start_frame > static_info[vehid]['finalFrame']*2 - future_length:
                continue
            for t0 in range(start_frame, static_info[vehid]['finalFrame']*2 - future_length, history_length):
                current_frame = t0
                nei_ids = [nei_id for nei_id in all_scenarios[t0-delta_frames].keys() if nei_id != vehid]
                nei_ids = [nei_id for nei_id in nei_ids if np.linalg.norm(np.array(all_scenarios[t0-delta_frames][vehid][:2])-np.array(all_scenarios[t0-delta_frames][nei_id][:2])) < 100]
                scenario_id, agent_inputs, agent_outputs, map_inputs = preprocessor.process_data(t0, {'scenario_id':scenario_id, 'sdc_id':vehid, 'nei_ids':nei_ids, 'tracks':tracks, 'scenes':all_scenarios, 'map_features':lanelet_network})
                scenario = dict()
                scenario['scenario_id'] = scenario_id + "_" + str(t0) + SPLIT_KEY + str(file_path)
                scenario['tracks'] = np.concatenate([agent_inputs['obj_trajs'], agent_outputs['center_gt_trajs']], axis=1)
                scenario['tracks_valid'] = np.concatenate([agent_inputs['obj_trajs_mask'], agent_outputs['center_gt_trajs_mask']], axis=1)
                scenario['sdc_track_index'] = agent_inputs['sdc_track_index'][0]
                scenario['obj_types'] = agent_inputs['obj_types'][:, -1]
                scenario['obj_ids'] = agent_inputs['obj_ids'][:, -1]
                scenario['map_inputs'] = lanelet_network
                
                yield scenario

    logger.info(f"Worker {worker_index} finished read {len(files)} files.")
    # logger.info("Worker {}: Process {} waymo scenarios".format(worker_index, len(scenarios)))
    # return scenarios

def preprocess_sinD_scenarios(folders, worker_index):
    """
    Convert the sinD files into scenarios. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenarios
    """
    delta_frames = 1
    history_length = 21
    future_length = 60
    lane_discretization = 0.5
    map_file = '/home/boqi/CoDriving/data/SinD/lanelets.xml'
    preprocessor = DataPreprocess(history_length, future_length, delta_frames, lane_discretization, num_points_each_polyline=21)
    lanelet_network = CommonRoadFileReader(map_file).open_lanelet_network()
    for subfolder_file in tqdm.tqdm(folders, leave=False, position=0, desc="Worker {} Number of raw file".format(worker_index)):
        subfolder_file = Path(subfolder_file)
        scenario_id = str(subfolder_file).split('/')[-1]
        logger.info(f"Worker {worker_index} is reading raw folder: {subfolder_file}")
        
        veh_static_filename = subfolder_file /  'Veh_tracks_meta.csv'
        veh_track_filename = subfolder_file /  'Veh_smoothed_tracks.csv'

        ped_static_filename = subfolder_file /  'Ped_tracks_meta.csv'
        ped_track_filename = subfolder_file /  'Ped_smoothed_tracks.csv'
        
        trafficlight_filename = subfolder_file /  'TrafficLight_modified.csv'
        meta_filename = subfolder_file /  'recoding_metas.csv'
        
        veh_static_info = pd.read_csv(veh_static_filename).to_dict(orient="records")
        veh_static_info = {x['trackId']:x for x in veh_static_info}
        veh_raw_tracks = pd.read_csv(veh_track_filename).groupby("track_id", sort=False)

        ped_static_info = pd.read_csv(ped_static_filename).to_dict(orient="records")
        ped_static_info = {x['trackId']:x for x in ped_static_info}
        ped_raw_tracks = pd.read_csv(ped_track_filename).groupby("track_id", sort=False)
        
        tracks = {}
        max_track_id = 0
        for track_id, track_rows in veh_raw_tracks:
            track = track_rows.to_dict(orient="list")
            for key, value in track.items():
                if key in ["track_id", "agent_type", "length", "width"]:
                    track[key] = value[0]
                else:
                    track[key] = np.array(value)
            track["center"] = np.stack([track["x"], track["y"]], axis=-1)
            track["frame"] = track["frame_id"]
            track["initialFrame"] = track["frame_id"][0]
            track["finalFrame"] = track["frame_id"][-1]
            tracks[track_id] = track
            if track_id > max_track_id:
                max_track_id = track_id+1
        p = 0
        for track_id, track_rows in ped_raw_tracks:
            track = track_rows.to_dict(orient="list")
            new_track_id = max_track_id + p
            for key, value in track.items():
                if key in ["agent_type"]:
                    track[key] = 'pedestrian' # actually there are animals but...
                elif key == "track_id":
                    track[key] = new_track_id
                else:
                    track[key] = np.array(value)
            track["center"] = np.stack([track["x"], track["y"]], axis=-1)
            track["frame"] = track["frame_id"]
            track["initialFrame"] = track["frame"][0]
            track["finalFrame"] = track["frame"][-1]
            tracks[new_track_id] = track
            p += 1

        center_vehids, center_vehids_start_frame = [], []
        all_scenarios = defaultdict(lambda:dict())
        for track in tracks.values():
            vehid = track['track_id']
            veh_type = track['agent_type']
            if veh_type == 'pedestrian':
                veh_length = 0.5
                veh_width = 0.5
            else:
                veh_length = track['length']
                veh_width = track['width']
            disp = np.linalg.norm(track['center'][0] - track['center'][-1])
            if veh_type != 'trailer' and veh_type != 'pedestrian' and veh_type != 'bicycle' and veh_type != 'motorcycle' and len(track['frame']) > history_length and disp > 1.0:
                center_vehids.append(vehid)
                center_vehids_start_frame.append(track['frame'][0])

            for frame_ind in range(len(track['frame'])):
                position_x = track['x'][frame_ind]
                position_y = track['y'][frame_ind]
                if veh_type == 'pedestrian':
                    orientation = np.arctan2(track['vy'][frame_ind], track['vx'][frame_ind])
                else:
                    orientation = track['heading_rad'][frame_ind]
                velocity_x = track['vx'][frame_ind]
                velocity_y = track['vy'][frame_ind]
                all_scenarios[track['frame'][frame_ind]][vehid] = [position_x, position_y, orientation, velocity_x, velocity_y, veh_length, veh_width, veh_type]
        # sort the vehids by start frame
        center_vehids = [x for _, x in sorted(zip(center_vehids_start_frame, center_vehids))]
        current_frame = -1
    
        for vehid in center_vehids:
            # start_frame = max(static_info[vehid]['initialFrame']*2+history_length, current_frame+1)
            start_frame = tracks[vehid]['initialFrame']+history_length
            if start_frame > tracks[vehid]['finalFrame'] - future_length:
                continue
            for t0 in range(start_frame, tracks[vehid]['finalFrame'] - future_length, history_length):
                current_frame = t0
                nei_ids = [nei_id for nei_id in all_scenarios[t0-delta_frames].keys() if nei_id != vehid]
                nei_ids = [nei_id for nei_id in nei_ids if np.linalg.norm(np.array(all_scenarios[t0-delta_frames][vehid][:2])-np.array(all_scenarios[t0-delta_frames][nei_id][:2])) < 100]
                scenario_id, agent_inputs, agent_outputs, map_inputs = preprocessor.process_data(t0, {'scenario_id':scenario_id, 'sdc_id':vehid, 'nei_ids':nei_ids, 'tracks':tracks, 'scenes':all_scenarios, 'map_features':lanelet_network})
                vel_norm = np.linalg.norm(agent_inputs['obj_trajs'][:, :, 4:6], axis=-1)
                vel_norm = vel_norm[~np.isnan(vel_norm)]
                if np.mean(vel_norm) < 1.0:
                    continue
                scenario = dict()
                scenario['scenario_id'] = scenario_id + "_" + str(t0) + "_" + str(vehid) + SPLIT_KEY + str(subfolder_file)
                scenario['tracks'] = np.concatenate([agent_inputs['obj_trajs'], agent_outputs['center_gt_trajs']], axis=1)
                scenario['tracks_valid'] = np.concatenate([agent_inputs['obj_trajs_mask'], agent_outputs['center_gt_trajs_mask']], axis=1)
                scenario['sdc_track_index'] = agent_inputs['sdc_track_index'][0]
                scenario['obj_types'] = agent_inputs['obj_types'][:, -1]
                scenario['obj_ids'] = agent_inputs['obj_ids'][:, -1]
                scenario['map_inputs'] = lanelet_network
                
                yield scenario

def preprocess_inD_scenarios(files, worker_index):
    """
    Convert the inD files into scenarios. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenarios
    """
    delta_frames = 5
    history_length = 105
    future_length = 300
    lane_discretization = 0.5
    map_file_dir = Path('/home/boqi/CoDriving/data/inD/repaired_maps/')
    preprocessor = DataPreprocess(history_length, future_length, delta_frames, lane_discretization, num_points_each_polyline=21)
    map_locations = {1: "Bendplatz", 2: "frankenberg", 3: "heckstrasse", 4: "aseag"}
    class_to_obstacleType = {"car": "car", "truck_bus": "bus", "pedestrian": "pedestrian", "bicycle": "bicycle"}
    
    for file_path in tqdm.tqdm(files, leave=False, position=0, desc="Worker {} Number of raw file".format(worker_index)):
        file_path = Path(file_path)
        filename = file_path.name
        logger.info(f"Worker {worker_index} is reading raw file: {filename}")
        scenario_id = filename.split('_')[-2]
        static_filename = file_path.parent / (scenario_id + "_tracksMeta.csv")
        meta_filename = file_path.parent / (scenario_id + "_recordingMeta.csv")
        meta_info = pd.read_csv(meta_filename)
        if ("tracks.csv" not in filename):
            logger.info(f"Worker {worker_index} skip this file: {filename}")
            continue
        scenario_id = f"{map_locations[meta_info['locationId'].values[0]]}_" + scenario_id
        map_file_path = map_file_dir / f"{map_locations[meta_info['locationId'].values[0]]}.xml"
        lanelet_network = CommonRoadFileReader(str(map_file_path)).open_lanelet_network()
        static_info = pd.read_csv(static_filename).to_dict(orient="records")
        raw_tracks = pd.read_csv(file_path).groupby("trackId", sort=False)
        tracks = {}
        
        for track_id, track_rows in raw_tracks:
            track = track_rows.to_dict(orient="list")
            for key, value in track.items():
                if key in ["trackId", "recordingId"]:
                    track[key] = value[0]
                else:
                    track[key] = np.array(value)
            track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
            tracks[track_id] = track
        
        center_vehids, center_vehids_start_frame = [], []
        for track_id in tracks:
            tracks[track_id] = interpolate_track(tracks[track_id])
            vehid = tracks[track_id]['trackId']
            veh_type = class_to_obstacleType[static_info[vehid]['class']]
            velocities = np.sqrt(np.square(tracks[track_id]['xVelocity']) + np.square(tracks[track_id]['yVelocity']))
            mean_velocity = np.mean(velocities)
            if veh_type != 'trailer' and veh_type != 'pedestrian' and veh_type != 'bicycle' and veh_type != 'motorcycle' and len(tracks[track_id]['frame']) > history_length and mean_velocity > 1.0:
                center_vehids.append(vehid)
                center_vehids_start_frame.append(tracks[track_id]['frame'][0])
        all_scenarios = defaultdict(lambda:dict())
        for track in tracks.values():
            vehid = track['trackId']
            veh_length = static_info[vehid]['length']
            veh_width = static_info[vehid]['width']
            veh_type = class_to_obstacleType[static_info[vehid]['class']]
            
            for frame_ind in range(len(track['frame'])):
                position_x = track['xCenter'][frame_ind]
                position_y = track['yCenter'][frame_ind]
                orientation = np.deg2rad(track['heading'][frame_ind])
                velocity_x = track['xVelocity'][frame_ind]
                velocity_y = track['yVelocity'][frame_ind]
                all_scenarios[track['frame'][frame_ind]][vehid] = [position_x, position_y, orientation, velocity_x, velocity_y, veh_length, veh_width, veh_type]
        
        # sort the vehids by start frame
        center_vehids = [x for _, x in sorted(zip(center_vehids_start_frame, center_vehids))]
        current_frame = -1
        for vehid in center_vehids:
            # start_frame = max(static_info[vehid]['initialFrame']*2+history_length, current_frame+1)
            start_frame = static_info[vehid]['initialFrame']*2+history_length
            if start_frame > static_info[vehid]['finalFrame']*2 - future_length:
                continue
            for t0 in range(start_frame, static_info[vehid]['finalFrame']*2 - future_length, history_length):
                current_frame = t0
                nei_ids = [nei_id for nei_id in all_scenarios[t0-delta_frames].keys() if nei_id != vehid]
                nei_ids = [nei_id for nei_id in nei_ids if np.linalg.norm(np.array(all_scenarios[t0-delta_frames][vehid][:2])-np.array(all_scenarios[t0-delta_frames][nei_id][:2])) < 100]
                scenario_id, agent_inputs, agent_outputs, map_inputs = preprocessor.process_data(t0, {'scenario_id':scenario_id, 'sdc_id':vehid, 'nei_ids':nei_ids, 'tracks':tracks, 'scenes':all_scenarios, 'map_features':lanelet_network})
                vel_norm = np.linalg.norm(agent_inputs['obj_trajs'][:, :, 4:6], axis=-1)
                vel_norm = vel_norm[~np.isnan(vel_norm)]
                if np.mean(vel_norm) < 1.0:
                    continue
                scenario = dict()
                scenario['scenario_id'] = scenario_id + "_" + str(t0) + "_" + str(vehid) + SPLIT_KEY + str(file_path)
                scenario['tracks'] = np.concatenate([agent_inputs['obj_trajs'], agent_outputs['center_gt_trajs']], axis=1)
                scenario['tracks_valid'] = np.concatenate([agent_inputs['obj_trajs_mask'], agent_outputs['center_gt_trajs_mask']], axis=1)
                scenario['sdc_track_index'] = agent_inputs['sdc_track_index'][0]
                scenario['obj_types'] = agent_inputs['obj_types'][:, -1]
                scenario['obj_ids'] = agent_inputs['obj_ids'][:, -1]
                scenario['map_inputs'] = lanelet_network
                
                yield scenario

    logger.info(f"Worker {worker_index} finished read {len(files)} files.")
    # logger.info("Worker {}: Process {} waymo scenarios".format(worker_index, len(scenarios)))
    # return scenarios

def preprocess_highD_scenarios(files, worker_index):
    """
    Convert the highD files into scenarios. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenarios
    """
    delta_frames = 5
    history_length = 105
    future_length = 300
    lane_discretization = 0.5
    preprocessor = DataPreprocess(history_length, future_length, delta_frames, lane_discretization, num_points_each_polyline=21)
    for file_path in tqdm.tqdm(files, leave=False, position=0, desc="Worker {} Number of raw file".format(worker_index)):
        file_path = Path(file_path)
        filename = file_path.name
        logger.info(f"Worker {worker_index} is reading raw file: {filename}")
        scenario_id = filename.split('_')[-2]
        static_filename = file_path.parent / (scenario_id + "_tracksMeta.csv")
        meta_filename = file_path.parent / (scenario_id + "_recordingMeta.csv")
        if ("tracks.csv" not in filename):
            logger.info(f"Worker {worker_index} skip this file: {filename}")
            continue
        static_info = pd.read_csv(static_filename).to_dict(orient="records")
        static_info = {x['id']:x for x in static_info}
        meta_info = pd.read_csv(meta_filename)
        raw_tracks = pd.read_csv(file_path).groupby("id", sort=False)
        lanelet_network_upper = create_highd_lanelet_network(meta_info, "upper")
        lanelet_network_lower = create_highd_lanelet_network(meta_info, "lower")

        tracks_upper, tracks_lower = {}, {}
        for track_id, track_rows in raw_tracks:
            track = track_rows.to_dict(orient="list")
            for key, value in track.items():
                if key == "id":
                    track[key] = value[0]
                elif key in ["y", "yVelocity"]:
                    track[key] = -np.array(value)
                else:
                    track[key] = np.array(value)
            track["center"] = np.stack([track["x"], track["y"]], axis=-1)
            track["xCenter"] = track["x"]
            track["yCenter"] = track["y"]
            track["heading"] = np.rad2deg(np.arctan2(track["yVelocity"], track["xVelocity"]))
            if static_info[track_id]['drivingDirection'] == 1:
                # upper lanes 
                tracks_upper[track_id] = track
            else:
                tracks_lower[track_id] = track
        
        center_vehids_upper, center_vehids_upper_start_frame = [], []
        for track_id in tracks_upper:
            tracks_upper[track_id] = interpolate_track(tracks_upper[track_id])
            vehid = tracks_upper[track_id]['id']
            veh_type = 'car' if static_info[vehid]['class'] == 'Car' else 'truck'
            if veh_type != 'bus' and veh_type != 'trailer' and veh_type != 'pedestrian' and veh_type != 'bicycle' and veh_type != 'motorcycle' and len(tracks_upper[track_id]['frame']) > history_length+future_length:
                center_vehids_upper.append(vehid)
                center_vehids_upper_start_frame.append(tracks_upper[track_id]['frame'][0])
        center_vehids_lower, center_vehids_lower_start_frame = [], []
        for track_id in tracks_lower:
            tracks_lower[track_id] = interpolate_track(tracks_lower[track_id])
            vehid = tracks_lower[track_id]['id']
            veh_type = 'car' if static_info[vehid]['class'] == 'Car' else 'truck'
            if veh_type != 'bus' and veh_type != 'trailer' and veh_type != 'pedestrian' and veh_type != 'bicycle' and veh_type != 'motorcycle' and len(tracks_lower[track_id]['frame']) > history_length+future_length:
                center_vehids_lower.append(vehid)
                center_vehids_lower_start_frame.append(tracks_lower[track_id]['frame'][0])

        all_scenarios_upper = defaultdict(lambda:dict())
        for track in tracks_upper.values():
            vehid = track['id']
            veh_length = static_info[vehid]['width']
            veh_width = static_info[vehid]['height']
            veh_type = 'car' if static_info[vehid]['class'] == 'Car' else 'truck'
            for frame_ind in range(len(track['frame'])):
                position_x = track['xCenter'][frame_ind]
                position_y = track['yCenter'][frame_ind]
                orientation = np.deg2rad(track['heading'][frame_ind])
                velocity_x = track['xVelocity'][frame_ind]
                velocity_y = track['yVelocity'][frame_ind]
                all_scenarios_upper[track['frame'][frame_ind]][vehid] = [position_x, position_y, orientation, velocity_x, velocity_y, veh_length, veh_width, veh_type]
        all_scenarios_lower = defaultdict(lambda:dict())
        for track in tracks_lower.values():
            vehid = track['id']
            veh_length = static_info[vehid]['width']
            veh_width = static_info[vehid]['height']
            veh_type = 'car' if static_info[vehid]['class'] == 'Car' else 'truck'
            for frame_ind in range(len(track['frame'])):
                position_x = track['xCenter'][frame_ind]
                position_y = track['yCenter'][frame_ind]
                orientation = np.deg2rad(track['heading'][frame_ind])
                velocity_x = track['xVelocity'][frame_ind]
                velocity_y = track['yVelocity'][frame_ind]
                all_scenarios_lower[track['frame'][frame_ind]][vehid] = [position_x, position_y, orientation, velocity_x, velocity_y, veh_length, veh_width, veh_type]
        
        # sort the vehids by start frame
        center_vehids_upper = [x for _, x in sorted(zip(center_vehids_upper_start_frame, center_vehids_upper))]
        center_vehids_lower = [x for _, x in sorted(zip(center_vehids_lower_start_frame, center_vehids_lower))]

        for scenario_name, center_vehids, all_scenarios, tracks, lanelet_network in zip(['upper', 'lower'], [center_vehids_upper, center_vehids_lower], [all_scenarios_upper, all_scenarios_lower], [tracks_upper, tracks_lower], [lanelet_network_upper, lanelet_network_lower]):
            current_frame = -1
            for vehid in center_vehids:
                start_frame = max(static_info[vehid]['initialFrame']*2+history_length, current_frame+1)
                if start_frame > static_info[vehid]['finalFrame']*2 - future_length:
                    continue
                for t0 in range(start_frame, static_info[vehid]['finalFrame']*2 - future_length, history_length):
                    current_frame = t0
                    nei_ids = [nei_id for nei_id in all_scenarios[t0-delta_frames].keys() if nei_id != vehid]
                    nei_ids = [nei_id for nei_id in nei_ids if np.linalg.norm(np.array(all_scenarios[t0-delta_frames][vehid][:2])-np.array(all_scenarios[t0-delta_frames][nei_id][:2])) < 100]
                    scenario_id, agent_inputs, agent_outputs, map_inputs = preprocessor.process_data(t0, {'scenario_id':scenario_id, 'sdc_id':vehid, 'nei_ids':nei_ids, 'tracks':tracks, 'scenes':all_scenarios, 'map_features':lanelet_network})
                    scenario = dict()
                    scenario['scenario_id'] = scenario_id + "_" + scenario_name + "_" + str(t0) + SPLIT_KEY + str(file_path)
                    scenario['tracks'] = np.concatenate([agent_inputs['obj_trajs'], agent_outputs['center_gt_trajs']], axis=1)
                    scenario['tracks_valid'] = np.concatenate([agent_inputs['obj_trajs_mask'], agent_outputs['center_gt_trajs_mask']], axis=1)
                    scenario['sdc_track_index'] = agent_inputs['sdc_track_index'][0]
                    scenario['obj_types'] = agent_inputs['obj_types'][:, -1]
                    scenario['obj_ids'] = agent_inputs['obj_ids'][:, -1]
                    scenario['map_inputs'] = lanelet_network
                    
                    yield scenario

    logger.info(f"Worker {worker_index} finished read {len(files)} files.")
    # logger.info("Worker {}: Process {} waymo scenarios".format(worker_index, len(scenarios)))
    # return scenarios

def preprocess_terasim_scenarios(files, worker_index):
    """
    Convert the rounD files into scenarios. This happens in each worker.
    :param files: a list of file path
    :param worker_index, the index for the worker
    :return: a list of scenarios
    """
    delta_frames = 1
    history_length = 21
    future_length = 60
    lane_discretization = 0.5
    map_file = '/mnt/space/data/terasim/mcity_lanelet_network.xml'
    preprocessor = DataPreprocess(history_length, future_length, delta_frames, lane_discretization, num_points_each_polyline=21)
    lanelet_network = CommonRoadFileReader(map_file).open_lanelet_network()

    # file_path = "/mnt/space/data/terasim/test_scenario"
    # with open(file_path, "rb") as f:
    #     data = f.read()
    # example_scenario = scenario_pb2.Scenario()
    # example_scenario.ParseFromString(data)
    # map_features = example_scenario.map_features

    for file_path in tqdm.tqdm(files, leave=False, position=0, desc="Worker {} Number of raw file".format(worker_index)):
        file_path = Path(file_path)
        filename = file_path.name
        logger.info(f"Worker {worker_index} is reading raw file: {filename}")
        scenario_id = filename.split('_')[-2]
        static_filename = file_path.parent / (scenario_id + "_tracksMeta.csv")
        meta_filename = file_path.parent / (scenario_id + "_recordingMeta.csv")
        if ("tracks.csv" not in filename):
            logger.info(f"Worker {worker_index} skip this file: {filename}")
            continue
        static_info = pd.read_csv(static_filename).to_dict(orient="records")
        static_info = {x['trackId']:x for x in static_info}
        raw_tracks = pd.read_csv(file_path).groupby("trackId", sort=False)
        tracks = {}
        
        for track_id, track_rows in raw_tracks:
            track = track_rows.to_dict(orient="list")
            for key, value in track.items():
                if key in ["trackId", "recordingId"]:
                    track[key] = value[0]
                else:
                    track[key] = np.array(value)
            track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
            tracks[track_id] = track
        
        center_vehids, center_vehids_start_frame = [], []
        for track_id in tracks:
            vehid = tracks[track_id]['trackId']
            veh_type = static_info[track_id]['class']
            if veh_type != 'bus' and veh_type != 'trailer' and veh_type != 'pedestrian' and veh_type != 'bicycle' and veh_type != 'motorcycle' and len(tracks[track_id]['frame']) > history_length+future_length:
                center_vehids.append(vehid)
                center_vehids_start_frame.append(tracks[track_id]['frame'][0])
        all_scenarios = defaultdict(lambda:dict())
        for track in tracks.values():
            vehid = track['trackId']
            veh_length = static_info[vehid]['length']
            veh_width = static_info[vehid]['width']
            veh_type = static_info[vehid]['class']
            
            for frame_ind in range(len(track['frame'])):
                position_x = track['xCenter'][frame_ind]
                position_y = track['yCenter'][frame_ind]
                orientation = np.deg2rad(track['heading'][frame_ind])
                velocity_x = track['xVelocity'][frame_ind]
                velocity_y = track['yVelocity'][frame_ind]
                all_scenarios[track['frame'][frame_ind]][vehid] = [position_x, position_y, orientation, velocity_x, velocity_y, veh_length, veh_width, veh_type]
        
        # sort the vehids by start frame
        center_vehids = [x for _, x in sorted(zip(center_vehids_start_frame, center_vehids))]
        # current_frame = -1

        delta_length = 10
        for vehid in center_vehids:
            start_frame = static_info[vehid]['initialFrame']+history_length
            # if start_frame > static_info[vehid]['finalFrame'] - future_length:
            #     continue
            for t0 in range(start_frame, static_info[vehid]['finalFrame'] - future_length, delta_length):
                # current_frame = t0
                nei_ids = [nei_id for nei_id in all_scenarios[t0-delta_frames].keys() if nei_id != vehid]
                nei_ids = [nei_id for nei_id in nei_ids if np.linalg.norm(np.array(all_scenarios[t0-delta_frames][vehid][:2])-np.array(all_scenarios[t0-delta_frames][nei_id][:2])) < 100]
                scenario_id, agent_inputs, agent_outputs, _ = preprocessor.process_data(t0, {'scenario_id':scenario_id, 'sdc_id':vehid, 'nei_ids':nei_ids, 'tracks':tracks, 'scenes':all_scenarios, 'map_features':lanelet_network})
                scenario = dict()
                scenario['scenario_id'] = scenario_id + "_" + str(t0) + "_" + str(vehid) + SPLIT_KEY + str(file_path)
                scenario['tracks'] = np.concatenate([agent_inputs['obj_trajs'], agent_outputs['center_gt_trajs']], axis=1)
                scenario['tracks_valid'] = np.concatenate([agent_inputs['obj_trajs_mask'], agent_outputs['center_gt_trajs_mask']], axis=1)
                scenario['sdc_track_index'] = agent_inputs['sdc_track_index'][0]
                scenario['obj_types'] = agent_inputs['obj_types'][:, -1]
                scenario['obj_ids'] = agent_inputs['obj_ids'][:, -1]
                scenario['map_inputs'] = lanelet_network
                
                yield scenario

    logger.info(f"Worker {worker_index} finished read {len(files)} files.")
    # logger.info("Worker {}: Process {} waymo scenarios".format(worker_index, len(scenarios)))
    # return scenarios

def convert_rounD_scenario(scenario, version):
    scenario = scenario
    md_scenario = SD()

    id_end = scenario['scenario_id'].find(SPLIT_KEY)

    md_scenario[SD.ID] = scenario['scenario_id'][:id_end]
    md_scenario[SD.VERSION] = version

    track_length = scenario['tracks'].shape[1]

    tracks, sdc_id = extract_tracks(scenario['tracks'], scenario['tracks_valid'], scenario['obj_ids'], scenario['obj_types'], scenario['sdc_track_index'], track_length, "terasim")

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    map_features = extract_map_features(scenario['map_inputs'])
    md_scenario[SD.MAP_FEATURES] = map_features

    compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
    md_scenario[SD.METADATA]["dataset"] = "rounD"
    md_scenario[SD.METADATA]["scenario_id"] = scenario['scenario_id'][:id_end]
    md_scenario[SD.METADATA]["source_file"] = scenario['scenario_id'][id_end + 1:]
    md_scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    md_scenario[SD.METADATA]["current_time_index"] = 20

    # obj id
    md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj_id) for obj_id in scenario['obj_ids']]

    md_scenario[SD.METADATA]["sdc_track_index"] = int(scenario['sdc_track_index'])

    track_index = [idx for idx in range(len(scenario['obj_ids']))][:4]
    if md_scenario[SD.METADATA]["sdc_track_index"] not in track_index:
        track_index.append(md_scenario[SD.METADATA]["sdc_track_index"])
    track_id = [str(scenario['obj_ids'][idx]) for idx in track_index]
    
    track_obj_type = [tracks[id]["type"] for id in track_id]
    md_scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": 0,
            "object_type": track_obj_type[count]
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    del scenario
    return md_scenario

def convert_inD_scenario(scenario, version):
    scenario = scenario
    md_scenario = SD()

    id_end = scenario['scenario_id'].find(SPLIT_KEY)

    md_scenario[SD.ID] = scenario['scenario_id'][:id_end]
    md_scenario[SD.VERSION] = version

    track_length = scenario['tracks'].shape[1]

    tracks, sdc_id = extract_tracks(scenario['tracks'], scenario['tracks_valid'], scenario['obj_ids'], scenario['obj_types'], scenario['sdc_track_index'], track_length, "inD")

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    map_features = extract_map_features(scenario['map_inputs'])
    md_scenario[SD.MAP_FEATURES] = map_features

    compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
    md_scenario[SD.METADATA]["dataset"] = "inD"
    md_scenario[SD.METADATA]["scenario_id"] = scenario['scenario_id'][:id_end]
    md_scenario[SD.METADATA]["source_file"] = scenario['scenario_id'][id_end + 1:]
    md_scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    md_scenario[SD.METADATA]["current_time_index"] = 20

    # obj id
    md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj_id) for obj_id, obj_type in zip(scenario['obj_ids'], scenario['obj_types']) if obj_type <= 3]

    md_scenario[SD.METADATA]["sdc_track_index"] = int(scenario['sdc_track_index'])

    # track_index = [idx for idx in range(len(scenario['obj_ids']))][:2]
    track_index = []
    if md_scenario[SD.METADATA]["sdc_track_index"] not in track_index:
        track_index.append(md_scenario[SD.METADATA]["sdc_track_index"])
    track_id = [str(scenario['obj_ids'][idx]) for idx in track_index]
    
    track_obj_type = [tracks[id]["type"] for id in track_id]
    md_scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": 0,
            "object_type": track_obj_type[count]
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    del scenario
    return md_scenario

def convert_sinD_scenario(scenario, version):
    scenario = scenario
    md_scenario = SD()

    id_end = scenario['scenario_id'].find(SPLIT_KEY)

    md_scenario[SD.ID] = scenario['scenario_id'][:id_end]
    md_scenario[SD.VERSION] = version

    track_length = scenario['tracks'].shape[1]

    tracks, sdc_id = extract_tracks(scenario['tracks'], scenario['tracks_valid'], scenario['obj_ids'], scenario['obj_types'], scenario['sdc_track_index'], track_length, "sinD")

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    map_features = extract_map_features(scenario['map_inputs'])
    md_scenario[SD.MAP_FEATURES] = map_features

    compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
    md_scenario[SD.METADATA]["dataset"] = "sinD"
    md_scenario[SD.METADATA]["scenario_id"] = scenario['scenario_id'][:id_end]
    md_scenario[SD.METADATA]["source_file"] = scenario['scenario_id'][id_end + 1:]
    md_scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    md_scenario[SD.METADATA]["current_time_index"] = 20

    # obj id
    md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj_id) for obj_id, obj_type in zip(scenario['obj_ids'], scenario['obj_types']) if obj_type <= 3]

    md_scenario[SD.METADATA]["sdc_track_index"] = int(scenario['sdc_track_index'])

    track_index = [idx for idx in range(len(scenario['obj_ids']))][:2]
    if md_scenario[SD.METADATA]["sdc_track_index"] not in track_index:
        track_index.append(md_scenario[SD.METADATA]["sdc_track_index"])
    track_id = [str(scenario['obj_ids'][idx]) for idx in track_index]
    
    track_obj_type = [tracks[id]["type"] for id in track_id]
    md_scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": 0,
            "object_type": track_obj_type[count]
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    del scenario
    return md_scenario

def convert_highD_scenario(scenario, version):
    scenario = scenario
    md_scenario = SD()

    id_end = scenario['scenario_id'].find(SPLIT_KEY)

    md_scenario[SD.ID] = scenario['scenario_id'][:id_end]
    md_scenario[SD.VERSION] = version

    track_length = scenario['tracks'].shape[1]

    tracks, sdc_id = extract_tracks(scenario['tracks'], scenario['tracks_valid'], scenario['obj_ids'], scenario['obj_types'], scenario['sdc_track_index'], track_length, "terasim")

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    map_features = extract_map_features(scenario['map_inputs'])
    md_scenario[SD.MAP_FEATURES] = map_features

    compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
    md_scenario[SD.METADATA]["dataset"] = "highD"
    md_scenario[SD.METADATA]["scenario_id"] = scenario['scenario_id'][:id_end]
    md_scenario[SD.METADATA]["source_file"] = scenario['scenario_id'][id_end + 1:]
    md_scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    md_scenario[SD.METADATA]["current_time_index"] = 20

    # obj id
    md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj_id) for obj_id in scenario['obj_ids']]

    md_scenario[SD.METADATA]["sdc_track_index"] = int(scenario['sdc_track_index'])

    track_index = [idx for idx in range(len(scenario['obj_ids']))][:4]
    if md_scenario[SD.METADATA]["sdc_track_index"] not in track_index:
        track_index.append(md_scenario[SD.METADATA]["sdc_track_index"])
    track_id = [str(scenario['obj_ids'][idx]) for idx in track_index]
    
    track_obj_type = [tracks[id]["type"] for id in track_id]
    md_scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": 0,
            "object_type": track_obj_type[count]
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    del scenario
    return md_scenario

def convert_terasim_scenario(scenario, version):
    scenario = scenario
    md_scenario = SD()

    id_end = scenario['scenario_id'].find(SPLIT_KEY)

    md_scenario[SD.ID] = scenario['scenario_id'][:id_end]
    md_scenario[SD.VERSION] = version

    track_length = scenario['tracks'].shape[1]

    tracks, sdc_id = extract_tracks(scenario['tracks'], scenario['tracks_valid'], scenario['obj_ids'], scenario['obj_types'], scenario['sdc_track_index'], track_length, "terasim")

    md_scenario[SD.LENGTH] = track_length

    md_scenario[SD.TRACKS] = tracks

    md_scenario[SD.DYNAMIC_MAP_STATES] = {}

    map_features = extract_map_features(scenario['map_inputs'])
    md_scenario[SD.MAP_FEATURES] = map_features

    compute_width(md_scenario[SD.MAP_FEATURES])

    md_scenario[SD.METADATA] = {}
    md_scenario[SD.METADATA][SD.ID] = md_scenario[SD.ID]
    md_scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO
    md_scenario[SD.METADATA][SD.TIMESTEP] = np.array(list(range(track_length))) / 10
    md_scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    md_scenario[SD.METADATA][SD.SDC_ID] = str(sdc_id)
    md_scenario[SD.METADATA]["dataset"] = "terasim"
    md_scenario[SD.METADATA]["scenario_id"] = scenario['scenario_id'][:id_end]
    md_scenario[SD.METADATA]["source_file"] = scenario['scenario_id'][id_end + 1:]
    md_scenario[SD.METADATA]["track_length"] = track_length

    # === Waymo specific data. Storing them here ===
    md_scenario[SD.METADATA]["current_time_index"] = 20

    # obj id
    md_scenario[SD.METADATA]["objects_of_interest"] = [str(obj_id) for obj_id, obj_type in zip(scenario['obj_ids'], scenario['obj_types'])]

    md_scenario[SD.METADATA]["sdc_track_index"] = int(scenario['sdc_track_index'])

    track_index = [int(scenario['sdc_track_index'])]
    if md_scenario[SD.METADATA]["sdc_track_index"] not in track_index:
        track_index.append(md_scenario[SD.METADATA]["sdc_track_index"])
    track_id = [str(scenario['obj_ids'][idx]) for idx in track_index]
    
    track_obj_type = [tracks[id]["type"] for id in track_id]
    md_scenario[SD.METADATA]["tracks_to_predict"] = {
        id: {
            "track_index": track_index[count],
            "track_id": id,
            "difficulty": 0,
            "object_type": track_obj_type[count]
        }
        for count, id in enumerate(track_id)
    }
    # clean memory
    del scenario
    return md_scenario

def extract_tracks(tracks, tracks_valid, obj_ids, obj_types, sdc_idx, track_length, dataset):
    ret = dict()

    def _object_state_template(object_id):
        return dict(
            type=None,
            state=dict(

                # Never add extra dim if the value is scalar.
                position=np.zeros([track_length, 3], dtype=np.float32),
                length=np.zeros([track_length], dtype=np.float32),
                width=np.zeros([track_length], dtype=np.float32),
                height=np.zeros([track_length], dtype=np.float32),
                heading=np.zeros([track_length], dtype=np.float32),
                velocity=np.zeros([track_length, 2], dtype=np.float32),
                valid=np.zeros([track_length], dtype=bool),
            ),
            metadata=dict(track_length=track_length, type=None, object_id=object_id, dataset=dataset)
        )
    
    for index in range(tracks.shape[0]):
        obj = tracks[index]
        obj_valid = tracks_valid[index]
        object_id = str(obj_ids[index])

        obj_state = _object_state_template(object_id)
        obj_type = obj_types[index]
        if obj_type <=3:
            obj_state['type'] = MetaDriveType.VEHICLE
        elif obj_type == 5:
            obj_state['type'] = MetaDriveType.PEDESTRIAN
        elif obj_type >=6:
            obj_state['type'] = MetaDriveType.CYCLIST
        else:
            obj_state['type'] = MetaDriveType.OTHER
        for step_count in range(obj.shape[0]):
            if obj_valid[step_count] == 0:
                continue
            obj_state['state']['position'][step_count][0] = obj[step_count, 0]
            obj_state['state']['position'][step_count][1] = obj[step_count, 1]
            obj_state['state']['position'][step_count][2] = 0
            obj_state['state']['heading'][step_count] = np.arctan2(obj[step_count, 3], obj[step_count, 2])
            obj_state['state']['velocity'][step_count][0] = obj[step_count, 4]
            obj_state['state']['velocity'][step_count][1] = obj[step_count, 5]

            obj_state['state']['length'][step_count] = obj[step_count, 6]
            obj_state['state']['width'][step_count] = obj[step_count, 7]
            obj_state['state']['height'][step_count] = 1

            obj_state['state']['valid'][step_count] = True
        
        obj_state["metadata"]["type"] = obj_state['type']
        
        ret[object_id] = obj_state

    return ret,  str(obj_ids[sdc_idx])

def mph_to_kmh(speed_in_mph: float):
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh

def extract_poly(vertices):
    lane_discretization = 0.5
    center_vertices = data_utils.resample_polyline(vertices, lane_discretization)

    x = [xy[0] for xy in center_vertices]
    y = [xy[1] for xy in center_vertices]
    z = [0.0 for xy in center_vertices]
    coord = np.stack((x, y, z), axis=1).astype("float32")
    return coord

def extract_boundaries(f, side):
    b = []
    center_vertices = data_utils.resample_polyline(f.center_vertices, 0.5)
    # b = np.zeros([len(fb), 4], dtype="int64")
    c = dict()
    c["lane_start_index"] = 0
    c["lane_end_index"] = len(center_vertices)-1
    if side == 'left':
        if f.line_marking_left_vertices == LineMarking.DASHED or f.line_marking_left_vertices == LineMarking.BROAD_DASHED:
            c["boundary_type"] = 'ROAD_LINE_BROKEN_SINGLE_WHITE'
        elif f.line_marking_left_vertices == LineMarking.SOLID:
            c["boundary_type"] = 'ROAD_LINE_SOLID_SINGLE_WHITE'
        else:
            c["boundary_type"] = 'UNKNOWN'
        c["boundary_feature_id"] = '1000' + str(f.lanelet_id)
    else:
        if f.line_marking_right_vertices == LineMarking.DASHED or f.line_marking_right_vertices == LineMarking.BROAD_DASHED:
            c["boundary_type"] = 'ROAD_LINE_BROKEN_SINGLE_WHITE'
        elif f.line_marking_right_vertices == LineMarking.SOLID:
            c["boundary_type"] = 'ROAD_LINE_SOLID_SINGLE_WHITE'
        else:
            c["boundary_type"] = 'UNKNOWN'
        if f.adj_right is not None:
            c["boundary_feature_id"] = '1000' + str(f.adj_right)
        else:
            c["boundary_feature_id"] = '2000' + str(f.lanelet_id)
    for key in c:
        c[key] = str(c[key])
    b.append(c)

    return b


def extract_neighbors(f, ln, side):
    nbs = []
    center_vertices = data_utils.resample_polyline(f.center_vertices, 0.5)
    nb = dict()
    if side == 'left' and f.adj_left is not None:
        nb["feature_id"] = f.adj_left
        nb["self_start_index"] = 0
        nb["self_end_index"] = len(center_vertices)-1
        nb["neighbor_start_index"] = 0
        left_center_vertices = data_utils.resample_polyline(ln.find_lanelet_by_id(f.adj_left).center_vertices, 0.5)
        nb["neighbor_end_index"] = len(left_center_vertices) - 1
        for key in nb:
            nb[key] = str(nb[key])
        nb["boundaries"] = extract_boundaries(f, side)
        nbs.append(nb)
    elif side == 'right' and f.adj_right is not None:
        nb["feature_id"] = f.adj_right
        nb["self_start_index"] = 0
        nb["self_end_index"] = len(center_vertices)-1
        nb["neighbor_start_index"] = 0
        right_center_vertices = data_utils.resample_polyline(ln.find_lanelet_by_id(f.adj_right).center_vertices, 0.5)
        nb["neighbor_end_index"] = len(right_center_vertices) - 1
        for key in nb:
            nb[key] = str(nb[key])
        nb["boundaries"] = extract_boundaries(f, side)
        nbs.append(nb)
    return nbs

def extract_center(f, ln):
    center = dict()
    center["speed_limit_mph"] = 75 if LaneletType.INTERSTATE in f.lanelet_type else 31

    center["speed_limit_kmh"] = mph_to_kmh(center["speed_limit_mph"])

    if LaneletType.INTERSTATE in f.lanelet_type:
        center["type"] = 'LANE_FREEWAY'
    elif LaneletType.BICYCLE_LANE in f.lanelet_type:
        center["type"] = 'LANE_BIKE_LANE'
    else:
        center["type"] = 'LANE_SURFACE_STREET'
    
    center["polyline"] = extract_poly(f.center_vertices)

    center["interpolating"] = False

    center["entry_lanes"] = [str(x) for x in f.predecessor]

    center["exit_lanes"] = [str(x) for x in f.successor]

    center["left_boundaries"] = extract_boundaries(f, 'left')

    center["right_boundaries"] = extract_boundaries(f, 'right')

    center["left_neighbor"] = extract_neighbors(f, ln, 'left')

    center["right_neighbor"] = extract_neighbors(f, ln, 'right')

    return center

def extract_left_line(f):
    line = dict()
    if f.line_marking_left_vertices == LineMarking.DASHED or f.line_marking_left_vertices == LineMarking.BROAD_DASHED:
        line["type"] = 'ROAD_LINE_BROKEN_SINGLE_WHITE'
    elif f.line_marking_left_vertices == LineMarking.SOLID:
        line["type"] = 'ROAD_LINE_SOLID_SINGLE_WHITE'
    else:
        line["type"] = 'UNKNOWN'
    line["polyline"] = extract_poly(f.left_vertices)
    return line

def extract_right_line(f):
    line = dict()
    if f.line_marking_right_vertices == LineMarking.DASHED or f.line_marking_right_vertices == LineMarking.BROAD_DASHED:
        line["type"] = 'ROAD_LINE_BROKEN_SINGLE_WHITE'
    elif f.line_marking_right_vertices == LineMarking.SOLID:
        line["type"] = 'ROAD_LINE_SOLID_SINGLE_WHITE'
    else:
        line["type"] = 'UNKNOWN'
    line["polyline"] = extract_poly(f.right_vertices)
    return line

def extract_left_edge(f):
    edge = dict()
    edge["type"] = 'ROAD_EDGE_BOUNDARY'
    edge["polyline"] = extract_poly(f.left_vertices)
    return edge

def extract_right_edge(f):
    edge = dict()
    edge["type"] = 'ROAD_EDGE_BOUNDARY'
    edge["polyline"] = extract_poly(f.right_vertices)
    return edge

def extract_map_features(map_features):
    ret = {}

    for lanelet in map_features.lanelets:
        lane_id = str(lanelet.lanelet_id)
        ret[lane_id] = extract_center(lanelet, map_features)
        if lanelet.adj_left is not None:
            ret['1000'+lane_id] = extract_left_line(lanelet)
        else:
            ret['1000'+lane_id] = extract_left_edge(lanelet)
        if lanelet.adj_right is None:
            ret['2000'+lane_id] = extract_right_edge(lanelet)

    return ret


def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)

def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0], dtype="float32")
    for b in boundary:
        boundary_int = {k: int(v) if k != "boundary_type" else v for k, v in b.items()}  # All values are int

        b_feat_id = str(boundary_int["boundary_feature_id"])
        lb = map[b_feat_id]
        b_polyline = lb["polyline"][:, :2]

        start_p = polyline[boundary_int["lane_start_index"]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = boundary_int["lane_end_index"] - boundary_int["lane_start_index"]
        end_index = min(start_index + seg_len, lb["polyline"].shape[0] - 1)
        length = min(end_index - start_index, seg_len) + 1
        self_range = range(boundary_int["lane_start_index"], boundary_int["lane_start_index"] + length)
        bound_range = range(start_index, start_index + length)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width


def compute_width(map):
    for map_feat_id, lane in map.items():

        if not "LANE" in lane["type"]:
            continue

        width = np.zeros((lane["polyline"].shape[0], 2), dtype="float32")

        width[:, 0] = extract_width(map, lane["polyline"][:, :2], lane["left_boundaries"])
        width[:, 1] = extract_width(map, lane["polyline"][:, :2], lane["right_boundaries"])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        lane["width"] = width
    return

from scenarionet.converter.waymo.type import WaymoLaneType, WaymoRoadLineType, WaymoRoadEdgeType

def extract_waymo_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1).astype("float32")
    return coord

def extract_waymo_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype="int64")
    for k in range(len(fb)):
        c = dict()
        c["lane_start_index"] = fb[k].lane_start_index
        c["lane_end_index"] = fb[k].lane_end_index
        c["boundary_type"] = WaymoRoadLineType.from_waymo(fb[k].boundary_type)
        c["boundary_feature_id"] = fb[k].boundary_feature_id
        for key in c:
            c[key] = str(c[key])
        b.append(c)

    return b

def extract_waymo_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb["feature_id"] = fb[k].feature_id
        nb["self_start_index"] = fb[k].self_start_index
        nb["self_end_index"] = fb[k].self_end_index
        nb["neighbor_start_index"] = fb[k].neighbor_start_index
        nb["neighbor_end_index"] = fb[k].neighbor_end_index
        for key in nb:
            nb[key] = str(nb[key])
        nb["boundaries"] = extract_waymo_boundaries(fb[k].boundaries)
        nbs.append(nb)
    return nbs


def extract_waymo_center(f):
    center = dict()
    f = f.lane
    center["speed_limit_mph"] = f.speed_limit_mph

    center["speed_limit_kmh"] = mph_to_kmh(f.speed_limit_mph)

    center["type"] = WaymoLaneType.from_waymo(f.type)

    center["polyline"] = extract_waymo_poly(f.polyline)

    center["interpolating"] = f.interpolating

    center["entry_lanes"] = [x for x in f.entry_lanes]

    center["exit_lanes"] = [x for x in f.exit_lanes]

    center["left_boundaries"] = extract_waymo_boundaries(f.left_boundaries)

    center["right_boundaries"] = extract_waymo_boundaries(f.right_boundaries)

    center["left_neighbor"] = extract_waymo_neighbors(f.left_neighbors)

    center["right_neighbor"] = extract_waymo_neighbors(f.right_neighbors)

    return center

def extract_waymo_line(f):
    line = dict()
    f = f.road_line
    line["type"] = WaymoRoadLineType.from_waymo(f.type)
    line["polyline"] = extract_waymo_poly(f.polyline)
    return line

def extract_waymo_edge(f):
    edge = dict()
    f_ = f.road_edge

    edge["type"] = WaymoRoadEdgeType.from_waymo(f_.type)

    edge["polyline"] = extract_waymo_poly(f_.polyline)

    return edge

def extract_waymo_stop(f):
    stop = dict()
    f = f.stop_sign
    stop["type"] = MetaDriveType.STOP_SIGN
    stop["lane"] = [x for x in f.lane]
    stop["position"] = np.array([f.position.x, f.position.y, f.position.z], dtype="float32")
    return stop

def extract_waymo_driveway(f):
    driveway_data = dict()
    f = f.driveway
    driveway_data["type"] = MetaDriveType.DRIVEWAY
    driveway_data["polygon"] = extract_waymo_poly(f.polygon)
    return driveway_data

def extract_waymo_crosswalk(f):
    cross_walk = dict()
    f = f.crosswalk
    cross_walk["type"] = MetaDriveType.CROSSWALK
    cross_walk["polygon"] = extract_waymo_poly(f.polygon)
    return cross_walk


def extract_waymo_bump(f):
    speed_bump_data = dict()
    f = f.speed_bump
    speed_bump_data["type"] = MetaDriveType.SPEED_BUMP
    speed_bump_data["polygon"] = extract_waymo_poly(f.polygon)
    return speed_bump_data

def extract_waymo_map_features(map_features):
    ret = {}

    for lane_state in map_features:
        lane_id = str(lane_state.id)

        if lane_state.HasField("lane"):
            ret[lane_id] = extract_waymo_center(lane_state)

        if lane_state.HasField("road_line"):
            ret[lane_id] = extract_waymo_line(lane_state)

        if lane_state.HasField("road_edge"):
            ret[lane_id] = extract_waymo_edge(lane_state)

        if lane_state.HasField("stop_sign"):
            ret[lane_id] = extract_waymo_stop(lane_state)

        if lane_state.HasField("crosswalk"):
            ret[lane_id] = extract_waymo_crosswalk(lane_state)

        if lane_state.HasField("speed_bump"):
            ret[lane_id] = extract_waymo_bump(lane_state)

        # Supported only in Waymo dataset 1.2.0
        if lane_state.HasField("driveway"):
            ret[lane_id] = extract_waymo_driveway(lane_state)

    return ret