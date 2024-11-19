import numpy as np
from domain_expansion.utils import data_utils
from commonroad.scenario.lanelet import LineMarking, LaneletType

class DataPreprocess(object):
    def __init__(self, history_length, pred_length, step_size, lane_discretization, num_points_each_polyline=5):
        self.history_length = history_length
        self.pred_length = pred_length
        self.step_size = step_size
        self.data_processor = DataProcess(step_size, lane_discretization, num_points_each_polyline)

    def process_data(self, t0, scenario):
        """
        Process the data to the needed format
        :return:
        """
        scenario_id = scenario['scenario_id']
        sdc_id = scenario['sdc_id']
        nei_ids = scenario['nei_ids']
        # Prepare input and ground truth
        agent_inputs, agent_outputs, map_inputs = self.data_processor.prepare_input_data(t0 = t0,
            history_length=self.history_length, pred_length=self.pred_length,
            scenes=scenario['scenes'], map=scenario['map_features'], ego_veh_id=sdc_id, neighbor_veh_ids=nei_ids)

        return scenario_id, agent_inputs, agent_outputs, map_inputs


class DataProcess(object):

    def __init__(self, delta_steps, lane_discretization=0.5, num_points_each_polyline=5):
        self.delta_steps = delta_steps
        self.lane_discretization = lane_discretization
        self.num_points_each_polyline = num_points_each_polyline
        self.veh_types = {'car':0, 'truck':1, 'bus':2, 'van':3, 'trailer':4, 'pedestrian':5, 'pedestrian/bicycle':5, 'bicycle':6, 'motorcycle':7, 'tricycle':8}

    def flatten_trajectory(self, scenes, start_time, time_length, veh_ids):

        """
        Generate buffer array from trajectory pool

        :param traj_pool:
        :param time_length:
        :param max_num_agents:
        :param output_vid:
        :return:
        """

        # create agent feature buffer (x, y, heading, vx, vy, l, w, h)
        veh_num = len(veh_ids)
        
        horizon_len = time_length//self.delta_steps
        buff_x = np.empty([veh_num, horizon_len])
        buff_x[:] = np.nan

        buff_y = np.empty([veh_num, horizon_len])
        buff_y[:] = np.nan

        buff_heading = np.empty([veh_num, horizon_len])
        buff_heading[:] = np.nan

        buff_v_x = np.empty([veh_num, horizon_len])
        buff_v_x[:] = np.nan

        buff_v_y = np.empty([veh_num, horizon_len])
        buff_v_y[:] = np.nan

        buff_length = np.empty([veh_num, horizon_len])
        buff_length[:] = np.nan

        buff_width = np.empty([veh_num, horizon_len])
        buff_width[:] = np.nan

        buff_type = np.empty([veh_num, horizon_len])
        buff_type[:] = np.nan

        buff_vid = np.empty([veh_num, horizon_len])
        buff_vid[:] = np.nan

        # fill-in x and y and heading buffer
        for i, veh_id in enumerate(veh_ids):
            for t in range(horizon_len):
                if veh_id in scenes[start_time + t*self.delta_steps]: 
                    buff_x[i, t] , buff_y[i, t], buff_heading[i, t], \
                        buff_v_x[i, t], buff_v_y[i, t], buff_length[i, t], buff_width[i, t], veh_type = scenes[start_time + t*self.delta_steps][veh_id]
                    buff_type[i, t] = self.veh_types[veh_type]
                    buff_vid[i, t] = veh_id

        return buff_x, buff_y, buff_heading, buff_v_x, buff_v_y, buff_length, buff_width, buff_type, buff_vid
    
    def agent_process(self, t0, scenes, ego_veh_id, neighbor_veh_ids, history_length, pred_length):

        # veh_ids = [veh_id for veh_id in scenes[t0-self.delta_steps].keys() if veh_id != ego_veh_id]
        veh_ids = neighbor_veh_ids
        # Flatten trajectories
        ego_buff_x, ego_buff_y,  ego_buff_heading, ego_buff_v_x, ego_buff_v_y, ego_buff_length, ego_buff_width, \
        ego_buff_type, ego_buff_vid = self.flatten_trajectory(scenes, start_time=t0-history_length,
                                                            time_length=history_length, veh_ids=[ego_veh_id])

        ego_buff_x_gt, ego_buff_y_gt,  ego_buff_heading_gt, ego_buff_v_x_gt, ego_buff_v_y_gt, ego_buff_length_gt, ego_buff_width_gt, \
        ego_buff_type_gt, ego_buff_vid_gt = self.flatten_trajectory(scenes, start_time=t0, 
                                                            time_length=pred_length, veh_ids=[ego_veh_id])
        

        buff_x, buff_y, buff_heading, buff_v_x, buff_v_y, buff_length, buff_width, buff_type, buff_vid = \
            self.flatten_trajectory(scenes, start_time=t0-history_length,
                                    time_length=history_length, veh_ids=veh_ids)

        buff_x_gt, buff_y_gt, buff_heading_gt, buff_v_x_gt, buff_v_y_gt, buff_length_gt, buff_width_gt, buff_type_gt, buff_vid_gt = \
            self.flatten_trajectory(scenes, start_time=t0, time_length=pred_length, veh_ids=veh_ids)

        ego_buff = {"ego_buff_x": ego_buff_x, "ego_buff_y": ego_buff_y, "ego_buff_heading": ego_buff_heading,
                    "ego_buff_v_x": ego_buff_v_x, "ego_buff_v_y": ego_buff_v_y, "ego_buff_length": ego_buff_length,
                    "ego_buff_width": ego_buff_width, "ego_buff_type": ego_buff_type, "ego_buff_vid": ego_buff_vid}
        neighbor_buff = {"buff_x": buff_x, "buff_y": buff_y, "buff_heading": buff_heading, "buff_v_x": buff_v_x,
                         "buff_v_y": buff_v_y, "buff_length": buff_length, "buff_width": buff_width,
                         "buff_type": buff_type, "buff_vid": buff_vid}
        ego_buff_gt = {"ego_buff_x": ego_buff_x_gt, "ego_buff_y": ego_buff_y_gt, "ego_buff_heading": ego_buff_heading_gt,
                    "ego_buff_v_x": ego_buff_v_x_gt, "ego_buff_v_y": ego_buff_v_y_gt, "ego_buff_length": ego_buff_length_gt,
                    "ego_buff_width": ego_buff_width_gt, "ego_buff_type": ego_buff_type_gt, "ego_buff_vid": ego_buff_vid_gt}
        neighbor_buff_gt = {"buff_x": buff_x_gt, "buff_y": buff_y_gt, "buff_heading": buff_heading_gt, "buff_v_x": buff_v_x_gt,
                         "buff_v_y": buff_v_y_gt, "buff_length": buff_length_gt, "buff_width": buff_width_gt,
                         "buff_type": buff_type_gt, "buff_vid": buff_vid_gt}

        ego, neighbors, neighbor_vid = self.NN_input_format(ego_buff, neighbor_buff)
        ego_gt, neighbors_gt, neighbor_vid_gt = self.NN_input_format(ego_buff_gt, neighbor_buff_gt)

        obj_trajs_data = np.concatenate([neighbors, ego[np.newaxis,:,:]], axis=0)
        obj_trajs_mask = np.isnan(obj_trajs_data[:, :, 0]) == False
        track_index_to_predict = np.arange(neighbors.shape[0])
        sdc_track_index = neighbors.shape[0]

        obj_types = np.concatenate([neighbor_buff["buff_type"], ego_buff["ego_buff_type"]], axis=0)
        obj_ids = np.concatenate([neighbor_buff["buff_vid"], ego_buff["ego_buff_vid"]], axis=0)

        center_gt_trajs = np.concatenate([neighbors_gt, ego_gt[np.newaxis,:,:]], axis=0)
        center_gt_trajs_mask = np.isnan(center_gt_trajs[:, :, 0]) == False
        # center_gt_trajs_mask = np.isnan(neighbors_gt[:, :, 0]) == False

        # sdc_gt_trajs_mask = np.isnan(ego_gt[:, 0]) == False

        agent_inputs = {
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            # 'track_index_to_predict': track_index_to_predict,
            'sdc_track_index': np.array([sdc_track_index]),
            'obj_types': obj_types,
            'obj_ids': obj_ids,
        }
        agent_outputs = {
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask
        }
        return agent_inputs, agent_outputs
    
    def map_process(self, map):
        
        lane_ctrs = {}
        lane_dir = {}
        lane_left_boundary, lane_right_boundary = {}, {}
        left_boundary_types, right_boundary_types = {}, {}
        lane_types, lane_speedlim = {}, {}
        point_dim = 12

        
        for lanelet in map.lanelets:
            lane_ctrs[lanelet.lanelet_id] = center_vertices = data_utils.resample_polyline(lanelet.center_vertices, self.lane_discretization)
            
            center_vertices_suc = np.roll(center_vertices, shift=-1, axis=0)
            center_vertices_suc[-1] = center_vertices[-1]
            diff = center_vertices_suc - center_vertices
            polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
            polyline_dir[-1] = polyline_dir[-2]
            lane_dir[lanelet.lanelet_id] = polyline_dir

            lane_left_boundary[lanelet.lanelet_id] = data_utils.resample_polyline_fixed_points(lanelet.left_vertices, len(center_vertices)) 
            lane_right_boundary[lanelet.lanelet_id] = data_utils.resample_polyline_fixed_points(lanelet.right_vertices, len(center_vertices))
            if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
                lane_types[lanelet.lanelet_id] = np.array([[0] for _ in range(len(center_vertices))])
            elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
                lane_types[lanelet.lanelet_id] = np.array([[1] for _ in range(len(center_vertices))])
            elif LaneletType.BICYCLE_LANE in lanelet.lanelet_type or LaneletType.CROSSWALK in lanelet.lanelet_type:
                lane_types[lanelet.lanelet_id] = np.array([[2] for _ in range(len(center_vertices))])
            else: 
                lane_types[lanelet.lanelet_id] = np.array([[3] for _ in range(len(center_vertices))])
            if lanelet.line_marking_left_vertices == LineMarking.DASHED or lanelet.line_marking_left_vertices == LineMarking.BROAD_DASHED:
                left_boundary_types[lanelet.lanelet_id] = np.array([[2] for _ in range(len(center_vertices))])
            elif lanelet.line_marking_left_vertices == LineMarking.SOLID:
                left_boundary_types[lanelet.lanelet_id] = np.array([[3] for _ in range(len(center_vertices))])
            elif lanelet.line_marking_left_vertices == LineMarking.NO_MARKING:
                left_boundary_types[lanelet.lanelet_id] = np.array([[1] for _ in range(len(center_vertices))])
            elif lanelet.line_marking_left_vertices == LineMarking.UNKNOWN:
                left_boundary_types[lanelet.lanelet_id] = np.array([[0] for _ in range(len(center_vertices))])
            else: 
                raise Exception("Unknown left line marking type {}".format(lanelet.line_marking_left_vertices))  
            if lanelet.line_marking_right_vertices == LineMarking.DASHED or lanelet.line_marking_right_vertices == LineMarking.BROAD_DASHED:
                right_boundary_types[lanelet.lanelet_id] = np.array([[2] for _ in range(len(center_vertices))])
            elif lanelet.line_marking_right_vertices == LineMarking.SOLID:
                right_boundary_types[lanelet.lanelet_id] = np.array([[3] for _ in range(len(center_vertices))])
            elif lanelet.line_marking_right_vertices == LineMarking.NO_MARKING:
                right_boundary_types[lanelet.lanelet_id] = np.array([[1] for _ in range(len(center_vertices))])
            elif lanelet.line_marking_right_vertices == LineMarking.UNKNOWN:
                right_boundary_types[lanelet.lanelet_id] = np.array([[0] for _ in range(len(center_vertices))])
            else: 
                raise Exception("Unknown right line marking type {}".format(lanelet.line_marking_right_vertices))
            speedlim = 25.0 if LaneletType.INTERSTATE in lanelet.lanelet_type else 13.89
            lane_speedlim[lanelet.lanelet_id] = np.array([[speedlim] for _ in range(len(center_vertices))])
        lane_ids = [lanelet.lanelet_id for lanelet in map.lanelets]    
        lane_id2ind = {lanelet.lanelet_id: i for i, lanelet in enumerate(map.lanelets)}
        list_map = [np.concatenate([lane_ctrs[lane_id], #0,1
            lane_dir[lane_id], #2,3
            lane_left_boundary[lane_id], #4,5
            lane_right_boundary[lane_id], #6,7
            lane_speedlim[lane_id], # 8
            lane_types[lane_id], # 9
            left_boundary_types[lane_id], # 10
            right_boundary_types[lane_id], #11
            
        ], -1) for lane_id in lane_id2ind.keys()]

        ret_polylines = []
        ret_polylines_lane_ids = []
        ret_polylines_mask = []
        polyline_indices  = {}
        num_points_each_polyline = self.num_points_each_polyline
        
        lane_suc_edges, lane_pre_edges, lane_left_edges, lane_right_edges = [], [], [], []
        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for lane_id, lane_polyline in zip(lane_ids, list_map):
            polyline_indices[lane_id] = []
            for idx in range(0, len(lane_polyline), num_points_each_polyline):
                if lane_polyline[idx: idx + num_points_each_polyline].shape[0] == 1 and idx != 0:
                    continue
                append_single_polyline(lane_polyline[idx: idx + num_points_each_polyline])
                ret_polylines_lane_ids.append(lane_id)
                polyline_indices[lane_id].append(len(ret_polylines) - 1)
            for idx in range(len(polyline_indices[lane_id])):
                if idx > 0:
                    lane_pre_edges.append((polyline_indices[lane_id][idx], polyline_indices[lane_id][idx - 1]))
                if idx < len(polyline_indices[lane_id]) - 1:
                    lane_suc_edges.append((polyline_indices[lane_id][idx], polyline_indices[lane_id][idx + 1]))
        
        for i, lane in enumerate(map.lanelets):
            lane_id = lane.lanelet_id
            # sucessor lane 
            for suc_lane_id in lane.successor:
                lane_suc_edges.append([polyline_indices[lane_id][-1], polyline_indices[suc_lane_id][0]])
            # predecessor lane
            for pre_lane_id in lane.predecessor:
                lane_pre_edges.append([polyline_indices[lane_id][0], polyline_indices[pre_lane_id][-1]])
            # left lane
            if lane.adj_left is not None and lane.adj_left_same_direction:
                # lane_left_edges.append([i, lane_id2ind[lane.adj_left]])
                left_lane_id = lane.adj_left
                pj_start = -1
                for pi, polyline_idx in enumerate(polyline_indices[lane_id]):
                    
                    for pj, left_polyline_idx in enumerate(polyline_indices[left_lane_id]):
                        if pj <= pj_start:
                            continue
                        lane_left_edges.append((polyline_idx, left_polyline_idx))
                        if pi < len(polyline_indices[lane_id]) - 1:
                            pj_start = pj
                            break
            # right lane
            if lane.adj_right is not None and lane.adj_right_same_direction:
                # lane_right_edges.append([i, lane_id2ind[lane.adj_right]])
                right_lane_id = lane.adj_right
                pj_start = -1
                for pi, polyline_idx in enumerate(polyline_indices[lane_id]): 
                    for pj, right_polyline_idx in enumerate(polyline_indices[right_lane_id]):
                        if pj <= pj_start:
                            continue
                        lane_right_edges.append((polyline_idx, right_polyline_idx))
                        if pi < len(polyline_indices[lane_id]) - 1:
                            pj_start = pj
                            break

        lane_suc_edges = np.array(lane_suc_edges)
        lane_pre_edges = np.array(lane_pre_edges)
        lane_left_edges = np.array(lane_left_edges)
        lane_right_edges = np.array(lane_right_edges)    
        if len(lane_left_edges) == 0:
            lane_left_edges = np.zeros((0, 2), dtype=np.int64)
        if len(lane_right_edges) == 0:
            lane_right_edges = np.zeros((0, 2), dtype=np.int64)

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        map_inputs = {
            'map_polylines': ret_polylines,
            'map_polylines_ids': ret_polylines_lane_ids,
            'map_polylines_mask': ret_polylines_mask > 0,
            'map_polyline_suc_edges': lane_suc_edges,
            'map_polyline_pre_edges': lane_pre_edges,
            'map_polyline_left_edges': lane_left_edges,
            'map_polyline_right_edges': lane_right_edges,
        }
        return map_inputs

    def NN_input_format(self, ego_buff, neighbor_buff):
        """
        Format the data
        :return:
        """

        ego_x, ego_y, ego_heading, ego_v_x, ego_v_y, ego_length, ego_width, ego_type = \
            ego_buff["ego_buff_x"], ego_buff["ego_buff_y"], ego_buff["ego_buff_heading"], \
            ego_buff["ego_buff_v_x"], ego_buff["ego_buff_v_y"], \
            ego_buff["ego_buff_length"], ego_buff["ego_buff_width"], ego_buff["ego_buff_type"]
        ego_cos_heading, ego_sin_heading = np.cos(ego_heading), np.sin(ego_heading)
        ego = np.stack([ego_x, ego_y, ego_cos_heading, ego_sin_heading,
                        ego_v_x, ego_v_y, ego_length, ego_width, ego_type], axis=-1)[0]

        neighbor_x, neighbor_y, neighbor_heading, neighbor_v_x, neighbor_v_y, \
        neighbor_length, neighbor_width, neighbor_type, neighbor_vid = \
            neighbor_buff["buff_x"], neighbor_buff["buff_y"], neighbor_buff["buff_heading"], \
            neighbor_buff["buff_v_x"], neighbor_buff["buff_v_y"], \
            neighbor_buff["buff_length"], neighbor_buff["buff_width"], \
            neighbor_buff["buff_type"], neighbor_buff["buff_vid"]
        neighbor_cos_heading, neighbor_sin_heading = np.cos(neighbor_heading), np.sin(neighbor_heading)
        neighbors = np.stack([neighbor_x, neighbor_y, neighbor_cos_heading, neighbor_sin_heading, neighbor_v_x, neighbor_v_y, neighbor_length,
                             neighbor_width, neighbor_type], axis=-1)
        return ego, neighbors, neighbor_vid
     
    def prepare_input_data(self, t0, history_length, pred_length, scenes, map, ego_veh_id, neighbor_veh_ids):
        """

        :param num_neighbors:
        :param history_length:
        :param traj_pool:
        :param ego_veh_id:
        :param local_origin_xyh:
        :param prepare_gt: if True, we will not impute traj and not generate map info since we just need the traj of ego and neighbors
        :return:
        """

        agent_inputs, agent_outputs = self.agent_process(t0, scenes, ego_veh_id, neighbor_veh_ids, history_length, pred_length)
        
        # Generate map features for all agents. [ego_map, neighbor1_map, neighbor2_map, ...]
        map_inputs= self.map_process(map)
        
        return agent_inputs, agent_outputs, map_inputs
