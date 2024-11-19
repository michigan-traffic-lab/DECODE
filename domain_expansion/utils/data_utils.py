import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate

# modified based on https://commonroad.in.tum.de/route-planner
def resample_polyline(polyline, step = 0.5):
    """Resamples the input polyline with the specified step size.

    The distances between each pair of consecutive vertices are examined. If it is larger than the step size,
    a new sample is added in between.

    :param polyline: polyline with 2D points
    :param step: minimum distance between each consecutive pairs of vertices
    :return: resampled polyline
    """
    if len(polyline) < 2:
        return polyline
    polyline_new = [polyline[0]]

    current_idx = 0
    current_position = step
    current_distance = np.linalg.norm([polyline[0][0] - polyline[1][0], polyline[0][1] - polyline[1][1]])

    # iterate through all pairs of vertices of the polyline
    while current_idx < len(polyline) - 1:
        if current_position <= current_distance:
            # add new sample and increase current position
            ratio = current_position / current_distance
            polyline_new.append([(1 - ratio) * polyline[current_idx][0] +
                                ratio * polyline[current_idx + 1][0], 
                                (1 - ratio) * polyline[current_idx][1] +
                                ratio * polyline[current_idx + 1][1]])
            current_position += step

        else:
            # move on to the next pair of vertices
            current_idx += 1
            # if we are out of vertices, then break
            if current_idx >= len(polyline) - 1:
                break
            # deduct the distance of previous vertices from the position
            current_position = current_position - current_distance
            # compute new distances of vertices
            current_distance = np.linalg.norm([polyline[current_idx + 1][0] - polyline[current_idx][0], 
                                               polyline[current_idx + 1][1] - polyline[current_idx][1]])

    # add the last vertex
    if len(polyline_new) == 1:
        polyline_new.append(polyline[-1])

    return np.array(polyline_new)

def resample_polyline_fixed_points(polyline, num_points):
    """Resamples the input polyline with a fixed number of evenly distributed points.

    :param polyline: polyline with 2D points
    :param num_points: number of points in the resampled polyline
    :return: resampled polyline
    """
    if len(polyline) < 2:
        return polyline

    # Calculate total length of the polyline
    total_length = sum(np.linalg.norm([polyline[i+1][0] - polyline[i][0], 
                                       polyline[i+1][1] - polyline[i][1]]) 
                       for i in range(len(polyline) - 1))
    
    # Calculate the distance between each point in the resampled polyline
    step = total_length / (num_points - 1)

    # Initialize the resampled polyline
    resampled_polyline = [polyline[0]]

    # Variables to track the current segment and position along the polyline
    current_idx = 0
    current_position = step

    while len(resampled_polyline) < num_points - 1:
        # Distance of current segment
        segment_length = np.linalg.norm([polyline[current_idx + 1][0] - polyline[current_idx][0],
                                         polyline[current_idx + 1][1] - polyline[current_idx][1]])

        if current_position <= segment_length:
            # Interpolate and add a new point
            ratio = current_position / segment_length
            new_point = [(1 - ratio) * polyline[current_idx][0] + ratio * polyline[current_idx + 1][0],
                         (1 - ratio) * polyline[current_idx][1] + ratio * polyline[current_idx + 1][1]]
            resampled_polyline.append(new_point)
            current_position += step
        else:
            # Move to the next segment
            current_position -= segment_length
            current_idx += 1

    # Add the last vertex
    resampled_polyline.append(polyline[-1])

    return np.array(resampled_polyline)

def wrap_to_pi(theta):
    """
    Wraps an angle in radians to the range of [-pi, pi)
    :param theta:
    :return:
    """
    if isinstance(theta, np.ndarray):
        valid = ~np.isnan(theta)
        output = np.full_like(theta, np.nan)
        output[valid] = (theta[valid] + np.pi) % (2*np.pi) - np.pi
        return output
    else:
        if np.isnan(theta):
            return theta
        else:
            return (theta + np.pi) % (2*np.pi) - np.pi

def compute_direction_diff(ego_theta, target_theta):
    delta = np.abs(ego_theta - target_theta)
    delta = np.where(delta > np.pi, 2*np.pi - delta, delta)

    return delta

def depth_first_search(cur_lane, lanes, dist=0, threshold=100):
    """
    Perform depth first search over lane graph up to the threshold.
    Args:
        cur_lane: Starting lane_id
        lanes: raw lane data
        dist: Distance of the current path
        threshold: Threshold after which to stop the search
    Returns:
        lanes_to_return (list of list of integers): List of sequence of lane ids
    """
    if dist > threshold:
        return [[cur_lane]]
    else:
        traversed_lanes = []
        child_lanes = lanes[cur_lane].exit_lanes

        if child_lanes:
            for child in child_lanes:
                centerline = np.array([(map_point.x, map_point.y, map_point.z) for map_point in lanes[child].polyline])
                cl_length = centerline.shape[0]
                curr_lane_ids = depth_first_search(child, lanes, dist + cl_length, threshold)
                traversed_lanes.extend(curr_lane_ids)

        if len(traversed_lanes) == 0:
            return [[cur_lane]]

        lanes_to_return = []

        for lane_seq in traversed_lanes:
            lanes_to_return.append([cur_lane] + lane_seq)

        return lanes_to_return

def is_overlapping_lane_seq(lane_seq1, lane_seq2):
    """
    Check if the 2 lane sequences are overlapping.
    Args:
        lane_seq1: list of lane ids
        lane_seq2: list of lane ids
    Returns:
        bool, True if the lane sequences overlap
    """

    if lane_seq2[1:] == lane_seq1[1:]:
        return True
    elif set(lane_seq2) <= set(lane_seq1):
        return True

    return False

def remove_overlapping_lane_seq(lane_seqs):
    """
    Remove lane sequences which are overlapping to some extent
    Args:
        lane_seqs (list of list of integers): List of list of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
    Returns:
        List of sequence of lane ids (e.g. ``[[12345, 12346, 12347], [12345, 12348]]``)
    """
    redundant_lane_idx = set()

    for i in range(len(lane_seqs)):
        for j in range(len(lane_seqs)):
            if i in redundant_lane_idx or i == j:
                continue
            if is_overlapping_lane_seq(lane_seqs[i], lane_seqs[j]):
                redundant_lane_idx.add(j)

    unique_lane_seqs = [lane_seqs[i] for i in range(len(lane_seqs)) if i not in redundant_lane_idx]

    return unique_lane_seqs

def polygon_completion(polygon):
    polyline_x = []
    polyline_y = []
    polyline_z = []

    for i in range(len(polygon)):
        if i+1 < len(polygon):
            next = i+1
        else:
            next = 0

        dist_x = polygon[next].x - polygon[i].x
        dist_y = polygon[next].y - polygon[i].y
        dist_z = polygon[next].z - polygon[i].z
        dist = np.linalg.norm([dist_x, dist_y, dist_z])
        interp_num = np.ceil(dist)*2
        interp_index = np.arange(2+interp_num)
        point_x = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].x, polygon[next].x]).tolist()
        point_y = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].y, polygon[next].y]).tolist()
        point_z = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].z, polygon[next].z]).tolist()
        polyline_x.extend(point_x[:-1])
        polyline_y.extend(point_y[:-1])
        polyline_z.extend(point_z[:-1])

    polyline_x, polyline_y, polyline_z = np.array(polyline_x), np.array(polyline_y), np.array(polyline_z)
    # polyline_heading = wrap_to_pi(np.arctan2(polyline_y[1:]-polyline_y[:-1], polyline_x[1:]-polyline_x[:-1]))
    # polyline_heading = np.insert(polyline_heading, -1, polyline_heading[-1])

    return np.stack([polyline_x, polyline_y, polyline_z], axis=1)

def get_polylines(lines):
    polylines = {}

    for line in lines.keys():
        polyline = np.array([(map_point.x, map_point.y, map_point.z) for map_point in lines[line].polyline])
        if len(polyline) > 1:
            direction = wrap_to_pi(np.arctan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0]))
            direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
        else:
            direction = np.array([0])[:, np.newaxis]
        polylines[line] = np.concatenate([polyline, direction], axis=-1)

    return polylines

def find_reference_lanes_by_radius(agent_traj, lanes, max_radius):
    curr_lane_ids = {}
    # while len(curr_lane_ids) < 1:
    for lane in lanes.keys():
        distance_to_ego = np.linalg.norm(agent_traj[-1, :2] - lanes[lane][:, :2], axis=-1)
        for i, j in zip(distance_to_ego, range(distance_to_ego.shape[0])):
            if i <= max_radius:
                curr_lane_ids[lane] = j
                break
    return curr_lane_ids


def find_reference_lanes(agent_type, agent_traj, lanes):
    curr_lane_ids = {}

    if agent_type == 2:
        distance_threshold = 5

        while len(curr_lane_ids) < 1:
            for lane in lanes.keys():
                if lanes[lane].shape[0] > 1:
                    distance_to_agent = LineString(lanes[lane][:, :2]).distance(Point(agent_traj[-1, :2]))
                    if distance_to_agent < distance_threshold:
                        curr_lane_ids[lane] = 0

            distance_threshold += 5
    else:
        distance_threshold = 3.5
        direction_threshold = 10

        while len(curr_lane_ids) < 1:
            for lane in lanes.keys():
                distance_to_ego = np.linalg.norm(agent_traj[-1, :2] - lanes[lane][:, :2], axis=-1)
                direction_to_ego = compute_direction_diff(agent_traj[-1, 3], lanes[lane][:, -1])
                for i, j, k in zip(distance_to_ego, direction_to_ego, range(distance_to_ego.shape[0])):
                    if i <= distance_threshold and j <= np.radians(direction_threshold):
                        curr_lane_ids[lane] = k
                        break

            distance_threshold += 3.5
            direction_threshold += 10

    return curr_lane_ids

def find_neighbor_lanes(curr_lane_ids, traj, lanes, lane_polylines):
    neighbor_lane_ids = {}

    for curr_lane, start in curr_lane_ids.items():
        left_lanes = lanes[curr_lane].left_neighbors
        right_lanes = lanes[curr_lane].right_neighbors
        left_lane = None
        right_lane = None
        curr_index = start

        for l_lane in left_lanes:
            if l_lane.self_start_index <= curr_index <= l_lane.self_end_index and not l_lane.feature_id in curr_lane_ids:
                left_lane = l_lane

        for r_lane in right_lanes:
            if r_lane.self_start_index <= curr_index <= r_lane.self_end_index and not r_lane.feature_id in curr_lane_ids:
                right_lane = r_lane

        if left_lane is not None:
            left_polyline = lane_polylines[left_lane.feature_id]
            start = np.argmin(np.linalg.norm(traj[-1, :2] - left_polyline[:, :2], axis=-1))
            neighbor_lane_ids[left_lane.feature_id] = start

        if right_lane is not None:
            right_polyline = lane_polylines[right_lane.feature_id]
            start = np.argmin(np.linalg.norm(traj[-1, :2] - right_polyline[:, :2], axis=-1))
            neighbor_lane_ids[right_lane.feature_id] = start

    return neighbor_lane_ids

def find_neareast_point(curr_point, line):
    distance_to_curr_point = np.linalg.norm(curr_point[np.newaxis, :2] - line[:, :2], axis=-1)
    neareast_point = line[np.argmin(distance_to_curr_point)]

    return neareast_point



def find_map_waypoint(pos, polylines):
    waypoint = [-1, -1, 1e9, 1e9]
    direction_threshold = 10

    for id, polyline in polylines.items():
        distance_to_gt = np.linalg.norm(pos[np.newaxis, :2] - polyline[:, :2], axis=-1)
        direction_to_gt = compute_direction_diff(pos[np.newaxis, 2], polyline[:, 2])
        for i, j, k in zip(range(polyline.shape[0]), distance_to_gt, direction_to_gt):
            if j < waypoint[2] and k <= np.radians(direction_threshold):
                waypoint = [id, i, j, k]

    lane_id = waypoint[0]
    waypoint_id = waypoint[1]

    if lane_id > 0:
        return lane_id, waypoint_id
    else:
        return None, None

def find_route(traj, timestep, cur_pos, map_lanes, map_crosswalks, map_signals):
    lane_polylines = get_polylines(map_lanes)
    end_lane, end_point = find_map_waypoint(np.array((traj[-1].center_x, traj[-1].center_y, traj[-1].heading)), lane_polylines)
    start_lane, start_point = find_map_waypoint(np.array((traj[0].center_x, traj[0].center_y, traj[0].heading)), lane_polylines)
    cur_lane, _ = find_map_waypoint(cur_pos, lane_polylines)

    path_waypoints = []
    for t in range(0, len(traj), 10):
        lane, point = find_map_waypoint(np.array((traj[t].center_x, traj[t].center_y, traj[t].heading)), lane_polylines)
        path_waypoints.append(lane_polylines[lane][point])

    before_waypoints = []
    if start_point < 40:
        if map_lanes[start_lane].entry_lanes:
            lane = map_lanes[start_lane].entry_lanes[0]
            for waypoint in lane_polylines[lane]:
                before_waypoints.append(waypoint)
    for waypoint in lane_polylines[start_lane][:start_point]:
        before_waypoints.append(waypoint)

    after_waypoints = []
    for waypoint in lane_polylines[end_lane][end_point:]:
        after_waypoints.append(waypoint)
    if len(after_waypoints) < 40:
        if map_lanes[end_lane].exit_lanes:
            lane = map_lanes[end_lane].exit_lanes[0]
            for waypoint in lane_polylines[lane]:
                after_waypoints.append(waypoint)

    waypoints = np.concatenate([before_waypoints[::5], path_waypoints, after_waypoints[::5]], axis=0)

    # generate smooth route
    tx, ty, tyaw, tc, _ = generate_target_course(waypoints[:, 0], waypoints[:, 1])
    ref_line = np.column_stack([tx, ty, tyaw, tc])

    # get reference path at current timestep
    current_location = np.argmin(np.linalg.norm(ref_line[:, :2] - cur_pos[np.newaxis, :2], axis=-1))
    start_index = np.max([current_location-200, 0])
    ref_line = ref_line[start_index:start_index+1200]

    # add speed limit, crosswalk, and traffic signal info to ref route
    line_info = np.zeros(shape=(ref_line.shape[0], 1))
    speed_limit = map_lanes[cur_lane].speed_limit_mph / 2.237
    ref_line = np.concatenate([ref_line, line_info], axis=-1)
    crosswalks = [Polygon([(point.x, point.y) for point in crosswalk.polygon]) for _, crosswalk in map_crosswalks.items()]
    signals = [Point([signal.stop_point.x, signal.stop_point.y]) for signal in map_signals[timestep].lane_states if signal.state in [1, 4, 7]]

    for i in range(ref_line.shape[0]):
        if any([Point(ref_line[i, :2]).distance(signal) < 0.2 for signal in signals]):
            ref_line[i, 4] = 0 # red light
        elif any([crosswalk.contains(Point(ref_line[i, :2])) for crosswalk in crosswalks]):
            ref_line[i, 4] = 1 # crosswalk
        else:
            ref_line[i, 4] = speed_limit

    return ref_line

def imputer(traj):
    x, y, v_x, v_y, theta = traj[:, 0], traj[:, 1], traj[:, 3], traj[:, 4], traj[:, 2]

    if np.any(x==0):
        for i in reversed(range(traj.shape[0])):
            if x[i] == 0:
                v_x[i] = v_x[i+1]
                v_y[i] = v_y[i+1]
                x[i] = x[i+1] - v_x[i]*0.1
                y[i] = y[i+1] - v_y[i]*0.1
                theta[i] = theta[i+1]
        return np.column_stack((x, y, theta, v_x, v_y))
    else:
        return np.column_stack((x, y, theta, v_x, v_y))

def agent_norm(traj, center, angle):
    raw_traj = copy.deepcopy(traj)

    # x,y
    line = LineString(traj[:, :2])
    line_offset = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
    line_rotate = rotate(line_offset, -angle, origin=(0, 0), use_radians=True)
    line_rotate = np.array(line_rotate.coords)
    line_rotate[np.isnan(raw_traj[:, :2])] = np.nan
    # heading
    heading = wrap_to_pi(traj[:, 2] - angle)
    heading[np.isnan(raw_traj[:, 2])] = np.nan

    if traj.shape[-1] > 3:
        velocity_x = traj[:, 3] * np.cos(angle) + traj[:, 4] * np.sin(angle)
        velocity_x[np.isnan(raw_traj[:, 4])] = np.nan
        velocity_y = traj[:, 4] * np.cos(angle) - traj[:, 3] * np.sin(angle)
        velocity_y[np.isnan(raw_traj[:, 4])] = np.nan
        return np.column_stack((line_rotate, heading, velocity_x, velocity_y))
    else:
        return  np.column_stack((line_rotate, heading))


def transform_torch(ctrs_local, sys_src=None, sys_des=None): 
    if sys_src is None:
        ctrs_global_x = ctrs_local[..., 0]
        ctrs_global_y = ctrs_local[..., 1]
        if ctrs_local.shape[-1] == 3:
            ctrs_global_h = ctrs_local[..., 2]
        elif ctrs_local.shape[-1] == 4:
            ctrs_global_cosh = ctrs_local[..., 2]
            ctrs_global_sinh = ctrs_local[..., 3]
    else:
        # convert ctrs in sys_src to global sys based on sys
        ctrs_local_x = ctrs_local[..., 0]
        ctrs_local_y = ctrs_local[..., 1]
        

        src_centers_x = sys_src[..., 0]
        src_centers_y = sys_src[..., 1]
        src_centers_h = sys_src[..., 2]

        src_cos = torch.cos(sys_src[..., 2])
        src_sin = torch.sin(sys_src[..., 2])

        ctrs_global_x = src_centers_x + ctrs_local_x * src_cos - ctrs_local_y * src_sin
        ctrs_global_y = src_centers_y + ctrs_local_x * src_sin + ctrs_local_y * src_cos
        if ctrs_local.shape[-1] == 3:
            ctrs_local_h = ctrs_local[..., 2]
            # heading
            ctrs_global_h = (ctrs_local_h + src_centers_h) % (2 * torch.pi)
        elif ctrs_local.shape[-1] == 4:
            ctrs_local_cosh = ctrs_local[..., 2]
            ctrs_local_sinh = ctrs_local[..., 3]
            ctrs_global_cosh = ctrs_local_cosh * src_cos - ctrs_local_sinh * src_sin
            ctrs_global_sinh = ctrs_local_cosh * src_sin + ctrs_local_sinh * src_cos

    if sys_des is None:
        if ctrs_local.shape[-1] == 3:
            return torch.stack([ctrs_global_x, ctrs_global_y, ctrs_global_h], dim=-1)
        elif ctrs_local.shape[-1] == 4:
            return torch.stack([ctrs_global_x, ctrs_global_y, ctrs_global_cosh, ctrs_global_sinh], dim=-1)
        else:
            return torch.stack([ctrs_global_x, ctrs_global_y], dim=-1)
    # convert ctrs in global sys to sys_des
    des_centers_x = sys_des[..., 0]
    des_centers_y = sys_des[..., 1]
    des_centers_h = sys_des[..., 2]
    des_cos = torch.cos(des_centers_h)
    des_sin = torch.sin(des_centers_h)

    ctrs_des_x = (ctrs_global_x - des_centers_x) * des_cos + (ctrs_global_y - des_centers_y) * des_sin
    ctrs_des_y = -(ctrs_global_x - des_centers_x) * des_sin + (ctrs_global_y - des_centers_y) * des_cos
    if ctrs_local.shape[-1] == 3:
        # heading
        ctrs_des_h = (ctrs_global_h - des_centers_h) % (2 * torch.pi)
        return torch.stack([ctrs_des_x, ctrs_des_y, ctrs_des_h], dim=-1)
    elif ctrs_local.shape[-1] == 4:
        ctrs_des_cosh = ctrs_global_cosh * des_cos + ctrs_global_sinh * des_sin
        ctrs_des_sinh = -ctrs_global_cosh * des_sin + ctrs_global_sinh * des_cos
        return torch.stack([ctrs_des_x, ctrs_des_y, ctrs_des_cosh, ctrs_des_sinh], dim=-1)
    return torch.stack([ctrs_des_x, ctrs_des_y], dim=-1)

def transform_torch2(delta_ctrs_local, sys_src=None, sys_des=None): 
    if sys_src is None:
        delta_ctrs_global_x = delta_ctrs_local[..., 0]
        delta_ctrs_global_y = delta_ctrs_local[..., 1]
    else:
        # convert ctrs in sys_src to global sys based on sys
        delta_ctrs_local_x = delta_ctrs_local[..., 0]
        delta_ctrs_local_y = delta_ctrs_local[..., 1]
        
        # src_centers_x = sys_src[..., 0]
        # src_centers_y = sys_src[..., 1]
        # src_centers_h = sys_src[..., 2]

        src_cos = torch.cos(sys_src[..., 2])
        src_sin = torch.sin(sys_src[..., 2])

        delta_ctrs_global_x = delta_ctrs_local_x * src_cos - delta_ctrs_local_y * src_sin
        delta_ctrs_global_y = delta_ctrs_local_x * src_sin + delta_ctrs_local_y * src_cos
        
    if sys_des is None:
        return torch.stack([delta_ctrs_global_x, delta_ctrs_global_y], dim=-1)
    # convert ctrs in global sys to sys_des
    # des_centers_x = sys_des[..., 0]
    # des_centers_y = sys_des[..., 1]
    des_centers_h = sys_des[..., 2]
    des_cos = torch.cos(des_centers_h)
    des_sin = torch.sin(des_centers_h)

    delta_ctrs_des_x = delta_ctrs_global_x * des_cos + delta_ctrs_global_y  * des_sin
    delta_ctrs_des_y = -delta_ctrs_global_x * des_sin + delta_ctrs_global_y * des_cos
    
    return torch.stack([delta_ctrs_des_x, delta_ctrs_des_y], dim=-1)    
    
def map_norm(map_line, center, angle, with_signal=True):
    # x, y
    self_line = LineString(map_line[:, 0:2])
    self_line = affine_transform(self_line, [1, 0, 0, 1, -center[0], -center[1]])
    self_line = rotate(self_line, -angle, origin=(0, 0), use_radians=True)
    self_line = np.array(self_line.coords)
    self_line[np.isnan(map_line[:, 0:2])] = np.nan

    # left boundary x, y
    left_line = LineString(map_line[:, 2:4])
    left_line = affine_transform(left_line, [1, 0, 0, 1, -center[0], -center[1]])
    left_line = rotate(left_line, -angle, origin=(0, 0), use_radians=True)
    left_line = np.array(left_line.coords)
    left_line[np.isnan(map_line[:, 2:4])] = np.nan

    # right boundary x, y
    right_line = LineString(map_line[:, 4:6])
    right_line = affine_transform(right_line, [1, 0, 0, 1, -center[0], -center[1]])
    right_line = rotate(right_line, -angle, origin=(0, 0), use_radians=True)
    right_line = np.array(right_line.coords)
    right_line[np.isnan(map_line[:, 4:6])] = np.nan

    return np.column_stack((self_line, left_line, right_line))
    
def ref_line_norm(ref_line, center, angle):
    xy = LineString(ref_line[:, 0:2])
    xy = affine_transform(xy, [1, 0, 0, 1, -center[0], -center[1]])
    xy = rotate(xy, -angle, origin=(0, 0), use_radians=True)
    yaw = wrap_to_pi(ref_line[:, 2] - angle)
    c = ref_line[:, 3]
    info = ref_line[:, 4]

    return np.column_stack((xy.coords, yaw, c, info))

def nan_intep_1d(y):
    "interplate a 1d np array with nan values"

    def _nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    if np.isnan(y).all():
        return y

    if not np.isnan(y).any():
        return y

    nans, x = _nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y

def nan_intep_2d(y, axis):
    "interplate a 2d np array with nan values"

    if np.isnan(y).all():
        return y

    if not np.isnan(y).any():
        return y

    h, w = y.shape

    if axis == 0:
        for i in range(w):
            y[:, i] = nan_intep_1d(y[:, i])
    elif axis == 1:
        for i in range(h):
            y[i, :] = nan_intep_1d(y[i, :])

    return y

def nan_intep_1d_heading(y):

    def _nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    if np.isnan(y).all():
        return y

    if not np.isnan(y).any():
        return y

    nans, x = _nan_helper(y)
    y[nans] = wrap_to_pi(np.interp(x(nans), x(~nans), np.unwrap(y[~nans])))

    return y

def nan_intep_2d_heading(y, axis):
    """
    interpolate a 2d np array of heading with nan values
    One dimensional linear interpolation for monotonically increasing sample points where points first are unwrapped,
    secondly interpolated and finally bounded within the specified period.

    :param y:
    :param axis:
    :return:
    """

    if np.isnan(y).all():
        return y

    if not np.isnan(y).any():
        return y

    h, w = y.shape

    if axis == 0:
        for i in range(w):
            y[:, i] = nan_intep_1d_heading(y[:, i])
    elif axis == 1:
        for i in range(h):
            y[i, :] = nan_intep_1d_heading(y[i, :])

    return y


def select_future(plans, predictions, scores):
    best_mode = torch.argmax(scores, dim=-1)
    plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
    prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])

    return plan, prediction


def transform(out_coord, ori_coord, include_curr=False):
    """

    Args:
        out_coord: 4d vector x, y, z, heading
        ori_coord: x, y, z, heading
        include_curr:

    Returns:

    """
    ori_x, ori_y, ori_z, ori_heading = ori_coord
    # x, y
    line = LineString(out_coord[:, :2])
    line = rotate(line, ori_heading, origin=(0, 0), use_radians=True)
    line = affine_transform(line, [1, 0, 0, 1, ori_x, ori_y])
    # z
    z = out_coord[:, 2, np.newaxis] + ori_z

    output = np.column_stack((np.array(line.coords), np.array(z)))
    if include_curr:
        line = np.insert(output, 0, ori_coord[:-1], axis=0)
    else:
        line = output

    return line

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((
            cosa,  sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

def feature_collate_fn(batch):
    return_batch = np.concatenate(batch, axis=0) 
    return torch.from_numpy(return_batch)


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        if key != 'map_graph':
            return_batch[key] = [x[key] for x in batch]
    batch_size = len(batch)
    map_graphs = []
    for i in range(batch_size):
        map = batch[i]['map_graph']
        graph = {
            "map_lanes":map['map_lanes'],
            "pre_edges":{k:v for k, v in map['pre_edges'].items()},
            "suc_edges":{k:v for k, v in map['suc_edges'].items()},
            "left_edges":map['left_edges'],
            "right_edges":map['right_edges']
            
        }
        if "map_signals" in map:
            graph["map_signals"] = map["map_signals"]
        if "map_lanes_ctrs" in map:
            graph["map_lanes_ctrs"] = map['map_lanes_ctrs']
            graph["map_lanes_mask"] = map['map_lanes_mask']
        map_graphs.append(graph)
    return_batch['map_graphs'] = map_graphs

    mask_batch = [np.ones(len(x), dtype=np.int32) * i for i, x in enumerate(return_batch['ground_truth'])]
    return_batch['mask_batch'] = torch.from_numpy(np.concatenate(mask_batch, axis=0))
    batch_indices = []
    counts = 0
    for _, x in enumerate(return_batch['ground_truth']):
        batch_indices.append(torch.from_numpy(np.arange(counts, counts+len(x))))
        counts += len(x)
    return_batch['batch_indices'] = batch_indices
    
    return return_batch    

def collate_fn2(batch):
    for data in batch:
        num_agents, num_timesteps, num_candidates, _ = data['candidates'].shape
        if num_candidates > 64:
            agt_idcs = np.arange(num_agents)[:, None, None]
            t_idcs = np.arange(num_timesteps)[None, :, None]
            ground_truth_candidates = data['candidates'][agt_idcs, t_idcs, data['gt_targets']]
            distances = np.sqrt(np.sum((data['candidates'] - ground_truth_candidates) ** 2, axis=-1))
            sorted_indices = np.argsort(distances, axis=-1)
            selected_indices = sorted_indices[:, :, :num_candidates // 2]
            selected_candidates = data['candidates'][agt_idcs, t_idcs, selected_indices]
            selected_candidates_inds = data['candidates_inds'][agt_idcs, t_idcs, selected_indices]
            selected_gt_targets = np.zeros_like(data['gt_targets'])
            data['candidates'] = selected_candidates
            data['candidates_inds'] = selected_candidates_inds
            data['gt_targets'] = selected_gt_targets
    
    return collate_fn(batch)    

def reduced_collate_fn(batch):
    for data in batch:
        num_agents, num_timesteps, num_candidates, _ = data['candidates'].shape
        agt_idcs = np.arange(num_agents)[:, None, None]
        t_idcs = np.arange(num_timesteps)[None, :, None]
        ground_truth_candidates = data['candidates'][agt_idcs, t_idcs, data['gt_targets']]
        distances = np.sqrt(np.sum((data['candidates'] - ground_truth_candidates) ** 2, axis=-1))
        sorted_indices = np.argsort(distances, axis=-1)
        selected_indices = sorted_indices[:, :, :num_candidates // 2]
        selected_candidates = data['candidates'][agt_idcs, t_idcs, selected_indices]
        selected_candidates_inds = data['candidates_inds'][agt_idcs, t_idcs, selected_indices]
        selected_gt_targets = np.zeros_like(data['gt_targets'])
        data['candidates'] = selected_candidates
        data['candidates_inds'] = selected_candidates_inds
        data['gt_targets'] = selected_gt_targets
    return collate_fn(batch)

import networkx as nx
def get_agent_targets(dataset, data, map_graph):
    # create a graph of the map 
    G = nx.DiGraph()
    for n in range(map_graph['map_lanes'].size(0)):
        G.add_node(n)
    for src, dest in zip(map_graph['suc_edges'][0][0], map_graph['suc_edges'][0][1]):
        G.add_edge(src, dest)
    # ego targets 
    ego_xy = data['ego'][-1, :2]
    closest_indices = k_closest_points(ego_xy, map_graph['map_lanes'], 1, dataset.source_range)
    nodes_within_N_steps = set()
    for source in closest_indices:
        nodes_within_N_steps.update(BFS_within_N(G, source, dataset.target_range))

    # neighbor targets 
    for i in range(data['neighbors'].shape[0]):
        neighbor_xy = data['neighbors'][i, -1, :2]
        closest_indices = np.concatenate([closest_indices, k_closest_points(neighbor_xy, map_graph['map_lanes'], 1, dataset.lane_discretization)], 0)

def k_closest_points(center, points, k, max_distance):
    # Subtract the center point from all points and square the differences
    differences = np.square(points - center)
    
    # Sum the squared differences along the second axis to get squared Euclidean distances
    sq_distances = np.sum(differences, axis=1)
    
    # Filter points based on max_distance
    close_enough = np.where(sq_distances <= max_distance**2)
    close_points = points[close_enough]
    
    # Check if k is greater than the number of points
    if k > len(close_points):
        print("K is greater than the number of points, returning all points.")
        return close_points
    
    # Get the indices of the k smallest distances among the close enough points
    closest_indices = np.argpartition(sq_distances[close_enough], k)[:k]
    
    return closest_indices

from collections import deque
def BFS_within_N(G, source, N):
    visited = set()
    queue = deque([(source, 0)])

    while queue:
        node, depth = queue.popleft()
        if node not in visited and depth <= N:
            visited.add(node)
            if depth < N:  # No need to add neighbors to queue if already at max depth
                queue.extend((neighbor, depth + 1) for neighbor in G.neighbors(node))

    return visited

# for rounD dataset with time interval of 0.04s
def interpolate_points(p1, p2):
    return (p1 + p2) / 2

def interpolate_angle(angle1, angle2):
    radians1 = np.deg2rad(angle1)
    radians2 = np.deg2rad(angle2)

    # Average the sine and cosine components
    sin_avg = 0.5 * (np.sin(radians1) + np.sin(radians2))
    cos_avg = 0.5 * (np.cos(radians1) + np.cos(radians2))

    # Compute the interpolated angle
    interpolated_angle = np.rad2deg(np.arctan2(sin_avg, cos_avg))
    return interpolated_angle % 360

def interpolate_track(track):
    x_centers = track['xCenter']
    y_centers = track['yCenter']

    x_velocity = track['xVelocity']
    y_velocity = track['yVelocity']

    head_angles = track['heading']

    new_x_centers = []
    new_y_centers = []

    new_x_velocity = []
    new_y_velocity = []

    new_headings = []
    new_frames = []
    for i in range(len(x_centers) - 1):
        # centers
        new_x_centers.append(x_centers[i])
        new_y_centers.append(y_centers[i])
        mid_x = interpolate_points(x_centers[i], x_centers[i + 1])
        mid_y = interpolate_points(y_centers[i], y_centers[i + 1])
        new_x_centers.append(mid_x)
        new_y_centers.append(mid_y)
        # velocities
        new_x_velocity.append(x_velocity[i])
        new_y_velocity.append(y_velocity[i])
        mid_x_velocity = interpolate_points(x_velocity[i], x_velocity[i + 1])
        mid_y_velocity = interpolate_points(y_velocity[i], y_velocity[i + 1])
        new_x_velocity.append(mid_x_velocity)
        new_y_velocity.append(mid_y_velocity)
        # headings
        new_headings.append(head_angles[i])
        mid_heading = interpolate_angle(head_angles[i], head_angles[i + 1])
        new_headings.append(mid_heading)
        # frames
        new_frames.append(track['frame'][i]*2)
        new_frames.append(track['frame'][i]*2+1)
    
    # Don't forget to add the last point
    new_x_centers.append(x_centers[-1])
    new_y_centers.append(y_centers[-1])
    new_x_velocity.append(x_velocity[-1])
    new_y_velocity.append(y_velocity[-1])
    new_headings.append(head_angles[-1])
    new_frames.append(track['frame'][-1]*2)

    track['xCenter'] = np.array(new_x_centers)
    track['yCenter'] = np.array(new_y_centers)
    track['center'] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
    track['xVelocity'] = np.array(new_x_velocity)
    track['yVelocity'] = np.array(new_y_velocity)
    track['heading'] = np.array(new_headings)
    track['frame'] = np.array(new_frames)

    return track

def get_highd_lane_markings(recording_df, extend_width=2.):
    """
    Extracts upper and lower lane markings from data frame;
    extend width of the outter lanes because otherwise some vehicles are off-road at the first time step.

    :param recording_df: data frame of the recording meta information
    """
    upper_lane_markings = [-float(x) for x in recording_df.upperLaneMarkings.values[0].split(";")]
    lower_lane_markings = [-float(x) for x in recording_df.lowerLaneMarkings.values[0].split(";")]
    len_upper = len(upper_lane_markings)
    len_lower = len(lower_lane_markings)
    # -8 + 1 = -7
    upper_lane_markings[0] += extend_width
    # -16 -1 = -17
    upper_lane_markings[len_upper - 1] += -extend_width
    # -22 + 1 = -21
    lower_lane_markings[0] += extend_width
    # -30 -1 = -31
    lower_lane_markings[len_lower - 1] += -extend_width
    return upper_lane_markings, lower_lane_markings

from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType, RoadUser, LineMarking
def create_highd_lanelet_network(meta, direction, resample_step=5.0):
    upper_lane_markings, lower_lane_markings = get_highd_lane_markings(meta)
    road_length = 420
    road_offset = 160
    lanelets = []
    if direction == "upper":
        for i in range(len(upper_lane_markings) - 1):
            # get two lines of current lane
            lane_y = upper_lane_markings[i]
            next_lane_y = upper_lane_markings[i + 1]

            right_vertices = np.array([[road_length + road_offset, lane_y], [-road_offset, lane_y]])
            
            left_vertices = np.array([[road_length + road_offset, next_lane_y], [-road_offset, next_lane_y]])
            
            center_vertices = (left_vertices + right_vertices) / 2.0

             # assign lanelet ID and adjacent IDs and lanelet types
            lanelet_id = i + 1
            lanelet_type = {LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY}
            adjacent_left = lanelet_id + 1
            adjacent_right = lanelet_id - 1
            adjacent_left_same_direction = True
            adjacent_right_same_direction = True
            line_marking_left_vertices = LineMarking.DASHED
            line_marking_right_vertices = LineMarking.DASHED

            if i == len(upper_lane_markings) - 2:
                adjacent_left = None
                adjacent_left_same_direction = False
                line_marking_left_vertices = LineMarking.SOLID
            elif i == 0:
                adjacent_right = None
                adjacent_right_same_direction = False
                line_marking_right_vertices = LineMarking.SOLID

            # create lanelet
            lanelets.append(Lanelet(lanelet_id=lanelet_id, left_vertices=left_vertices,      right_vertices=right_vertices,
                        center_vertices=center_vertices, adjacent_left=adjacent_left,
                        adjacent_left_same_direction=adjacent_left_same_direction, adjacent_right=adjacent_right,
                        adjacent_right_same_direction=adjacent_right_same_direction, user_one_way={RoadUser.VEHICLE},
                        line_marking_left_vertices=line_marking_left_vertices,
                        line_marking_right_vertices=line_marking_right_vertices, lanelet_type=lanelet_type))

    elif direction == "lower":
        for i in range(len(lower_lane_markings) - 1):
            # get two lines of current lane
            next_lane_y = lower_lane_markings[i + 1]
            lane_y = lower_lane_markings[i]

            left_vertices = np.array([[-road_offset, lane_y], [road_length + road_offset, lane_y]])
            right_vertices = np.array([[-road_offset, next_lane_y], [road_length + road_offset, next_lane_y]])

            center_vertices = (left_vertices + right_vertices) / 2.0

             # assign lanelet ID and adjacent IDs and lanelet types
            lanelet_id = i + 1
            lanelet_type = {LaneletType.INTERSTATE, LaneletType.MAIN_CARRIAGE_WAY}
            adjacent_left = lanelet_id - 1
            adjacent_right = lanelet_id + 1
            adjacent_left_same_direction = True
            adjacent_right_same_direction = True
            line_marking_left_vertices = LineMarking.DASHED
            line_marking_right_vertices = LineMarking.DASHED

            if i == 0:
                adjacent_left = None
                adjacent_left_same_direction = False
                line_marking_left_vertices = LineMarking.SOLID
            elif i == len(lower_lane_markings) - 2:
                adjacent_right = None
                adjacent_right_same_direction = False
                line_marking_right_vertices = LineMarking.SOLID

            # create lanelet
            lanelets.append(Lanelet(lanelet_id=lanelet_id, left_vertices=left_vertices,  right_vertices=right_vertices,
                        center_vertices=center_vertices, adjacent_left=adjacent_left,
                        adjacent_left_same_direction=adjacent_left_same_direction, adjacent_right=adjacent_right,
                        adjacent_right_same_direction=adjacent_right_same_direction, user_one_way={RoadUser.VEHICLE},
                        line_marking_left_vertices=line_marking_left_vertices,
                        line_marking_right_vertices=line_marking_right_vertices, lanelet_type=lanelet_type))
    return LaneletNetwork.create_from_lanelet_list(lanelets)

def rotate_point(x, y, angle_rad):
    """
    Rotate point (x, y) around the origin by angle_rad radians.
    """
    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_new, y_new

def get_rectangle_vertices(center_x, center_y, length, width, angle_rad):
    half_length = length / 2.0
    half_width = width / 2.0
    corners_at_origin = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width)
    ]
    rotated_corners = [rotate_point(x, y, angle_rad) for x, y in corners_at_origin]
    
    # Step 3: Translate back to original position
    translated_corners = [(x + center_x, y + center_y) for x, y in rotated_corners]
    
    return translated_corners