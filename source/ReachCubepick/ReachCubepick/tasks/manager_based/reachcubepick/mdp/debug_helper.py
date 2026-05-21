import torch

def debug_robot_state(
    env, 
    threshold_arm: float = 10.0,      
    threshold_gripper: float = 10.0,  
    report_top_k: int = 5,           
    print_freq_stats: int = 1000    
):

    step_count = env.unwrapped.common_step_counter
    if step_count % print_freq_stats != 0:
        return
    
    robot = env.scene["robot"]
    joint_vels = robot.data.joint_vel  # (N, J)
    joint_names = robot.joint_names
    abs_vels = torch.abs(joint_vels)
    
    actions = env.action_manager.action
    abs_actions = torch.abs(actions)
    
    arm_ids = [0, 1, 2, 3, 4, 5]
    gripper_ids = [6, 7, 8, 9, 10, 11] 
    
    # Arm
    arm_vels = abs_vels[:, arm_ids]
    arm_max_vals, arm_max_indices = torch.max(arm_vels, dim=1) # (N,), (N,)
    arm_violations = arm_max_vals > threshold_arm
    
    # Gripper
    gripper_vels = abs_vels[:, gripper_ids]
    gripper_max_vals, gripper_max_indices = torch.max(gripper_vels, dim=1)
    gripper_violations = gripper_max_vals > threshold_gripper
    
    any_violation = arm_violations | gripper_violations
    violation_count = int(torch.sum(any_violation).item())
    

    print(f"{'='*62}\n")
    print(f"[ROBOT STATE DEBUG STATS] Step {step_count}: "
          f"Max Arm Vel={arm_max_vals.max():.2f}/{threshold_arm}, "
          f"Max Grip Vel={gripper_max_vals.max():.2f}/{threshold_gripper}, "
          f"Violations={violation_count}/{env.num_envs}")

    if violation_count > 0:
        violation_ids = torch.nonzero(any_violation, as_tuple=True)[0].cpu().tolist()
        
        ids_to_report = violation_ids[:report_top_k]
        
        print(f"[Velocity DEBUG] Step {step_count}")
        print(f"Total Violating Envs: {violation_count} / {env.num_envs}")
        print(f"Thresholds -> Arm: {threshold_arm} rad/s, Gripper: {threshold_gripper} rad/s")
        
        for i, env_id in enumerate(ids_to_report):
            env_id = int(env_id)
            reasons = []
            
            if arm_violations[env_id]:
                idx_in_group = int(arm_max_indices[env_id])
                global_idx = arm_ids[idx_in_group]
                val = float(arm_max_vals[env_id])
                name = joint_names[global_idx]
                reasons.append(f"ARM:'{name}'(Idx:{global_idx})={val:.2f}")
            
            if gripper_violations[env_id]:
                idx_in_group = int(gripper_max_indices[env_id])
                global_idx = gripper_ids[idx_in_group]
                val = float(gripper_max_vals[env_id])
                name = joint_names[global_idx]
                reasons.append(f"GRIP:'{name}'(Idx:{global_idx})={val:.2f}")
            
            max_act_val = float(torch.max(abs_actions[env_id]))
            
            context_info = ""
            if env.scene["cube"] is not None:
                cube = env.scene["cube"]
                r_pos = robot.data.root_pos_w[env_id, :3]
                c_pos = cube.data.root_pos_w[env_id, :3]
                dist = float(torch.norm(r_pos - c_pos).item())
                context_info = f"| Dist={dist:.3f}m | MaxAct={max_act_val:.2f}"
            
            print(f"  Env [{env_id:3d}]: " + " | ".join(reasons) + " " + context_info)
            
        if violation_count > report_top_k:
            print(f"  ... and {violation_count - report_top_k} more environments hidden.")
        
        robot_pos_rel = robot.data.root_pos_w - env.scene.env_origins
        cube_pos_rel = env.scene["cube"].data.root_pos_w - env.scene.env_origins
        
        r_dist = torch.norm(robot_pos_rel, dim=1)
        c_dist = torch.norm(cube_pos_rel, dim=1) 
        
        explosion_mask = (r_dist > 5.0) | (c_dist > 5.0)
        if explosion_mask.any():
            exp_count = int(torch.sum(explosion_mask).item())
            print(f"   CRITICAL: {exp_count} environments have exploded (Pos > 5m)!")  
        mask_pos = (r_dist > 10) | (c_dist > 10)
        if torch.any(mask_pos):
            bad_ids = torch.nonzero(mask_pos, as_tuple=True)[0].cpu().tolist()[:3]
            print(f"[CRITICAL] PHYSICS EXPLOSION at Step {step_count}! Envs: {bad_ids}")
            print(f"  Max Robot Dist: {r_dist.max().item():.2f}m, Max Cube Dist: {c_dist.max().item():.2f}m")
    else:
        print(f"[ROBOT STATE DEBUG STATS] Step {step_count}: No velocity violations detected. Max Arm Vel={arm_max_vals.max():.2f}, Max Grip Vel={gripper_max_vals.max():.2f}")


def debug_robot_joint_acc(env, thres_acc=3.0, report_top_k: int = 5, print_freq_stats: int = 1000):
    step_count = env.unwrapped.common_step_counter
    if step_count % print_freq_stats != 0:
        return

    robot = env.scene["robot"]
    joint_acc = robot.data.joint_acc
    joint_names = robot.joint_names
    abs_acc = torch.abs(joint_acc)

    violating_acc = abs_acc > thres_acc
    any_violation = violating_acc.any(dim=1)
    violation_count = int(torch.sum(any_violation).item())

    if violation_count > 0:
        print(f"{'='*62}\n")
        print(f"[ROBOT ACC DEBUG] Step {step_count}: Detected {violating_acc.sum().item()} joints with high acceleration (> {thres_acc} rad/s²)")

        # Rank violating environments by their max absolute joint acceleration
        max_acc_vals, _ = torch.max(abs_acc, dim=1)  # (N,)
        violation_ids = torch.nonzero(any_violation, as_tuple=True)[0].cpu()
        if violation_ids.numel() == 0:
            return

        viol_vals = max_acc_vals[violation_ids].cpu()
        sorted_idx = torch.argsort(viol_vals, descending=True)
        top_ids = violation_ids[sorted_idx][:report_top_k].tolist()

        print(f"Total Violating Envs: {violation_count} / {env.num_envs}")
        print(f"Threshold Acc -> {thres_acc} rad/s² | Reporting Top {min(report_top_k, len(top_ids))} Envs")

        for env_id in top_ids:
            env_id = int(env_id)
            env_violations = torch.nonzero(violating_acc[env_id], as_tuple=True)[0]
            reasons = []
            for idx in env_violations:
                idx = int(idx)
                acc_val = float(abs_acc[env_id, idx])
                joint_name = joint_names[idx]
                reasons.append(f"'{joint_name}'(Idx:{idx})={acc_val:.2f} rad/s²")

            print(f"  Env [{env_id:3d}]: " + " | ".join(reasons))

        if violation_count > report_top_k:
            print(f"  ... and {violation_count - report_top_k} more environments hidden.")
    else:
        print(f"[ROBOT ACC DEBUG] Step {step_count}: No joint acceleration violations detected. Max Acc={max_acc_vals.max():.2f} rad/s²")


def debug_cube_move_state(env, distance_tensor, distance_reward, print_freq_stats):
    step_count = env.unwrapped.common_step_counter
    # print mean and max of cube distance
    if step_count % print_freq_stats == 0:
        print(f"{'='*62}\n")
        cube = env.scene["cube"]
        cube_vel = cube.data.root_vel_w[:, :3]
        mean_cube_vel = torch.norm(cube_vel, dim=1).mean().item()
        max_cube_vel = torch.norm(cube_vel, dim=1).max().item()
        print(f"[CUBE MOVE DEBUG STATS] Step {step_count}: Cube Vel -> Mean: {mean_cube_vel:.3f}m/s, Max: {max_cube_vel:.3f}/s")
        mean_dist = distance_tensor.mean().item()
        max_dist = distance_tensor.max().item()
        distance_reward_mean = distance_reward.mean().item()
        print(f"[CUBE MOVE DEBUG STATS] Step {step_count}: Cube Dist -> Mean: {mean_dist:.3f}m, Max: {max_dist:.3f}m | Reward -> {distance_reward_mean}")