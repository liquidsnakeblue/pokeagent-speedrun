"""
Phase 6 prompt - Road to Rustboro City
"""

from typing import List, Any
from .common import build_base_prompt


def _get_full_path_to_direction(state_data, target_direction: str, location_name: str = '') -> List[str]:
    """
    Get full A* path to the most extreme REACHABLE point in a given direction.
    Tries multiple candidate points until it finds one that's actually pathable.
    """
    print(f"[PHASE 6 DEBUG] _get_full_path_to_direction called: direction={target_direction}, location={location_name}")
    
    if not state_data:
        print("[PHASE 6 DEBUG] No state_data provided!")
        return []
    
    try:
        from utils.map_formatter import format_map_grid
        from utils.state_formatter import astar_pathfind
        
        # Get player info
        player_data = state_data.get('player', {})
        player_position = player_data.get('position', {})
        player_coords = (int(player_position.get('x', 0)), int(player_position.get('y', 0)))
        
        # Get map data
        map_info = state_data.get('map', {})
        raw_tiles = map_info.get('tiles', [])
        npcs = map_info.get('npcs', [])
        
        if not raw_tiles:
            return []
        
        # Generate grid
        grid = format_map_grid(raw_tiles, "South", npcs, player_coords, location_name=location_name)
        
        if not grid:
            return []
        
        grid_height = len(grid)
        grid_width = len(grid[0]) if grid else 0
        
        # Find player on grid
        player_grid_x = None
        player_grid_y = None
        for y_idx, row in enumerate(grid):
            for x_idx, symbol in enumerate(row):
                if symbol == 'P':
                    player_grid_x = x_idx
                    player_grid_y = y_idx
                    break
            if player_grid_x is not None:
                break
        
        if player_grid_x is None:
            return []
        
        # Convert to grid array index
        player_y_idx = grid_height - 1 - player_grid_y
        
        # Find ALL walkable tiles and filter by direction
        walkable_tiles = ['.', '~', 'S']
        candidates = []
        center_x = grid_width // 2
        center_y = grid_height // 2
        
        for y_idx, row in enumerate(grid):
            for x_idx, symbol in enumerate(row):
                if symbol in walkable_tiles:
                    # Calculate grid Y coordinate
                    grid_y = grid_height - 1 - y_idx
                    
                    # Filter by direction
                    if target_direction == 'UP' and grid_y > player_grid_y:
                        # Higher Y = more north, prefer centered
                        score = (grid_y * 1000) - abs(x_idx - center_x)
                        candidates.append((score, (x_idx, y_idx)))
                    elif target_direction == 'DOWN' and grid_y < player_grid_y:
                        score = (grid_y * -1000) - abs(x_idx - center_x)
                        candidates.append((score, (x_idx, y_idx)))
                    elif target_direction == 'LEFT' and x_idx < player_grid_x:
                        score = (x_idx * -1000) - abs(y_idx - center_y)
                        candidates.append((score, (x_idx, y_idx)))
                    elif target_direction == 'RIGHT' and x_idx > player_grid_x:
                        score = (x_idx * 1000) - abs(y_idx - center_y)
                        candidates.append((score, (x_idx, y_idx)))
        
        # Sort by score (best first)
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        print(f"[PHASE 6 DEBUG] Found {len(candidates)} candidate tiles in direction {target_direction}")
        
        # Try to path to each candidate until we find one that works
        for i, (score, goal_pos) in enumerate(candidates[:20]):  # Try up to 20 best candidates
            path = astar_pathfind(grid, (player_grid_x, player_y_idx), goal_pos, location_name)
            if path:
                print(f"[PHASE 6 DEBUG] Found path to candidate #{i+1}: {path}")
                return path
        
        print(f"[PHASE 6 DEBUG] No path found to any candidate")
        return []
        
    except Exception as e:
        import traceback
        print(f"[PHASE 6 PATHFINDING ERROR] {e}")
        traceback.print_exc()
        return []


def _get_route_104_north_path(state_data, location_name: str = '') -> List[str]:
    """Get full A* path to the most northern point in Route 104."""
    return _get_full_path_to_direction(state_data, 'UP', location_name)


def _get_petalburg_west_path(state_data, location_name: str = '') -> List[str]:
    """Get full A* path to the most western point in Petalburg."""
    return _get_full_path_to_direction(state_data, 'LEFT', location_name)


def get_phase_6_prompt(
    objectives=None,  # Accept objectives parameter for consistency
    debug: bool = False,
    include_base_intro: bool = False,  # No generic intro
    include_pathfinding_rules: bool = False,  # No long pathfinding rules
    include_pathfinding_helper: bool = False,  # NO - we provide custom suggested actions!
    include_response_structure: bool = True,  # Keep action input format
    include_action_history: bool = False,  # Remove action history
    include_location_history: bool = False,  # Remove location history
    include_objectives: bool = False,  # Remove objectives (we have milestone instructions)
    include_movement_memory: bool = False,  # Remove movement memory
    include_stuck_warning: bool = False,  # Remove stuck warnings
    state_data=None,  # For pathfinding
    formatted_state='',  # For getting current location
    **kwargs
) -> str:
    """
    Get the Phase 6 prompt template.
    
    Phase 6 covers: Road to Rustboro City
    """
    # Get current location from formatted_state
    current_location = ''
    if formatted_state and "Current Location:" in formatted_state:
        for line in formatted_state.split('\n'):
            if line.strip().startswith("Current Location:"):
                current_location = line.split(":", 1)[1].strip()
                break
    
    # Calculate suggested action based on location
    suggested_action = _get_phase_6_suggested_action(state_data, current_location)
    
    # Handle special locations in Phase 6  
    loc_upper = current_location.upper() if current_location else ""
    
    if "ROUTE_102" in loc_upper or "ROUTE 102" in loc_upper:
        # Wally is showing you how to catch Pokemon - spam A
        phase_intro = """🎮 PHASE 6: Road to Rustboro City

📍 ROUTE 102 - WALLY'S TUTORIAL
- Wally is showing you how to catch Pokemon
- Just spam A to skip the dialogue/tutorial: A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A"""
    
    elif "GYM" in loc_upper:
        phase_intro = """🎮 PHASE 6: Road to Rustboro City

📍 IN PETALBURG GYM  
- Spam A to finish all dialogue/Wally tutorial
- Then exit the gym via the door"""
    
    elif "PETALBURG" in loc_upper:
        phase_intro = """🎮 PHASE 6: Road to Rustboro City

📍 PETALBURG CITY
- Exit Petalburg and head WEST toward Route 104
- Follow the suggested action to path west"""
    
    elif "ROUTE_104" in loc_upper or "ROUTE 104" in loc_upper:
        phase_intro = """🎮 PHASE 6: Road to Rustboro City

📍 ROUTE 104
- CHAIN THE ENTIRE THING IN ONE ACTION. 
- FOLLOW THE SUGGESTED ACTIONS EXACTLY! """
    
    else:
        phase_intro = """🎮 PHASE 6: Road to Rustboro City
Navigate through Route 104, Petalburg Woods, and reach Rustboro City.
Defeat Team Aqua grunt in the woods and continue north to Rustboro."""
    
    return build_base_prompt(
        phase_intro=phase_intro,
        debug=debug,
        include_base_intro=include_base_intro,
        include_pathfinding_rules=include_pathfinding_rules,
        include_pathfinding_helper=include_pathfinding_helper,
        include_response_structure=include_response_structure,
        include_action_history=include_action_history,
        include_location_history=include_location_history,
        include_objectives=include_objectives,
        include_movement_memory=include_movement_memory,
        include_stuck_warning=include_stuck_warning,
        phase_intro_at_end=True,  # Put milestone instructions AFTER map/state
        suggested_action_suffix=suggested_action,  # Add custom suggested action
        formatted_state=formatted_state,
        state_data=state_data,
        **kwargs
    )


def _get_phase_6_suggested_action(state_data, current_location: str = None) -> str:
    """
    Calculate the suggested action for Phase 6 based on location.
    """
    if not current_location or not state_data:
        return ""
    
    loc_upper = current_location.upper()
    
    # Check if in battle
    is_in_battle = False
    game_state = state_data.get('game_state', {})
    state_name = game_state.get('state', '').lower()
    is_in_battle = 'battle' in state_name
    
    # Route 102 - Wally tutorial, spam A
    if "ROUTE_102" in loc_upper or "ROUTE 102" in loc_upper:
        return "\nSuggested action: A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A"
    
    # Gym - spam A then exit
    elif "GYM" in loc_upper:
        return "\nSuggested action: A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, RIGHT, DOWN, A"
    
    # Petalburg City - path west
    elif "PETALBURG" in loc_upper:
        action_seq = _get_petalburg_west_path(state_data, current_location)
        if action_seq and len(action_seq) > 0:
            action_seq.append('A')
            return f"\nSuggested action: {', '.join(action_seq)}"
        return "\nSuggested action: LEFT, A"
    
    # Route 104
    elif "ROUTE_104" in loc_upper or "ROUTE 104" in loc_upper:
        if is_in_battle:
            # In battle - randomized sequences
            import random
            if random.random() < 0.5:
                return "\nSuggested action: A, B, LEFT, LEFT, LEFT, A, RIGHT, RIGHT, A"
            else:
                return "\nSuggested action: A, B, LEFT, LEFT, LEFT, A, DOWN, DOWN, A"
        else:
            # Not in battle - path north
            print(f"[PHASE 6 DEBUG] Calling _get_route_104_north_path with location={current_location}")
            action_seq = _get_route_104_north_path(state_data, current_location)
            print(f"[PHASE 6 DEBUG] Got action_seq: {action_seq}")
            if action_seq and len(action_seq) > 0:
                action_seq.append('A')
                result = f"\nSuggested action: {', '.join(action_seq)}"
                print(f"[PHASE 6 DEBUG] Returning: {result}")
                return result
            print(f"[PHASE 6 DEBUG] Returning fallback")
            return "\nSuggested action: UP, A"
    
    return ""

