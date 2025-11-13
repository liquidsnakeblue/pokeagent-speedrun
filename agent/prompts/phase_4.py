"""
Phase 4 prompt - Rival Battle & Pokedex
"""

from typing import List, Any
from .common import build_base_prompt
from utils.state_formatter import find_path_around_obstacle


def _is_objective_completed(objectives: List[Any], objective_id: str) -> bool:
    """
    Check if a specific objective is completed.
    
    Args:
        objectives: List of objective objects
        objective_id: ID of the objective to check
        
    Returns:
        True if objective is completed, False otherwise
    """
    if not objectives:
        return False
    
    for obj in objectives:
        if hasattr(obj, 'id') and obj.id == objective_id and hasattr(obj, 'completed'):
            return obj.completed
    
    return False


def _get_phase_4_suggested_action(state_data, current_location: str = None, objectives: List[Any] = None) -> str:
    """
    Calculate the suggested action for Phase 4.
    
    Logic:
    - If in Birch's lab -> suggest DOWN (exit south)
    - If at highest point in map AND working on OLDALE_TOWN or ROUTE_103 -> suggest RIGHT
    - If NORTH is not blocked -> suggest UP
    - If NORTH is blocked (including doors) -> suggest the detour path
    
    Args:
        state_data: Game state data
        current_location: Current location name
        objectives: Current objectives list
        
    Returns:
        Suggested action string to append at end of prompt
    """
    if not state_data:
        return ""
    
    # Check if in Birch's lab
    if current_location and "BIRCH" in current_location.upper():
        return "\nSuggested action: DOWN"
    
    # Check if working on OLDALE_TOWN or ROUTE_103 objectives
    working_on_north_objectives = (
        (objectives and not _is_objective_completed(objectives, "OLDALE_TOWN")) or
        (objectives and not _is_objective_completed(objectives, "ROUTE_103"))
    )
    
    # Check if player is at the highest point in the map
    if working_on_north_objectives:
        # Get map tiles to check if we're at the top
        map_info = state_data.get('map', {})
        raw_tiles = map_info.get('tiles', [])
        
        if raw_tiles:
            # Player is at center of 15x15 grid
            center_y = len(raw_tiles) // 2
            
            # Check if there are mostly walls/blocked tiles above the player
            # (indicating we're at the top of the accessible area)
            tiles_above = 0
            walkable_above = 0
            
            for y in range(center_y):  # Check all rows above center
                if y < len(raw_tiles):
                    for tile in raw_tiles[y]:
                        if tile:
                            tiles_above += 1
                            # Check if tile is walkable
                            from utils.map_formatter import format_tile_to_symbol
                            symbol = format_tile_to_symbol(tile)
                            if symbol in ['.', '~', 'D', 'S']:
                                walkable_above += 1
            
            # If less than 10% of tiles above are walkable, we're at the top
            if tiles_above > 0 and (walkable_above / tiles_above) < 0.1:
                return "\nSuggested action: RIGHT"
    
    # Get pathfinding info for NORTH
    path_info = find_path_around_obstacle(state_data, 'UP')
    
    if not path_info:
        return "\nSuggested action: UP"
    
    # Check if NORTH is blocked
    if path_info.get('is_blocked') and path_info.get('detour_needed'):
        # Get the detour path
        action_seq = path_info.get('action_sequence', [])
        if action_seq and len(action_seq) > 0:
            actions_str = ', '.join(action_seq)
            return f"\nSuggested action: {actions_str}"
    
    # NORTH is clear - just go UP
    return "\nSuggested action: UP"


def _get_phase_4_conditional_prompts(objectives: List[Any], current_location: str = None) -> str:
    """
    Generate conditional prompt sections based on completed objectives.
    
    Phase 4 Objectives (in order):
    - OLDALE_TOWN: Reach Oldale Town (first town north of Route 101)
    - ROUTE_103: Navigate to Route 103 to battle your rival
    - RECEIVED_POKEDEX: Receive the Pokedex from Professor Birch
    
    Args:
        objectives: List of objective objects
        current_location: Current location name
        
    Returns:
        Conditional prompt text based on current progress
    """
    conditional_sections = []
    
    
#     if not _is_objective_completed(objectives, "OLDALE_TOWN"):
#         # Still need to reach Oldale Town
#         conditional_sections.append("""
# 🗺️  FIND YOUR RIVAL:

# YOUR GOAL IS TO GO NORTH (UP) NEVER GO DOWN UNLESS YOU HAVE A REASON TO.

# IMPORTANT - THIS IS THE MOST IMPORTANT INSTRUCTION:
# - WHEN IN BATTLE, ALWAYS DO: "A, B, LEFT, A, RIGHT, A" TO ATTACK. CHAIN THE ENTIRE THING IN ONE ACTION.
# - YOUR GOAL IS TO GO COMPLETELY NORTH AND AVOID DOORS AT ALL COSTS. IF YOU CANNOT ANYMORE, YOUR GOAL IS TO GO NEXT TO THE 'N' ON THE MAP AND PRESS A.
# - IF YOU ARE IN PROFESSOR BIRCHS LAB, GO DOWN!!! PLEASE JUST GO DOWN IF U SEE PROFESSIOR BIRCH'S LAB ANYWHERE IN THE LOCATION INFORMATION. JUST DOWN!
# - YOUR RIVAL IS MARKED AS N ON THE MAP in the most NORTHERN part of the map on route 103. YOU NEED TO GO RIGHT BESIDE HER AND CLICK A. THIS WILL INVOLVE GOING DOWN ONE AND RIGHT A BUNCH.
# - YOUR RIVAL IS AT (7,13) FACE IT AND PRESS A. MOVE RIGHT IF YOU DO NOT SEE THEM.
# - FOLLOW THE SUGGESTED ACTION UNLESS U ARE GOING TO YOUR RIVAL. IT IS ALWAYS SMARTER THAN YOU.
# """)


    if not _is_objective_completed(objectives, "OLDALE_TOWN"):
        # Still need to reach Oldale Town
        conditional_sections.append("""
🗺️  FIND YOUR RIVAL:

IF YOU SEE THE MAP IT MEANS YOU ARE NOT IN BATTLE. TO ENTER BATTLE: GO 'DOWN, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, A' CHAIN THE ENTIRE THING

THEN ONCE IN BATTLE ALWAYS DO: "A, B, LEFT, A, RIGHT, A" TO ATTACK. CHAIN THE ENTIRE THING IN ONE ACTION.
""")
    
    elif not _is_objective_completed(objectives, "ROUTE_103"):
        conditional_sections.append("""
⚔️ FIND YOUR RIVAL:
- From Oldale Town, head north to Route 103
- Your rival is waiting on Route 103 which is straight NORTH for a Pokemon battle
- Safe walkable spots on the map are denoted as "." while blocked spots are denoted as "#".
- Keep going north as far as you can. Note this may mean you have to go around blocked spots by going left then right.
- You may encounter pokemon battles - if you do press over and over until the battle is over.
""")
    
    elif not _is_objective_completed(objectives, "RECEIVED_POKEDEX"):
        conditional_sections.append("""
📱 GET THE POKEDEX:
- After defeating your rival, return to Littleroot Town
- Visit Professor Birch's lab
- You'll receive the Pokedex - a device to record Pokemon data
- Press A to advance dialogue and receive the Pokedex
- The Pokedex is essential for tracking your Pokemon journey!""")
    
    
    return "\n".join(conditional_sections)


def get_phase_4_prompt(
    objectives: List[Any] = None,
    debug: bool = False,
    include_base_intro: bool = False,  # No generic intro
    include_pathfinding_rules: bool = False,  # No long pathfinding rules
    include_pathfinding_helper: bool = True,  # YES - show A* pathfinding actions!
    include_response_structure: bool = True,  # Keep action input format
    include_action_history: bool = False,  # Remove action history
    include_location_history: bool = False,  # Remove location history
    include_objectives: bool = False,  # Remove objectives (we have milestone instructions)
    include_movement_memory: bool = False,  # Remove movement memory
    include_stuck_warning: bool = False,  # Remove stuck warnings
    include_phase_tips: bool = False,  # No tips
    formatted_state: str = None,  # To extract location
    state_data = None,  # For pathfinding helper
    **kwargs
) -> str:
    """
    Get the Phase 4 prompt template with conditional sections based on objectives.
    
    Phase 4 covers: Rival battle on Route 103 and receiving the Pokedex
    
    CLEAN PROMPT - Only includes:
    - Milestone-based instructions
    - Map / Legend / Map Tiles (from formatted_state)
    - Game State
    - Action input instructions
    - Pathfinding recommended actions (A* paths)
    
    Args:
        objectives: List of current objective objects (for conditional prompts)
        debug: If True, log the prompt to console
        formatted_state: Formatted state string (includes map)
        state_data: Game state data for pathfinding helper
        **kwargs: All other prompt building arguments (passed to build_base_prompt)
        
    Returns:
        Complete formatted prompt string
    """
    # Extract current location from formatted_state if available
    current_location = None
    if formatted_state and "Current Location:" in formatted_state:
        # Extract location from "Current Location: OLDALE_TOWN" line
        for line in formatted_state.split('\n'):
            if line.strip().startswith("Current Location:"):
                current_location = line.split(":", 1)[1].strip()
                break
    
    # If no location from formatted_state, try state_data
    if not current_location and state_data:
        current_location = state_data.get('player', {}).get('location', '')
    
    # Build base intro
    base_intro = "🎮 PHASE 4: Rival Battle & Pokedex"
    
    # Add conditional prompts based on objectives and location
    conditional_prompts = _get_phase_4_conditional_prompts(objectives or [], current_location)
    
    # Build phase intro - only add tips if requested
    if include_phase_tips:
        phase_intro = f"""{base_intro}

{conditional_prompts}
"""
    else:
        phase_intro = f"""{base_intro}

{conditional_prompts}"""
    
    # Calculate suggested action for Phase 4
    suggested_action = _get_phase_4_suggested_action(state_data, current_location, objectives)
    
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
        suggested_action_suffix=suggested_action,  # Add suggested action at the end
        formatted_state=formatted_state,
        state_data=state_data,
        **kwargs
    )

