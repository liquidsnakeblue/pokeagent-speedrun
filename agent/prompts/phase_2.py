"""
Phase 2 prompt - Mid game (after first gym through mid-game milestones)
"""

from typing import List, Any
from .common import build_base_prompt


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


def _get_phase_2_conditional_prompts(objectives: List[Any]) -> str:
    """
    Generate conditional prompt sections based on completed objectives.
    
    Phase 2 Objectives (in order):
    - story_intro_complete: Complete intro cutscene with moving van
    - story_player_house: Enter player's house for the first time
    - story_player_bedroom: Go upstairs to player's bedroom
    - story_clock_set: Set the clock on the wall in the player's bedroom
    - story_rival_house: Visit May's house next door
    - story_rival_bedroom: Go to May's bedroom (upstairs in her house)
    
    Args:
        objectives: List of objective objects
        
    Returns:
        Conditional prompt text based on current progress
    """
    conditional_sections = []
    
    # Check each objective and add specific guidance
    if not _is_objective_completed(objectives, "story_intro_complete"):
        conditional_sections.append("""
🚚 MOVING VAN:
- You're inside the moving van
- Move RIGHT to exit the truck""")
    
    elif not _is_objective_completed(objectives, "story_player_house"):
        conditional_sections.append("""
🏘️ LITTLEROOT TOWN:
- Press A to advance dialogue""")
    
    elif not _is_objective_completed(objectives, "story_player_bedroom"):
        conditional_sections.append("""
🏠 PLAYER'S HOUSE - 1ST FLOOR:
- Go upstairs to your bedroom
- Find the stairs tile (S) on the map
- Move ONTO the stairs tile to go up""")
    
    elif not _is_objective_completed(objectives, "story_clock_set"):
        conditional_sections.append("""
🛏️ PLAYER'S BEDROOM - 2ND FLOOR:
- You need to set the clock on the wall
- Press A to interact with the clock
- Then press: A, UP, A to set it
- After setting, go back downstairs and exit the house""")
    
    elif not _is_objective_completed(objectives, "story_rival_house"):
        conditional_sections.append("""
🏘️ LITTLEROOT TOWN:
- You've set your clock, now visit May's house next door
- Look for the rival's house (should be nearby)
- Walk into the entrance""")
    
    elif not _is_objective_completed(objectives, "story_rival_bedroom"):
        conditional_sections.append("""
🏠 MAY'S HOUSE:
- Go upstairs to May's bedroom (Brendan's house)
- Find the stairs tile (S) on the map
- Move ONTO the stairs tile to go up
- On 2F (second floor), you must interact with the clock at coordinate (5,1)""")
    
    else:
        # All phase 2 objectives complete
        conditional_sections.append("""
✅ PHASE 2 COMPLETE:
- All initial house objectives are done
- Continue with your next objectives""")
    
    return "\n".join(conditional_sections)


def get_phase_2_prompt(
    objectives: List[Any] = None,
    debug: bool = False,
    include_pathfinding_rules: bool = False,
    include_response_structure: bool = True,
    include_action_history: bool = True,
    include_location_history: bool = False,
    include_objectives: bool = False,
    include_movement_memory: bool = False,
    include_stuck_warning: bool = True,
    **kwargs
) -> str:
    """
    Get the Phase 2 prompt template with conditional sections based on objectives.
    
    Phase 2 covers: Initial game setup from moving van through setting up in Littleroot Town
    
    Args:
        objectives: List of current objective objects (for conditional prompts)
        debug: If True, log the prompt to console
        include_pathfinding_rules: Include pathfinding rules (default: False)
        include_response_structure: Include response structure (default: True)
        include_action_history: Include action history (default: True)
        include_location_history: Include location history (default: False)
        include_objectives: Include objectives (default: False)
        include_movement_memory: Include movement memory (default: False)
        include_stuck_warning: Include stuck warning (default: True)
        **kwargs: All other prompt building arguments (passed to build_base_prompt)
        
    Returns:
        Complete formatted prompt string
    """
    # Build base intro
    base_intro = "🎮 PHASE 2: Initial Setup in Littleroot Town"
    
    # Add conditional prompts based on objectives
    conditional_prompts = _get_phase_2_conditional_prompts(objectives or [])
    
    # Combine intro with conditional sections
    phase_intro = f"""{base_intro}

{conditional_prompts}

💡 IMPORTANT TIPS:
- Always end actions with "A" when you need to advance dialogue
- When going upstairs, move ONTO the stairs tile (S)
- Use the coordinate system to find specific objects (like clocks)
- Check the visual map to see where stairs (S) and doors (D) are located"""
    
    return build_base_prompt(
        phase_intro=phase_intro,
        debug=debug,
        include_pathfinding_rules=include_pathfinding_rules,
        include_response_structure=include_response_structure,
        include_action_history=include_action_history,
        include_location_history=include_location_history,
        include_objectives=include_objectives,
        include_movement_memory=include_movement_memory,
        include_stuck_warning=include_stuck_warning,
        **kwargs
    )

