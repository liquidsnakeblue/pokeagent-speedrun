"""
Phase 2 prompt - Mid game (after first gym through mid-game milestones)
"""

from .common import build_base_prompt


def get_phase_2_prompt(
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
    Get the Phase 2 prompt template.
    
    Phase 2 covers: [PLACEHOLDER - describe what phase 2 covers]
    
    Args:
        debug: If True, log the prompt to console
        include_pathfinding_rules: Include pathfinding rules (default: True)
        include_response_structure: Include response structure (default: True)
        include_action_history: Include action history (default: True)
        include_location_history: Include location history (default: True)
        include_objectives: Include objectives (default: True)
        include_movement_memory: Include movement memory (default: True)
        include_stuck_warning: Include stuck warning (default: True)
        **kwargs: All other prompt building arguments (passed to build_base_prompt)
        
    Returns:
        Complete formatted prompt string
    """
    # Phase 2 Milestones: LITTLEROOT_TOWN, PLAYER_HOUSE_ENTERED, PLAYER_BEDROOM, CLOCK_SET, RIVAL_HOUSE, RIVAL_BEDROOM
    phase_intro = """🎮 PHASE 2:
    If inside the truck, move right to exit it.
    When in littleroot, click A to finish the dialoguje which will bring you into the player's house.
    When in player's house, go upstairs by moving ONTO the stairs tile.
    When upstairs, move UP to the wall that has the clock on it. Once you have clicked the clock, do "A, UP, A" to set it.

    Once it is set, move back onto the stairs to go downstairs, and exit the house.
    
    Always end each action with "A" at the end after your choice.

    When in Brenden's house at 1F (first floor), GO TOWARDS THE "S" on the map which will be UP.
    
    When on 2F (second floor), you must interact with the clock.
    """
    
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

