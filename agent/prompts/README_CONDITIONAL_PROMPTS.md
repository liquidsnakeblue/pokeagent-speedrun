# Conditional Prompts System

## Overview

The conditional prompts system allows phase prompts to dynamically change based on which objectives have been completed. This provides more targeted, context-specific guidance to the agent.

## How It Works

1. **Raw objectives are passed** from `SimpleAgent` to each phase prompt function
2. **Helper functions check** which objectives are completed
3. **Conditional sections are built** based on current progress
4. **Dynamic prompt is returned** with only relevant guidance

## Example: Phase 2 Implementation

Phase 2 (`phase_2.py`) demonstrates the full pattern:

### Helper Functions

```python
def _is_objective_completed(objectives: List[Any], objective_id: str) -> bool:
    """Check if a specific objective is completed."""
    if not objectives:
        return False
    
    for obj in objectives:
        if hasattr(obj, 'id') and obj.id == objective_id and hasattr(obj, 'completed'):
            return obj.completed
    
    return False


def _get_phase_2_conditional_prompts(objectives: List[Any]) -> str:
    """Generate conditional prompt sections based on completed objectives."""
    conditional_sections = []
    
    # Check each objective sequentially
    if not _is_objective_completed(objectives, "story_intro_complete"):
        conditional_sections.append("""
🚚 MOVING VAN:
- You're inside the moving van
- Move RIGHT to exit the truck
- Press A to advance any dialogue""")
    
    elif not _is_objective_completed(objectives, "story_player_house"):
        conditional_sections.append("""
🏘️ LITTLEROOT TOWN:
- You need to enter your player's house
- Look for the house and walk into the entrance
- Press A to advance dialogue when needed""")
    
    # ... more conditions ...
    
    return "\n".join(conditional_sections)
```

### Phase Prompt Function

```python
def get_phase_2_prompt(
    objectives: List[Any] = None,  # Accept objectives
    debug: bool = False,
    **kwargs
) -> str:
    # Build base intro
    base_intro = "🎮 PHASE 2: Initial Setup in Littleroot Town"
    
    # Add conditional prompts
    conditional_prompts = _get_phase_2_conditional_prompts(objectives or [])
    
    # Combine
    phase_intro = f"""{base_intro}

{conditional_prompts}

💡 IMPORTANT TIPS:
- Always end actions with "A" when you need to advance dialogue
- Use the coordinate system to find specific objects"""
    
    return build_base_prompt(phase_intro=phase_intro, **kwargs)
```

## Adding Conditional Prompts to Other Phases

### Step 1: Add objectives parameter

```python
def get_phase_X_prompt(
    objectives: List[Any] = None,  # Add this
    debug: bool = False,
    **kwargs
) -> str:
```

### Step 2: Create conditional helper function

```python
def _get_phase_X_conditional_prompts(objectives: List[Any]) -> str:
    conditional_sections = []
    
    # Check your phase objectives
    if not _is_objective_completed(objectives, "story_your_objective_id"):
        conditional_sections.append("""
📌 OBJECTIVE NAME:
- Specific guidance for this objective
- Step-by-step instructions
- Important tips""")
    
    elif not _is_objective_completed(objectives, "story_next_objective"):
        conditional_sections.append("""
📌 NEXT OBJECTIVE:
- Different guidance
- Adapted to current progress""")
    
    else:
        conditional_sections.append("""
✅ PHASE X COMPLETE:
- All objectives done
- Ready for next phase""")
    
    return "\n".join(conditional_sections)
```

### Step 3: Use in prompt building

```python
def get_phase_X_prompt(objectives: List[Any] = None, **kwargs) -> str:
    base_intro = "🎮 PHASE X: Your Phase Description"
    conditional_prompts = _get_phase_X_conditional_prompts(objectives or [])
    
    phase_intro = f"""{base_intro}

{conditional_prompts}

💡 PHASE-SPECIFIC TIPS:
- Tip 1
- Tip 2"""
    
    return build_base_prompt(phase_intro=phase_intro, **kwargs)
```

## Best Practices

1. **Use sequential checks** with `elif` so only ONE section shows at a time
2. **Be specific** about current task (coordinates, actions, etc.)
3. **Include visual cues** (emojis) to make sections scannable
4. **Always provide fallback** for completed objectives
5. **Keep sections focused** on immediate next steps
6. **Test with mock objectives** to verify logic

## Objective Structure

Objectives have these fields:
- `id` (str): Unique identifier (e.g., "story_clock_set")
- `completed` (bool): Whether objective is done
- `description` (str): Human-readable description
- `objective_type` (str): Type of objective
- `storyline` (bool): True for main story objectives

## Finding Objective IDs

Check `agent/simple.py` in `_initialize_storyline_objectives()` for all objective IDs:
- Phase 1: Title sequence objectives
- Phase 2: Littleroot setup objectives
- Phase 3+: Game progression objectives

## Testing

To test your conditional prompts:

```python
from dataclasses import dataclass
from agent.prompts.phase_X import _get_phase_X_conditional_prompts

@dataclass
class MockObjective:
    id: str
    completed: bool

objectives = [
    MockObjective("story_obj_1", True),
    MockObjective("story_obj_2", False),
]

result = _get_phase_X_conditional_prompts(objectives)
print(result)
```

