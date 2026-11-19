"""
Student MCP Server for Text Adventure Games

This is your MCP server submission. Implement the tools that your agent
will use to play text adventure games.

Required tool:
    play_action(action: str) -> str
        Execute a game command and return the result.

Recommended tools:
    memory() -> str
        Return current game state, score, and recent history.
    
    inventory() -> str  
        Return the player's current inventory.
    
    get_map() -> str
        Return a map of explored locations.

Test your server with:
    fastmcp dev submission_template/mcp_server.py

Then open the MCP Inspector in your browser to test the tools interactively.
"""

import sys
import os

# Add parent directory to path to import games module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from games.zork_env import TextAdventureEnv


# =============================================================================
# Create the MCP Server
# =============================================================================

mcp = FastMCP("Student Text Adventure Server")


# =============================================================================
# Game State Management
# =============================================================================

class GameManager:
    """
    Manages the text adventure game state.
    
    TODO: Extend this class to track:
    - Action history (for memory tool)
    - Explored locations (for mapping)
    - Current score and moves
    """
    
    def __init__(self, game: str = "lostpig"):
        self.game_name = game
        self.env = TextAdventureEnv(game)
        self.state = self.env.reset()
        self.history: list[tuple[str, str]] = []
        self.explored_locations: dict[str, set[str]] = {}
        self.current_location: str = self._extract_location(self.state.observation)
    
    def _extract_location(self, observation: str) -> str:
        """Extract location name from observation (usually first line)."""
        lines = observation.strip().split('\n')
        return lines[0] if lines else "Unknown"
    
    def take_action(self, action: str) -> str:
        """Execute a game action and return the result."""
        self.state = self.env.step(action)
        result = self.state.observation
        
        # Track history
        self.history.append((action, result))
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        # Update map
        new_location = self._extract_location(result)
        if new_location == 'Unknown':
            new_location = self.current_location
        if action in ["north", "south", "east", "west", "up", "down", 
                      "enter", "exit", "n", "s", "e", "w", "u", "d","ne", "nw", "se", "sw",
                      "northeast", "northwest", "southeast", "southwest"]:
            if self.current_location not in self.explored_locations:
                self.explored_locations[self.current_location] = set()
            if new_location != self.current_location:
                self.explored_locations[self.current_location].add(f"{action} -> {new_location}")
        self.current_location = new_location
        
        return result
    
    def get_memory(self) -> str:
        """Get a summary of current game state."""
        recent = self.history[-5:] if self.history else []
        recent_str = "\n".join([f"  > {a} -> {r}..." for a, r in recent]) if recent else "  (none yet)"
        
        return f"""Current State:
- Location: {self.current_location}
- Score: {self.state.score} points
- Moves: {self.state.moves}
- Game: {self.game_name}

Recent Actions:
{recent_str}

Current Observation:
{self.state.observation}"""
    
    def get_map(self) -> str:
        """Get a map of explored locations."""
        if not self.explored_locations:
            return "Map: No locations explored yet. Try moving around!"
        
        lines = ["Explored Locations and Exits:"]
        for loc, exits in self.explored_locations.items():
            lines.append(f"\n* {loc}")
            for exit_info in sorted(exits):
                lines.append(f"    -> {exit_info}")
        
        lines.append(f"\n[Current] {self.current_location}")

        lines.append("")  # Blank line between locations
        
        # Add exploration suggestions
        lines.append("=== SUGGESTIONS ===")
        unexplored_exits = []
        for loc, data in self.explored_locations.items():
            # Check for directions that might not have been tried
            all_dirs = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest', 'up', 'down']
            tried_dirs = [e.split(" -> ")[0] for e in data]
            untried = [d for d in all_dirs if d not in tried_dirs]
            if untried and loc == self.current_location:
                unexplored_exits.extend(untried)
        
        if unexplored_exits:
            lines.append(f"Untried directions from {self.current_location}: {', '.join(unexplored_exits)}")

        return "\n".join(lines)
    
    def get_inventory(self) -> str:
        """Get current inventory."""
        items = self.state.inventory if hasattr(self.state, 'inventory') and self.state.inventory else []
        
        if not items:
            return "Inventory: You are empty-handed."
        
        item_names = []
        for item in items:
            item_str = str(item)
            item_lower = item_str.lower()
            if "parent" in item_lower:
                idx = item_lower.index("parent")
                name = item_str[:idx].strip()
                if ":" in name:
                    name = name.split(":", 1)[1].strip()
                item_names.append(name)
            elif ":" in item_str:
                name = item_str.split(":")[1].strip()
                item_names.append(name)
            else:
                item_names.append(item_str)
        
        return f"Inventory: {', '.join(item_names)}"



# Global game manager
_game = GameManager()


def get_game() -> GameManager:
    """Get or initialize the game manager."""
    global _game
    if _game.env is None:
        # Get game from environment variable (set by evaluator)
        game = os.environ.get("GAME", "zork1")
        _game.initialize(game)
    return _game


# =============================================================================
# MCP Tools - IMPLEMENT THESE
# =============================================================================


@mcp.tool()
def play_action(action: str) -> str:
    """
    Execute a game action in the text adventure.
    
    Args:
        action: The command to execute (e.g., 'north', 'take lamp', 'open mailbox')
    
    Returns:
        The game's response to your action
    """
    game = get_game()
    result = game.take_action(action)

    # Add score info
    score_info = f"\n\n[Score: {game.state.score} | Moves: {game.state.moves}]"
    
    if game.state.reward > 0:
        score_info = f"\n\n+{game.state.reward} points! (Total: {game.state.score})"
    
    done_info = ""
    if game.state.done:
        done_info = "\n\nGAME OVER"
    
    return result + score_info + done_info 

@mcp.tool()
def memory() -> str:
    """
    Get a summary of the current game state.
    
    Returns location, score, moves, recent actions, and current observation.
    """
    return get_game().get_memory()


@mcp.tool()
def get_map() -> str:
    """
    Get a map showing explored locations and connections.
    
    Useful for navigation and avoiding getting lost.
    """
    return get_game().get_map()


@mcp.tool()
def inventory() -> str:
    """
    Check what items you are currently carrying.
    """
    return get_game().get_inventory()


# =============================================================================
# Run the server
# =============================================================================

if __name__ == "__main__":
    # This runs the server with stdio transport (for MCP clients)
    mcp.run()
