"""
Student Agent for Text Adventure Games

This is your submission file. Implement the StudentAgent class to play
text adventure games using the MCP server you also implement.

Your agent should:
1. Connect to the MCP server via the provided client
2. Use the ReAct pattern (Thought -> Action -> Observation)
3. Call MCP tools to interact with the game
4. Maximize the game score within the step limit

Required method:
    async def run(self, client, game, max_steps, seed, verbose) -> RunResult

The 'client' is a FastMCP Client already connected to your MCP server.
Use it to call tools like: await client.call_tool("play_action", {"action": "look"})

Tips:
- Start by looking around and understanding your environment
- Keep track of visited locations to avoid loops
- Pick up useful items (lamp, sword, etc.)
- The seed parameter should be used to set your LLM's seed for reproducibility
"""
import torch
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional
from random import choice

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from transformers.utils import logging
logging.set_verbosity_error()

# Load environment variables
load_dotenv()

# Set USE_LOCAL_MODEL=1 in your .env to use a locally downloaded model
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "0").strip() in ("1", "true", "yes")
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# =============================================================================
# LLM Configuration - DO NOT MODIFY
# =============================================================================

# Model to use (fixed for fair evaluation)
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Initialize the LLM client based on mode
_local_pipeline = None

if USE_LOCAL_MODEL:
    import torch
    from transformers import pipeline as _hf_pipeline

    print("using local model")
    
    _local_pipeline = _hf_pipeline(
        "text-generation",
        model=LOCAL_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    LLM_CLIENT = None
else:
    _hf_token = os.getenv("HF_TOKEN")
    if not _hf_token:
        raise ValueError("HF_TOKEN not found. Set it in your .env file.")
    LLM_CLIENT = InferenceClient(token=_hf_token)


def call_llm(prompt: str, system_prompt: str, seed: int, max_tokens: int = 300) -> str:
    """
    Call the LLM with the given prompt. Use this function in your agent.
    
    Args:
        prompt: The user prompt (current game state, history, etc.)
        system_prompt: The system prompt (instructions for the agent)
        seed: Random seed for reproducibility
        max_tokens: Maximum tokens in response (default: 300)
        
    Returns:
        The LLM's response text
        
    Example:
        response = call_llm(
            prompt="You are in a forest. What do you do?",
            system_prompt=SYSTEM_PROMPT,
            seed=42,
        )
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if USE_LOCAL_MODEL and _local_pipeline is not None:
        outputs = _local_pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=0.0001,  # Near-deterministic (0.0 unsupported by some backends)
            do_sample=True,
        )
        return outputs[0]["generated_text"][-1]["content"]

    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,  # Deterministic for reproducibility
        max_tokens=max_tokens,
        seed=seed,
    )

    return response.choices[0].message.content

@dataclass
class RunResult:
    """Result of running the agent. Do not modify this class."""
    final_score: int
    max_score: int
    moves: int
    locations_visited: set[str]
    game_completed: bool
    error: Optional[str] = None
    history: list[tuple[str, str, str]] = field(default_factory=list)


# =============================================================================
# System Prompt - Customize this for your agent
# =============================================================================


SYSTEM_PROMPT = """You are an expert interactive fiction agent playing any Jericho-supported text adventure game.
Your objective is to explore efficiently, solve puzzles, gather useful items, and maximize score when possible.
Act strategically, avoid repeating failed actions, and adapt to the specific mechanics of the current game.

GENERAL PRINCIPLES:
- Think step by step.
- Base decisions only on the current observation, inventory, map, and memory.
- Do not assume the game follows a specific theme (fantasy, sci-fi, etc.).
- Different games have different mechanics, learn them from interaction.
- Track visited locations, tried exits, locked objects, hazards, and NPC behavior.

AVAILABLE TOOLS (use these via MCP):
1. play_action - Execute game commands (north, take lamp, open mailbox, etc.)
2. memory - Get current game state, score, and recent history
3. get_map - See explored locations and connections
4. inventory - Check what you're carrying

There are only TWO possible tool calls:

1) To perform an action in the game:
TOOL: play_action
ARGS: {"action": "<action_string>"}
- Replace <action_string> with the exact action you want to take.
- This includes movement (e.g., "north"), interaction (e.g., "open door"), or sensing (e.g., "listen").
- play_action is the ONLY tool that takes arguments.

2) To access memory, get_map:
TOOL: memory
ARGS: {}

- memory, get_map take NO arguments.
- The ARGS must be exactly {}.

Your entire response must be EXACTLY one of the two formats above.
Do not output anything else.

VALID GAME COMMANDS for play_action:
- Movement: north, south, east, west, northeast, northwest, southeast, southwest, up, down, enter, exit
- Objects: take <item>, drop <item>, open <object>, close <object>, look, look at <object>, look in <container>, examine <object>, read <object>, climb <object>, 
move <object>, push <object>, pull <object>, turn on <item>, turn off <item>, unlock <object> with <item>, lock <object> with <item>, put <item> in <container>, get <item> from <container>
- NPC interaction: talk to <character>, ask <character> about <topic>, give <item> to <character>, attack <creature> with <weapon>
- Passive sensing: wait, listen, smell, feel, shout
- Other: inventory

FORBIDDEN (will NOT work): check, inspect, search, grab, use, help

STRATEGY RULES:

1. Examine new objects before manipulating them.
2. Only interact with objects explicitly mentioned in the observation or inventory.
3. Take portable and potentially useful items.
4. Track locked doors and containers; only retry them after obtaining new tools.
5. Mark exits that lead nowhere or loop back as DEAD_END in reasoning and avoid retrying unless state changes.
6. If progress stalls:
   - Try passive sensing actions.
   - Re-examine objects.
   - Re-check containers (look in, under, behind if supported).
   - Revisit previously blocked paths if inventory changed.
7. Use light sources before entering dark areas.
8. Pay attention to score changes they signal meaningful progress.
9. Some objects are decorative. If repeated interaction yields nothing new, deprioritize them.
10. NPCs may provide hints. If dialogue yields no new information, do not repeat it.
11. If you can climb an object do so.
12. If the game rejects a verb, switch to another standard verb instead of retrying.
13. If an object is described as closed, locked, hinged, clasped, sealed, or a container, try to open it. 
14. If you have thoroughly explored the current location without finding anything new, check available exits. If all exits and possible movements have been attempted with no progress, perform passive sensing actions to gather additional information, they might give you hints.
15. If you need something first check if you have it in your inventory, if not look for it.

INVENTORY TOOL USAGE:
If:
- An object is visible but NOT directly takeable, AND
- The object is described as distant, unreachable, inside, above, stuck, fragile, dangerous, or obstructed

THEN:
- Check inventory for a tool that could logically extend reach, apply force, protect, or interact indirectly.

Attempt interaction using a STANDARD verb:
- get <object> with <tool>
- move <object> with <tool>
- pull <object> with <tool>
- push <object> with <tool>
- attack <object> with <tool> 

RESPONSE FORMAT (strict):

THOUGHT: <Explain why this action advances exploration or puzzle-solving based on current state. Mention if testing an exit, checking a container, or following a clue.>
TOOL: <tool_name>
ARGS: <JSON arguments>

Example:

THOUGHT: I have examined visible objects here. The north and east exits can't be explored, south leads me back home, west is a dead_end. I should try passive sensing.
TOOL: play_action
ARGS: {"action": "listen"}

THOUGHT: Let me check my current state and score.
TOOL: memory
ARGS: {}
"""

# =============================================================================
# Student Agent - IMPLEMENT THIS CLASS
# =============================================================================

class StudentAgent:
    """
    Your ReAct agent implementation.
    
    TODO:
    1. Implement the run() method with the ReAct loop
    2. Parse LLM responses to extract tool calls
    3. Track state and avoid loops
    
    Use the provided call_llm() function to interact with the LLM.
    """ 
    def __init__(self):
        """Initialize the agent state."""
        self.history: list[dict] = []
        self.recent_actions: list[str] = []
        self.score: int = 0
        self.explored_locations = set()
        self.COMMANDS = [
            "look", "inventory",
            "wait", "listen"
        ]
        self.Long_Term_Memory = {}
        self.last_actions = []


    async def run(
        self,
        client,
        game: str,
        max_steps: int,
        seed: int,
        verbose: bool = False,
    ) -> RunResult:
        """Run the agent for a game session."""
        locations_visited = set()
        history = []
        moves = 0

        # Get list of available tools
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        
        # Get initial observation
        result = await client.call_tool("play_action", {"action": "look"})
        observation = self._extract_result(result)

        # Track initial location
        location = observation.split("\n")[0] if observation else "Unknown"
        locations_visited.add(location)

        tool_args = {"action": "look"}
        tool_name = "play_action"
        # Update history
        self.history.append({
            "thought": "",
            "tool": tool_name,
            "args": tool_args,
            "result": observation,
        })
        
        # Track score from observation
        self._update_score(observation)
        
        # Record in result history
        history.append(("", f"{tool_name} ({tool_args})", observation))

        self.explored_locations.add(location)
        
        observation_formatted = location + " → " + observation.split("\n")[1] if observation else ""
        self.Long_Term_Memory[location] = observation_formatted
        
        if verbose:
            print(f"\n--- Initial Observation ---")
            print(f"[RESULT] {observation}")
        
        # Main ReAct loop
        for step in range(1, max_steps + 1):
            # Build prompt with context
            prompt = self._build_prompt(observation)
            
            # Call LLM for reasoning (use step-based seed for variety)
            response = call_llm(prompt, SYSTEM_PROMPT, seed + step)
            
            # Parse the response
            thought, tool_name, tool_args = self._parse_response(response, tool_names)
            
            if verbose:
                print(f"\n--- Step {step} ---")
                print(f"[THOUGHT] {thought}")
                print(f"[TOOL] {tool_name}({tool_args})")

            # Validate and fix common issues
            tool_name, tool_args = self._validate_tool_call(tool_name, tool_args, tool_names)
            newest_location = list(self.Long_Term_Memory.values())[-1].split("→")
            if len(newest_location) >=3:
                newest_location = newest_location[1]
            else:
                newest_location = newest_location[0]

            # Loop detection
            if tool_name == "play_action":
                action = tool_args.get("action", "look")
                self.recent_actions.append(action)
                if len(self.recent_actions) > 5:
                     self.recent_actions = self.recent_actions[-5:]
                
                occ_actions = self.last_actions.count((newest_location, action))

                # Detect loops - if same action 3 times, force "look"
                if (len(self.recent_actions) >= 3 and len(set(self.recent_actions[-3:])) == 1) or \
                    (len(self.recent_actions) >= 4 and self.recent_actions[-4:-2] == self.recent_actions[-2:]) or \
                    (occ_actions == 4):
                    if verbose:
                        if occ_actions:
                            print(f"You have chosen this action {action} more than 3 times in this location {newest_location}!")
                        else:
                            print(f"[WARNING] Loop detected")
                    new_action = "look"
                    directions = self.COMMANDS.copy() 
                    if self.recent_actions[-1] in directions:
                        directions.remove(self.recent_actions[-1])
                    new_action = choice(directions) if directions else "look"

                    tool_args = {"action": new_action}
                    self.recent_actions.pop()
                    self.recent_actions.append(new_action)
                    if len(self.recent_actions) > 5:
                        self.recent_actions = self.recent_actions[-5:]

                    print(f"[LOOP FIX] Changing action to '{new_action}' to break loop.")
                moves += 1
            else:
                self.recent_actions.append(tool_name)

            # Execute the tool
            try:
                result = await client.call_tool(tool_name, tool_args)
                observation = self._extract_result(result)

                summary_prompt = "OLD_LOCATION: " + newest_location + " ACTION_TAKEN: " + self.recent_actions[-1] + " OBSERVATION: " + observation + " PREVIOUS_SUMMARIES: " + "\n\n".join(list(self.Long_Term_Memory.values()))

                new_location, history_summary = self._memory_summarizer(seed, step, summary_prompt)
                if new_location not in self.explored_locations:
                    self.explored_locations.add(new_location)
                self.Long_Term_Memory[new_location] = history_summary

                if verbose:
                    print(f"[RESULT] {observation}...")
                    print("[new location]", new_location)
                    print("[history summary]", history_summary)

                self.last_actions.append((newest_location, self.recent_actions[-1]))

            except Exception as e:
                observation = f"Error: {e}"
                if verbose:
                    print(f"[ERROR] {e}")
            
            # Track location
            location = observation.split("\n")[0] if observation else "Unknown"
            locations_visited.add(location)
            
            # Update history
            self.history.append({
                "thought": thought,
                "tool": tool_name,
                "args": tool_args,
                "result": observation,
            })
            if len(self.history) > 10:
                self.history = self.history[-10:]
            
            # Track score from observation
            self._update_score(observation)
            
            # Record in result history
            history.append((thought, f"{tool_name}({tool_args})", observation))
            
            # Check for game over
            if self._is_game_over(observation):
                if verbose:
                    print("\n*** GAME OVER ***")
                break
        
        return RunResult(
            final_score=self.score,
            max_score=350,
            moves=moves,
            locations_visited=locations_visited,
            game_completed=self._is_game_over(observation),
            history=history,
        )
    
    def _build_prompt(self, observation: str) -> str:
        """Build the prompt for the LLM with context."""
        parts = []
        
        parts.append(f"Current Score: {self.score}")
        
        # Recent history
        if self.history:
            parts.append("\nHistory (last 7 steps) logs of each action taken followed by the result:")
            for entry in self.history[-10:]:
                action = entry.get("args", {}).get("action", entry["tool"])
                result_short = entry["result"]
                parts.append(f"> {action} -> {result_short}")
            
            if len(self.history) >= 2:
                if self.history[-1]["result"] == self.history[-2]["result"]:
                    parts.append("\n[CRITICAL: Your last action did not change the environment. DO NOT repeat it! Try a different verb or direction.]")
            

        history_str = "\n".join(parts)
        long_term_memory = "\n".join(list(self.Long_Term_Memory.values())) if self.Long_Term_Memory else "(none yet)"

        return f"""
        {history_str}

        LONG_TERM_MEMORY:
        {long_term_memory}
        
        current observation:
        {observation}

        What do you do next?
        """

    def _memory_summarizer(self, seed, step, prompt):
        sys_prompt = """
            You are a persistent memory summarizer for a Jericho text adventure game.

            INPUT YOU WILL RECEIVE :
            - OLD_LOCATION:
            - ACTION_TAKEN:
            - OBSERVATION:
            - PREVIOUS_SUMMARIES: (may be empty)

            LOCATION RULES

            1. You MUST determine the NEW_LOCATION from the OBSERVATION.
            - The NEW_LOCATION is ALWAYS the first line of the OBSERVATION UNLESS the OBSERVATION explicitly says you moved. Sometimes the first line doesn't mention the location in that case it's the old location.
            - If the player has not actually changed rooms/areas, NEW_LOCATION = OLD_LOCATION.
            - If the observation does NOT indicate a location change, then NEW_LOCATION = OLD_LOCATION.

            2. If NEW_LOCATION == OLD_LOCATION:
            - Find the previous summary whose location (the text before the first "→") matches OLD_LOCATION.
            - Update and extend that summary.
            - Do NOT create a new summary.

            3. If NEW_LOCATION != OLD_LOCATION:
            - Create a completely new summary for that new location.
            - Do NOT mix it with summaries of other locations.

            SUMMARY RULES

            - Only ADD new details or UPDATE information if the observation provides new facts, in case you are updating a previous summary.
            - Do NOT simplify or compress richer existing descriptions.
            - Always preserve all previously known exits and details for the location.
            - Never delete known exits unless clearly blocked or changed.
            - Combine previous summary with new information; do NOT overwrite older details.

            - Do NOT copy the full observation.
            - Extract only important information:
            • new environmental details
            • items discovered or interacted with
            • important events or outcomes
            • exit information and their status

            Be concise but complete.
            RESPONSE FORMAT: 
            SUMMARY: OLD_LOCATION → NEW_LOCATION → Combined detailed description including all known important facts. → Exits: direction (description/status), direction (description/status)
            """

        response = call_llm(prompt, sys_prompt, seed + step)
        response = response.split("SUMMARY:")[1].strip()
        new_location = response.split("→")[1].strip() 
        return new_location, response
    
    def _parse_response(self, response: str, valid_tools: list[str]) -> tuple[str, str, dict]:
        """Parse the LLM response to extract thought, tool, and arguments."""
        thought = "No reasoning provided"
        tool_name = "play_action"
        tool_args = {"action": "look"}
        lines = response.strip().split("\n")
        
        for line in lines:
            line_clean = line.strip()
            line_upper = line_clean.upper()
            
            if line_upper.startswith("THOUGHT:"):
                thought = line_clean.split(":", 1)[1].strip()

            elif line_upper.startswith("TOOL:"):
                raw_tool = line_clean.split(":", 1)[1].strip().lower()
                raw_tool = raw_tool.replace("**", "").replace("*", "").replace("`", "")
                raw_tool = raw_tool.split()[0] if raw_tool else "play_action"
                tool_name = raw_tool
            
            elif line_upper.startswith("ARGS:"):
                args_part = line_clean.split(":", 1)[1].strip()
                try:
                    args_part = args_part.replace("'", '"')
                    tool_args = json.loads(args_part)
                except json.JSONDecodeError:
                    match = re.search(r'"action"\s*:\s*"([^"]+)"', args_part)
                    if match:
                        tool_args = {"action": match.group(1)}
                    else:
                        tool_args = {"action": "look"}
        return thought, tool_name, tool_args
    
    def _validate_tool_call(self, tool_name: str, tool_args: dict, valid_tools: list[str]) -> tuple[str, dict]:
        """Validate and fix common tool call issues."""
        # Fix tool name

        if tool_name in self.COMMANDS:
            return "play_action", {"action": tool_name}

        if tool_name not in valid_tools:
            if tool_name in ["action", "do", "command"]:
                tool_name = "play_action"
            elif tool_name in ["map", "location"]:
                tool_name = "get_map"
            elif tool_name in ["mem", "state", "status"]:
                tool_name = "memory"
            elif tool_name in ["inv", "items"]:
                tool_name = "inventory"
            else:
                if "action" not in tool_args or not tool_args["action"]:
                    tool_args["action"] = tool_name
                tool_name = "play_action"

        # Fix action verbs
        if tool_name == "play_action":
            action = tool_args.get("action", "look")
            
            action_splitted = action.split(" ")[0]

            if action_splitted == "move":
                action_splitted[0] = "go"
                action = " ".join(action_splitted)

            invalid_verb_map = {
                "check": "look",
                "inspect": "look",
                "search": "look",
                "grab": "take",
                "pick": "take",
                "use": "look",
                "investigate": "look",
                "look for path": "look",
                "look around": "look",
            }

            words = action.lower().split()
            if words and words[0] in invalid_verb_map:
                words[0] = invalid_verb_map[words[0]]
                action = " ".join(words)

            action = action.lower().strip()
            action = action.replace("**", "").replace("*", "").replace("`", "")
            action = " ".join(action.split())
            
            tool_args["action"] = action
        
        return tool_name, tool_args
    
    def _extract_result(self, result) -> str:
        """Extract text from MCP tool result."""
        if hasattr(result, 'content') and result.content:
            return result.content[0].text
        if isinstance(result, list) and result:
            return result[0].text if hasattr(result[0], 'text') else str(result[0])
        return str(result)
    
    def _update_score(self, text: str) -> None:
        """Update score from game text."""
        patterns = [
            r'Score:\s*(\d+)',
            r'score[:\s]+(\d+)',
            r'\[Score:\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.score = max(self.score, int(match.group(1)))
    
    def _is_game_over(self, text: str) -> bool:
        """Check if the game is over."""
        game_over_phrases = [
            "game over",
            "you have died",
            "you are dead",
            "you have lost",
            "*** you have died ***",
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in game_over_phrases)




# =============================================================================
# For local testing
# =============================================================================

async def test_agent():
    """Test the agent locally."""
    from fastmcp import Client
    
    # Path to your MCP server
    server_path = "mcp_server.py"
    
    agent = StudentAgent()
    
    async with Client(server_path) as client:
        result = await agent.run(
            client=client,
            game="lostpig",
            max_steps=10,
            seed=42,
            verbose=True,
        )
        
        print(f"\nFinal Score: {result.final_score}")
        print(f"Moves: {result.moves}")
        print(f"Locations: {result.locations_visited}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent())
