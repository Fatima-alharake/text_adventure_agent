---
title: Text Adventure Agent Submission
emoji: "\U0001F5FA"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: mit
---

# Text Adventure Agent Submission

## Overview

This is my submission for the Text Adventure Agent assignment. My agent uses the ReAct pattern to play text adventure games via MCP.

## Approach

<!-- Describe your approach here -->

- What strategy does your agent use?
- What tools did you implement in your MCP server?
- Any interesting techniques or optimizations?


In my describition I will be mentioning what problems I noticed in the example submission and how I tried to solve them since I used it as the base.

## Problems: 
The LLM was forgetting easily what already happened, hence each time it needs to look again and that was leading to waste of steps and it wasn't able to do much with a history of 5 steps. In addition the result that was being saved in the history and passed was being truncated hence leading to a lot of information loss, and the system prompt was too simple and lacked a lot of actions especially interactions.

## My Solution:
First, I increased the number of history steps passed to 10, enough for the model to keep track but not too much for it to get confused by the past or be too long.
I am passing all of the result output to the history thus it is able to know all of the details of the game. I also changed the system prompt so it would include new actions, strategies and notes that can help the LLM (these notes/strategies I got them from running the agent multiple times and seeing what were the critical problems that were stopping it from progressing). And most importantly I changed the response to return also a Long_Term_Memory by the LLM itself about the main observations, exits and the location name which I pass every time and I always pass them all, because the idea is that I want to keep a short history summary that it can refer so it doesn't have to re-execute the same commands due to loss of memory.

Example: Suppose the LLM explored the entire fountain room and found everything then went to another room and explored it then went back to the fountain room, but it has forgotten that it has explored the fountain room so it would explore it again which is unnecessary. So instead I use Long_Term_Memory to help it keep track of main information of everything it visited since I am passing the entire Long_Term_Memory each time. Hence, it doesn't re-explore unnecessarly.

Long_Term_Memory: <Location -> key objects and their status -> NPC info -> exits and their state (open, locked, dead_end)>

Long_Term_Memory: Living Room -> trophy case (empty), large oriental rug (reveals trap door when moved), elvish sword (not taken). -> Exits: east (Kitchen), west (wooden door nailed shut).

The way it is being saved is by using a set of explored_locations and a history summary dictionnary. I take the name of the location and add it to the explored_locations if it isn't in the set yet, then I add the history summary it returned in the response to the dictionnary (if there was one for that location it gets overwritten).

In addition, I am validating main actions, example it sometimes say light torch using wall but when it comes to light it can be used only with another word "light torch"
so what I did is create a dictionnary of main actions each with max length of the possible sentence it is in as a value. in the validate tools function I check if the 
action is in that dictionnary I truncate the action passed to the max length thus I make sure that I don't have actions that make no sense.

## Files

| File | Description |
|------|-------------|
| `agent.py` | ReAct agent with `StudentAgent` class |
| `mcp_server.py` | MCP server with game interaction tools |
| `app.py` | Gradio interface for HF Space |
| `requirements.txt` | Additional dependencies |

## How to Submit

1. Fork the template Space: `https://huggingface.co/spaces/LLM-course/text-adventure-template`
2. Clone your fork locally
3. Implement your agent in `agent.py` and `mcp_server.py`
4. Test locally (see below)
5. Push your changes to your Space
6. Submit your Space URL on the course platform

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test the MCP server interactively
fastmcp dev mcp_server.py

# Run your agent on a game
python run_agent.py --agent . --game lostpig -v -n 20

# Run evaluation
python -m evaluation.evaluate -s . -g lostpig -t 3

```
