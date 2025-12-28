# LangGraph Multi-Agent Systems - Important Concepts

## Table of Contents
- [Core Concepts](#core-concepts)
- [State Management](#state-management)
- [Agent Architectures](#agent-architectures)
- [Tool Integration](#tool-integration)
- [Graph Construction](#graph-construction)
- [Memory & Persistence](#memory--persistence)
- [Common Patterns](#common-patterns)
- [Interview Questions](#interview-questions)

---

## Core Concepts

### What is LangGraph?
LangGraph is a framework for building stateful, multi-agent applications with LLMs. It's built on top of LangChain and uses **graph-based architecture** where nodes represent computations and edges represent flow.

### Key Components
1. **StateGraph**: Main graph builder that manages state transitions
2. **Nodes**: Functions that process and update state
3. **Edges**: Define flow between nodes (simple or conditional)
4. **State**: Shared data structure passed between nodes
5. **Checkpointer**: Enables memory/persistence across invocations

---

## State Management

### MessagesState
Built-in state for chat applications with automatic message handling:

```python
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    # MessagesState provides:
    # - messages: List[BaseMessage]
    # - add_messages() function for appending
    pass
```

### Custom State with TypedDict
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-append messages
    research_data: str
    next_agent: str
    task_complete: bool
```

**Key Interview Point**: The `Annotated[list, add_messages]` pattern ensures messages are **appended** rather than replaced.

---

## Agent Architectures

### 1. Simple Linear Flow
**Pattern**: `start → agent1 → agent2 → end`

```python
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)
```

**Use Cases**: Sequential processing, pipeline workflows

---

### 2. ReAct Agent (Reasoning + Acting)
**Pattern**: LLM decides → calls tools → processes results → decides again

```python
def agent(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # Loop back to agent
```

**Key Interview Point**: The **loop from tools back to agent** enables iterative reasoning.

---

### 3. Supervisor Pattern
**Pattern**: Supervisor routes tasks to specialized agents

```python
def router(state):
    next_agent = state.get("next_agent", "supervisor")
    if next_agent == "end":
        return END
    return next_agent

workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        END: END
    }
)
```

**Use Cases**: Complex workflows requiring orchestration, dynamic routing

---

## Tool Integration

### Defining Tools
```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    search = TavilySearchResults(max_results=3)
    return str(search.invoke(query))
```

### Binding Tools to LLM
```python
llm_with_tools = llm.bind_tools([search_web, calculator])
```

### Executing Tools
```python
from langgraph.prebuilt import ToolNode

def execute_tools(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_node = ToolNode(tools)
        return tool_node.invoke(state)
    return state
```

**Critical Interview Point**: `bind_tools()` only tells the LLM about available tools. You must **explicitly execute** tool calls using `ToolNode` or custom logic.

---

## Graph Construction

### Basic Graph Pattern
```python
# 1. Create graph
workflow = StateGraph(State)

# 2. Add nodes (functions)
workflow.add_node("node_name", node_function)

# 3. Define edges
workflow.add_edge(START, "first_node")
workflow.add_edge("first_node", "second_node")

# 4. Conditional routing
workflow.add_conditional_edges(
    "node",
    routing_function,
    {"path1": "node1", "path2": "node2", END: END}
)

# 5. Compile
graph = workflow.compile()
```

### Conditional Edges vs Simple Edges

| Type | When to Use | Example |
|------|-------------|---------|
| **Simple Edge** | Fixed, deterministic flow | `add_edge("A", "B")` |
| **Conditional Edge** | Dynamic routing based on state | `add_conditional_edges("A", router, {...})` |

---

## Memory & Persistence

### Adding Memory (Checkpointer)
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Use with thread IDs
config = {"configurable": {"thread_id": "1"}}
response = graph.invoke({"messages": "Hi"}, config)
```

**Key Interview Point**: 
- **Without checkpointer**: Each invocation is stateless
- **With checkpointer**: State persists across invocations within same thread

### Thread IDs
- Different thread IDs = separate conversation sessions
- Same thread ID = continuous conversation with memory

---

## Common Patterns

### 1. Message Input Formats

✅ **Correct Ways**:
```python
# Auto-converted to HumanMessage
graph.invoke({"messages": "Hello"})

# Explicit message object
graph.invoke({"messages": [HumanMessage(content="Hello")]})
```

❌ **Wrong**:
```python
graph.invoke("Hello")  # Missing state structure
```

---

### 2. Routing Function Pattern
```python
def should_continue(state):
    """Return next node name or END"""
    last_message = state["messages"][-1]
    
    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    
    # Check custom routing field
    next_agent = state.get("next_agent")
    if next_agent == "end":
        return END
    
    return next_agent or END
```

---

### 3. Agent Node Pattern
```python
def agent_function(state: AgentState) -> dict:
    """
    Standard agent pattern:
    1. Extract state
    2. Create system prompt
    3. Call LLM
    4. Return state updates
    """
    messages = state["messages"]
    
    system_msg = SystemMessage(content="You are an expert...")
    response = llm.invoke([system_msg] + messages)
    
    return {
        "messages": [response],
        # Optional: routing info
        "next_agent": "writer"
    }
```

---

## Interview Questions

### Q1: What's the difference between LangChain and LangGraph?
**Answer**: 
- **LangChain**: Sequential chains, linear processing
- **LangGraph**: Graph-based, supports cycles, loops, conditional routing, and complex multi-agent workflows

### Q2: How does tool execution work in LangGraph?
**Answer**:
1. LLM with `bind_tools()` generates `tool_calls` in response
2. These are **not executed automatically**
3. Must use `ToolNode` or custom executor to run tools
4. Tool results returned as `ToolMessage`
5. Loop back to LLM for processing results

### Q3: Explain the ReAct pattern
**Answer**: 
- **Re**asoning + **Act**ing loop
- LLM reasons about next action → executes tool → observes result → reasons again
- Implemented via: `agent → conditional_edge → tools → edge_back_to_agent`

### Q4: When do you need a Supervisor agent?
**Answer**:
- Complex workflows with 3+ specialized agents
- Dynamic routing based on task requirements
- Need centralized orchestration and decision-making
- Examples: Research team (researcher + analyst + writer)

### Q5: What's the purpose of MessagesState?
**Answer**:
- Provides standard structure for chat applications
- Automatically handles message list updates
- Built-in `add_messages` reducer (appends instead of replacing)
- Compatible with LangChain message types

### Q6: How do you implement memory in LangGraph?
**Answer**:
```python
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user_123"}}
graph.invoke(input, config)  # State persists per thread_id
```

### Q7: What's the difference between `tools_condition` and custom routing?
**Answer**:
- **`tools_condition`**: Built-in prebuilt function, checks if last message has tool_calls
- **Custom routing**: Write your own logic based on state, supports complex multi-path decisions

### Q8: Common mistake when using bind_tools()?
**Answer**: 
**Mistake**: Thinking `bind_tools()` executes tools automatically
**Reality**: It only informs LLM about available tools. Must add `ToolNode` and conditional edges to actually execute.

---

## Best Practices

### 1. State Design
- Keep state minimal and focused
- Use `MessagesState` for chat applications
- Add custom fields only when needed
- Use `Annotated[list, add_messages]` for message lists

### 2. Error Handling
```python
def safe_agent(state):
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}
```

### 3. Graph Visualization
```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

### 4. Testing Patterns
- Test individual agent functions separately
- Use different thread IDs for isolated tests
- Verify state updates at each node
- Check routing logic with debug prints

---

## Common Pitfalls

❌ **Don't**:
- Forget to compile graph before invoking
- Mix thread IDs when testing memory
- Assume tools execute automatically
- Create circular dependencies without exit conditions

✅ **Do**:
- Always return dict with state updates from nodes
- Use conditional edges for dynamic routing
- Add END edges to prevent infinite loops
- Initialize all state fields properly

---

## Production Considerations

1. **Error Recovery**: Add try-catch in agent nodes
2. **Rate Limiting**: Handle API rate limits gracefully  
3. **Logging**: Log state transitions and decisions
4. **Monitoring**: Track agent performance and tool usage
5. **Persistence**: Use database-backed checkpointers for production
6. **Security**: Validate tool inputs, sanitize outputs

---

## Resources

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangChain Docs**: https://python.langchain.com/
- **GitHub Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples

---

## Key Takeaways for Interviews

1. **LangGraph = Stateful Graphs** for multi-agent systems
2. **MessagesState** is the standard for chat applications
3. **ReAct Pattern** = Reasoning + Acting in loops
4. **Tools must be executed explicitly** via ToolNode
5. **Conditional edges** enable dynamic routing
6. **Checkpointer** = Memory across invocations
7. **Thread IDs** = Separate conversation sessions
8. **Supervisor Pattern** for complex orchestration

---

*Last Updated: December 28, 2025*
