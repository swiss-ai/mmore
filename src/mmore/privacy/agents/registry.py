"""Global registry mapping tool names to callables.

Tools are registered programmatically (typically via the ``@register_tool``
decorator) and referenced from agent YAML configs by name.
"""

from typing import Callable, Dict, List, Optional

tool_registry: Dict[str, Callable] = {}


class ToolNotRegisteredError(KeyError):
    """Raised when an agent config references a tool name that was never registered."""


def register_tool(name: str, fn: Optional[Callable] = None) -> Callable:
    """Register ``fn`` under ``name``. Usable as a decorator or direct call.

    Examples:
        >>> @register_tool("greet")
        ... def greet(who: str) -> str: ...

        >>> register_tool("greet", greet_fn)
    """
    if fn is not None:
        tool_registry[name] = fn
        return fn

    else:

        def decorator(f: Callable) -> Callable:
            tool_registry[name] = f
            return f

        return decorator


def resolve_tools(names: List[str]) -> List[Callable]:
    """Resolve a list of tool names into callables."""
    resolved: List[Callable] = []
    for tool in names:
        if tool not in tool_registry:
            raise ToolNotRegisteredError(
                f"Tool '{tool}' is not registered. "
                f"Available tools: {sorted(tool_registry.keys())}"
            )
        resolved.append(tool_registry[tool])
    return resolved


def list_tools() -> List[str]:
    """Return the names of all currently registered tools."""
    return list(tool_registry.keys())
