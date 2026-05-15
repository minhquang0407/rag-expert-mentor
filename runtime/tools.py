import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schemas import ToolArtifact, ToolResult


class ToolRegistry:
    """Small MCP-like registry for safe template-based Python tools."""

    DEFAULT_PERMISSIONS = {
        "plot_graph": {"concept", "math", "example"},
        "plot_matrix_heatmap": {"concept", "formula", "math", "example"},
        "plot_function": {"math", "example"},
        "run_algorithm_benchmark": {"algorithm"},
    }

    def __init__(
        self,
        *,
        enabled: bool = False,
        python_enabled: bool = False,
        artifact_root: str = "runtime_artifacts",
        max_runtime_seconds: int = 5,
        permissions: Optional[Dict[str, set[str]]] = None,
    ):
        self.enabled = enabled
        self.python_enabled = python_enabled
        self.artifact_root = Path(artifact_root)
        self.max_runtime_seconds = max_runtime_seconds
        self.permissions = permissions or self.DEFAULT_PERMISSIONS
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def execute_tool(
        self,
        *,
        agent_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        run_id: str = "manual",
        call_id: Optional[str] = None,
    ) -> ToolResult:
        call_id = call_id or str(uuid.uuid4())
        if not self.enabled:
            return self._error(call_id, agent_name, tool_name, "Tools are disabled.")
        if not self.python_enabled:
            return self._error(call_id, agent_name, tool_name, "Python tools are disabled.")
        if tool_name not in self.permissions:
            return self._error(call_id, agent_name, tool_name, f"Unknown tool: {tool_name}")
        if agent_name not in self.permissions[tool_name]:
            return self._error(call_id, agent_name, tool_name, f"Agent {agent_name} is not allowed to use {tool_name}.")

        try:
            if tool_name == "plot_matrix_heatmap":
                return self._plot_matrix_heatmap(call_id, agent_name, arguments, run_id)
            if tool_name == "plot_graph":
                return self._plot_graph(call_id, agent_name, arguments, run_id)
            if tool_name == "plot_function":
                return self._plot_function(call_id, agent_name, arguments, run_id)
            if tool_name == "run_algorithm_benchmark":
                return self._run_algorithm_benchmark(call_id, agent_name, arguments, run_id)
            return self._error(call_id, agent_name, tool_name, f"Tool is not implemented: {tool_name}")
        except Exception as exc:
            return self._error(call_id, agent_name, tool_name, str(exc))

    def _run_dir(self, run_id: str) -> Path:
        path = self.artifact_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact(self, artifact_type: str, path: Optional[Path], title: str, metadata: Optional[Dict[str, Any]] = None) -> ToolArtifact:
        return ToolArtifact(
            artifact_id=str(uuid.uuid4()),
            artifact_type=artifact_type,
            path=str(path) if path else None,
            title=title,
            metadata=metadata or {},
        )

    def _error(self, call_id: str, agent_name: str, tool_name: str, message: str) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            agent_name=agent_name,
            status="error",
            content=message,
            stderr=message,
        )

    def _plot_matrix_heatmap(self, call_id: str, agent_name: str, arguments: Dict[str, Any], run_id: str) -> ToolResult:
        import matplotlib.pyplot as plt
        import numpy as np

        matrix = arguments.get("matrix") or []
        title = str(arguments.get("title") or "Matrix Heatmap")
        arr = np.array(matrix, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("matrix must be a non-empty 2D numeric array")
        if arr.shape[0] > 20 or arr.shape[1] > 20:
            raise ValueError("matrix dimensions must be <= 20x20")

        path = self._run_dir(run_id) / f"matrix_heatmap_{call_id[:8]}.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(arr, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:g}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

        return ToolResult(
            call_id=call_id,
            tool_name="plot_matrix_heatmap",
            agent_name=agent_name,
            status="success",
            content=f"Generated matrix heatmap: {title}",
            artifacts=[self._artifact("image", path, title, {"shape": list(arr.shape)})],
        )

    def _plot_graph(self, call_id: str, agent_name: str, arguments: Dict[str, Any], run_id: str) -> ToolResult:
        import matplotlib.pyplot as plt
        import networkx as nx

        nodes = arguments.get("nodes") or []
        edges = arguments.get("edges") or []
        title = str(arguments.get("title") or "Graph Visualization")
        if len(nodes) > 50 or len(edges) > 100:
            raise ValueError("graph tool supports at most 50 nodes and 100 edges")

        graph = nx.Graph()
        graph.add_nodes_from([str(node) for node in nodes])
        graph.add_edges_from([(str(edge[0]), str(edge[1])) for edge in edges if isinstance(edge, list) and len(edge) >= 2])

        path = self._run_dir(run_id) / f"graph_{call_id[:8]}.png"
        fig, ax = plt.subplots(figsize=(6, 4.5))
        pos = nx.spring_layout(graph, seed=42)
        nx.draw_networkx(graph, pos=pos, ax=ax, node_color="#8ecae6", edge_color="#64748b", font_size=9)
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

        return ToolResult(
            call_id=call_id,
            tool_name="plot_graph",
            agent_name=agent_name,
            status="success",
            content=f"Generated graph visualization: {title}",
            artifacts=[self._artifact("image", path, title, {"nodes": len(nodes), "edges": len(edges)})],
        )

    def _plot_function(self, call_id: str, agent_name: str, arguments: Dict[str, Any], run_id: str) -> ToolResult:
        import matplotlib.pyplot as plt
        import numpy as np

        expression = str(arguments.get("expression") or "x")
        title = str(arguments.get("title") or f"f(x) = {expression}")
        x_min = float(arguments.get("x_min", -5))
        x_max = float(arguments.get("x_max", 5))
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")

        allowed_names = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "pi": math.pi}
        x = np.linspace(x_min, x_max, 400)
        y = eval(expression, {"__builtins__": {}}, {"x": x, **allowed_names})

        path = self._run_dir(run_id) / f"function_{call_id[:8]}.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, color="#7c3aed", linewidth=2)
        ax.axhline(0, color="#94a3b8", linewidth=0.8)
        ax.axvline(0, color="#94a3b8", linewidth=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

        return ToolResult(
            call_id=call_id,
            tool_name="plot_function",
            agent_name=agent_name,
            status="success",
            content=f"Generated function plot: {title}",
            artifacts=[self._artifact("image", path, title, {"expression": expression})],
        )

    def _run_algorithm_benchmark(self, call_id: str, agent_name: str, arguments: Dict[str, Any], run_id: str) -> ToolResult:
        algorithm = str(arguments.get("algorithm") or "degree_count")
        sizes = [int(size) for size in arguments.get("sizes", [10, 100, 1000])][:5]
        sizes = [size for size in sizes if 1 <= size <= 10000]
        if algorithm != "degree_count":
            raise ValueError("v1 benchmark supports only algorithm='degree_count'")

        rows = []
        start_all = time.perf_counter()
        for n in sizes:
            edges = [(i, (i + 1) % n) for i in range(n)]
            t0 = time.perf_counter()
            degrees = [0] * n
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rows.append({"n": n, "edges": len(edges), "elapsed_ms": round(elapsed_ms, 6), "max_degree": max(degrees)})
            if time.perf_counter() - start_all > self.max_runtime_seconds:
                break

        path = self._run_dir(run_id) / f"benchmark_{call_id[:8]}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        return ToolResult(
            call_id=call_id,
            tool_name="run_algorithm_benchmark",
            agent_name=agent_name,
            status="success",
            content="Completed degree-count benchmark.",
            artifacts=[self._artifact("json", path, "Algorithm Benchmark", {"rows": rows})],
            metadata={"rows": rows},
        )
