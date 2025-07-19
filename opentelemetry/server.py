# server.py
import asyncio
import sys
from mcp.server import Server
from mcp.types import Tool, TextContent
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# ------------- OTEL SETUP -------------
# Setup tracer provider with proper service name
resource = Resource.create({"service.name": "mcp-demo-server"})
trace.set_tracer_provider(TracerProvider(resource=resource))

# OTLP exporter - for Jaeger visualization (gRPC approach)
try:
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317",  # gRPC endpoint
        insecure=True
    )
    otlp_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(otlp_processor)
    print("OTLP exporter configured for Jaeger", file=sys.stderr)
except Exception as e:
    print(f"OTLP not available: {e}", file=sys.stderr)

tracer = trace.get_tracer(__name__)

# ------------- MCP SERVER -------------
server = Server("demo")


@server.list_tools()
async def list_tools():
    return [Tool(name="add", description="Add two ints", inputSchema={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    })]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    with tracer.start_as_current_span("mcp.tool.add") as span:
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.args", str(arguments))
        
        if name == "add":
            result = arguments["a"] + arguments["b"]
            span.set_attribute("tool.result", result)
            return [TextContent(type="text", text=str(result))]
        else:
            span.set_attribute("error", f"Unknown tool: {name}")
            raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main function to run the MCP server."""
    import sys
    from mcp.server.stdio import stdio_server
    
    with tracer.start_as_current_span("mcp.server.startup") as span:
        span.set_attribute("server.name", "demo")
        print("Starting OpenTelemetry-enabled MCP Server...", file=sys.stderr)
        print("Server will be available for MCP client connections", file=sys.stderr)
        print("OpenTelemetry traces will be sent to: http://localhost:4317 (gRPC)", file=sys.stderr)
        
        # Run the server using stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, 
                write_stream, 
                server.create_initialization_options()
            )


if __name__ == "__main__":
    asyncio.run(main())


