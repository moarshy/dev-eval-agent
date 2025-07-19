# client.py
import asyncio
import json
import uuid
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ------------- OTEL SETUP -------------
# Setup tracer provider with proper service name
resource = Resource.create({"service.name": "mcp-demo-client"})
trace.set_tracer_provider(TracerProvider(resource=resource))

# OTLP exporter - for Jaeger visualization (gRPC approach)
try:
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317",  # gRPC endpoint
        insecure=True
    )
    otlp_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(otlp_processor)
    print("OTLP exporter configured for Jaeger")
except Exception as e:
    print(f"OTLP not available: {e}")

tracer = trace.get_tracer(__name__)


async def main():
    # Start a root span for the whole operation
    with tracer.start_as_current_span("mcp.client.demo") as root:
        root.set_attribute("demo.id", str(uuid.uuid4()))
        root.set_attribute("client.transport", "stdio")

        # Connect to the MCP server via stdio
        server_params = StdioServerParameters(
            command="python", 
            args=["server.py"],
            env=None
        )
        
        with tracer.start_as_current_span("mcp.client.connect") as connect_span:
            connect_span.set_attribute("server.command", "python server.py")
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    with tracer.start_as_current_span("mcp.client.initialize") as init_span:
                        await session.initialize()
                        init_span.set_attribute("session.initialized", True)
                    
                    # List available tools
                    with tracer.start_as_current_span("mcp.client.list_tools") as tools_span:
                        tools_response = await session.list_tools()
                        print("Raw tools response:", tools_response)
                        
                        # Handle the tools response properly
                        if hasattr(tools_response, 'tools'):
                            tools = tools_response.tools
                            tool_names = [t.name for t in tools]
                        else:
                            # Fallback if it's a different structure
                            tools = tools_response
                            tool_names = [str(t) for t in tools]
                        
                        tools_span.set_attribute("tools.count", len(tools))
                        tools_span.set_attribute("tools.names", str(tool_names))
                        print("Available tools:", tool_names)

                    # Call the add tool
                    with tracer.start_as_current_span("mcp.client.call_tool") as call_span:
                        call_span.set_attribute("tool.name", "add")
                        call_span.set_attribute("tool.args", str({"a": 2, "b": 3}))
                        
                        resp = await session.call_tool("add", arguments={"a": 2, "b": 3})
                        call_span.set_attribute("tool.response", str(resp))
                        print("Result:", resp)


if __name__ == "__main__":
    import time
    asyncio.run(main())
    # Give time for traces to be exported before process ends
    time.sleep(0.5)
    print("Demo completed! Check Jaeger UI at http://localhost:16686")