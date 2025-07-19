# # pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-console opentelemetry-exporter-jaeger opentelemetry-exporter-otlp
# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
# import time

# # Set up the tracer provider
# trace.set_tracer_provider(TracerProvider())

# # Create a console exporter to see traces in terminal
# console_exporter = ConsoleSpanExporter()
# span_processor = BatchSpanProcessor(console_exporter)
# trace.get_tracer_provider().add_span_processor(span_processor)

# # Get a tracer
# tracer = trace.get_tracer(__name__)

# def basic_example():
#     # Create your first span
#     with tracer.start_as_current_span("hello_world") as span:
#         print("Hello, OpenTelemetry!")
#         span.set_attribute("greeting", "hello")
#         span.add_event("Said hello")
#         time.sleep(0.1)  # Simulate some work
        
#     print("Span completed - check console output!")

# if __name__ == "__main__":
#     basic_example()


# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
# import time
# import random

# # Setup (same as above)
# trace.set_tracer_provider(TracerProvider())
# console_exporter = ConsoleSpanExporter()
# span_processor = BatchSpanProcessor(console_exporter)
# trace.get_tracer_provider().add_span_processor(span_processor)
# tracer = trace.get_tracer(__name__)

# def process_order(order_id, items):
#     """Simulate processing an order with multiple steps"""
#     with tracer.start_as_current_span("process_order") as span:
#         span.set_attribute("order.id", order_id)
#         span.set_attribute("order.item_count", len(items))
        
#         # Step 1: Validate order
#         with tracer.start_as_current_span("validate_order") as validate_span:
#             validate_span.set_attribute("validation.status", "started")
#             time.sleep(0.05)  # Simulate validation time
#             validate_span.set_attribute("validation.status", "completed")
#             validate_span.add_event("Order validated successfully")
        
#         # Step 2: Check inventory
#         with tracer.start_as_current_span("check_inventory") as inventory_span:
#             inventory_span.set_attribute("inventory.items_to_check", len(items))
#             for item in items:
#                 # Simulate checking each item
#                 available = random.choice([True, False])
#                 inventory_span.add_event(f"Checked {item}", {"available": available})
#                 time.sleep(0.02)
            
#         # Step 3: Calculate total
#         with tracer.start_as_current_span("calculate_total") as calc_span:
#             total = sum(random.randint(10, 100) for _ in items)
#             calc_span.set_attribute("order.total", total)
#             time.sleep(0.01)
        
#         span.set_attribute("order.status", "completed")
#         span.add_event("Order processing completed")
#         return total

# if __name__ == "__main__":
#     order_total = process_order("ORDER-123", ["laptop", "mouse", "keyboard"])
#     print(f"Order processed. Total: ${order_total}")


# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
# import time
# import random

# # Setup
# trace.set_tracer_provider(TracerProvider())
# console_exporter = ConsoleSpanExporter()
# span_processor = BatchSpanProcessor(console_exporter)
# trace.get_tracer_provider().add_span_processor(span_processor)
# tracer = trace.get_tracer(__name__)

# class Calculator:
#     def __init__(self):
#         self.tracer = trace.get_tracer(__name__)
    
#     def add(self, a, b):
#         with self.tracer.start_as_current_span("calculator.add") as span:
#             span.set_attribute("operation", "addition")
#             span.set_attribute("operand.a", a)
#             span.set_attribute("operand.b", b)
            
#             # Simulate some processing time
#             time.sleep(0.01)
#             result = a + b
            
#             span.set_attribute("result", result)
#             span.add_event("Addition completed")
#             return result
    
#     def multiply(self, a, b):
#         with self.tracer.start_as_current_span("calculator.multiply") as span:
#             span.set_attribute("operation", "multiplication")
#             span.set_attribute("operand.a", a)
#             span.set_attribute("operand.b", b)
            
#             time.sleep(0.02)
#             result = a * b
            
#             span.set_attribute("result", result)
#             span.add_event("Multiplication completed")
#             return result
    
#     def complex_calculation(self, x, y, z):
#         """A complex calculation that uses multiple operations"""
#         with self.tracer.start_as_current_span("calculator.complex_calculation") as span:
#             span.set_attribute("input.x", x)
#             span.set_attribute("input.y", y)
#             span.set_attribute("input.z", z)
            
#             # Step 1: Add x and y
#             step1 = self.add(x, y)
            
#             # Step 2: Multiply result by z
#             step2 = self.multiply(step1, z)
            
#             # Step 3: Add a random factor
#             with self.tracer.start_as_current_span("add_random_factor") as random_span:
#                 random_factor = random.randint(1, 10)
#                 random_span.set_attribute("random_factor", random_factor)
#                 final_result = self.add(step2, random_factor)
            
#             span.set_attribute("final_result", final_result)
#             span.add_event("Complex calculation completed")
#             return final_result

# if __name__ == "__main__":
#     calc = Calculator()
#     result = calc.complex_calculation(5, 3, 2)
#     print(f"Complex calculation result: {result}")


# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
# from opentelemetry.trace import Status, StatusCode
# import time
# import random

# # Setup
# trace.set_tracer_provider(TracerProvider())
# console_exporter = ConsoleSpanExporter()
# span_processor = BatchSpanProcessor(console_exporter)
# trace.get_tracer_provider().add_span_processor(span_processor)
# tracer = trace.get_tracer(__name__)

# def unreliable_service(item_id):
#     """Simulate a service that sometimes fails"""
#     with tracer.start_as_current_span("unreliable_service") as span:
#         span.set_attribute("item.id", item_id)
#         span.add_event("Service started")
        
#         # Simulate processing
#         time.sleep(0.1)
        
#         # Randomly succeed or fail
#         if random.choice([True, False]):
#             span.add_event("Processing completed successfully")
#             span.set_status(Status(StatusCode.OK))
#             return f"Processed {item_id}"
#         else:
#             span.add_event("Processing failed", {"error.type": "random_failure"})
#             span.set_status(Status(StatusCode.ERROR, "Random failure occurred"))
#             raise Exception(f"Failed to process {item_id}")

# def batch_processor(items):
#     """Process multiple items, handling failures gracefully"""
#     with tracer.start_as_current_span("batch_processor") as span:
#         span.set_attribute("batch.size", len(items))
#         span.add_event("Batch processing started")
        
#         successful = 0
#         failed = 0
        
#         for item in items:
#             try:
#                 with tracer.start_as_current_span(f"process_item_{item}") as item_span:
#                     item_span.set_attribute("item.id", item)
#                     result = unreliable_service(item)
#                     item_span.add_event("Item processed successfully")
#                     successful += 1
#             except Exception as e:
#                 with tracer.start_as_current_span(f"handle_error_{item}") as error_span:
#                     error_span.set_attribute("error.message", str(e))
#                     error_span.set_status(Status(StatusCode.ERROR, str(e)))
#                     error_span.add_event("Error handled")
#                     failed += 1
        
#         span.set_attribute("batch.successful", successful)
#         span.set_attribute("batch.failed", failed)
#         span.add_event("Batch processing completed", {
#             "successful": successful,
#             "failed": failed
#         })
        
#         return successful, failed

# if __name__ == "__main__":
#     items = ["item1", "item2", "item3", "item4", "item5"]
#     success_count, failure_count = batch_processor(items)
#     print(f"Batch completed: {success_count} successful, {failure_count} failed")


from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
import time

# Setup tracer provider with proper service name
resource = Resource.create({"service.name": "checkout-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))

# Console exporter - for development
console_exporter = ConsoleSpanExporter()
console_processor = BatchSpanProcessor(console_exporter)
trace.get_tracer_provider().add_span_processor(console_processor)

# OTLP exporter - for Jaeger visualization (modern approach)
try:
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317",  # gRPC endpoint
        insecure=True
    )
    otlp_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(otlp_processor)
    print("OTLP exporter configured")
except Exception as e:
    print(f"OTLP not available: {e}")

tracer = trace.get_tracer(__name__)

def multi_service_simulation():
    """Simulate multiple services working together"""
    with tracer.start_as_current_span("user_request") as span:
        span.set_attribute("user.id", "user123")
        span.set_attribute("request.type", "checkout")
        
        # Service 1: Authentication
        with tracer.start_as_current_span("auth_service") as auth_span:
            auth_span.set_attribute("service.name", "authentication")
            time.sleep(0.05)
            auth_span.add_event("User authenticated")
        
        # Service 2: Inventory check
        with tracer.start_as_current_span("inventory_service") as inv_span:
            inv_span.set_attribute("service.name", "inventory")
            time.sleep(0.1)
            inv_span.add_event("Inventory checked")
        
        # Service 3: Payment processing
        with tracer.start_as_current_span("payment_service") as pay_span:
            pay_span.set_attribute("service.name", "payment")
            pay_span.set_attribute("amount", 99.99)
            time.sleep(0.2)
            pay_span.add_event("Payment processed")
        
        # Service 4: Order confirmation
        with tracer.start_as_current_span("notification_service") as notif_span:
            notif_span.set_attribute("service.name", "notification")
            notif_span.set_attribute("notification.type", "email")
            time.sleep(0.03)
            notif_span.add_event("Confirmation sent")
        
        span.add_event("Checkout completed")

if __name__ == "__main__":
    multi_service_simulation()
    print("Multi-service simulation completed")
    
    # Give time for export
    time.sleep(1)