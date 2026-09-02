# Kapsl llama.cpp adapter

The current native llama.cpp backend pack will migrate here after the generic
host is proven. The published backend ABI preserves its v1 table during the
transition, and the patched allocator/request-lifecycle surface remains
explicit until equivalent upstream hooks exist.
