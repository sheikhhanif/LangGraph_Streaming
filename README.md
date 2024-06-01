# LangGraph Streaming FastAPI
This repository contains a powerful API service leveraging the capabilities of LangGraph Agent, real-time streaming tokens via Websocket, and the high-performance FastAPI framework. This project is designed to provide seamless and efficient data processing and communication for your applications.

## Key Features
- LangGraph Agent: LangGraph is a library for building stateful, multi-actor applications with LLMs. Inspired by Pregel and Apache Beam, LangGraph lets you coordinate and checkpoint multiple chains (or actors) across cyclic computational steps using regular Python functions (or JS). The public interface draws inspiration from NetworkX. The agent is created with LangGraph and has access to one tool, but you may integrate as many tools as you want.
- Streaming Tokens: Enables real-time ChatGPT-like word streaming (not token streaming; tokens are converted to words before displaying in the webUI).
- Websocket: Utilizes the Websocket protocol for real-time, bidirectional communication, ensuring low-latency data exchange.
- FastAPI: Built on FastAPI, providing a high-performance, easy-to-use web framework with automatic interactive API documentation.
