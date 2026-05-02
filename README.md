# 👁️ IPCI - Image Processing Chat Interface

IPCI is an AI-driven, **stateful** interface that allows users to perform complex image processing operations on both static images and live webcam feeds using natural language commands.

## 🚀 Key Features

- **Natural Language Interface:** Instead of tweaking complex parameters, simply state your goal (e.g., "Make the image slightly brighter and highlight the edges").
- **LangGraph Agent:** The core logic is driven by a smart LangGraph agent that understands the context and dynamically executes the appropriate tools via tool calling.
- **Stateful Workflow:** All operations are managed within a persistent state. This ensures the original image is preserved, operations can be easily undone, and multiple edits can be stacked sequentially.
- **Voice Control (TTS & STT):** Command the interface using your voice and receive auditory feedback from the agent.
- **Wide Range of Operations:** Full control from basic visual adjustments to complex frequency domain analysis.

## 🛠️ Supported Operations

The agent can dynamically call the following tools based on user requests:

- **Basic Adjustments:** Brightness and Contrast tuning.
- **Filtering:** Gaussian Blur and Custom Convolution filters.
- **Advanced Vision:** Edge Detection algorithms.
- **Frequency Domain:** Fourier Transform and Inverse Fourier Transform.
- **Mathematical Operations:** Image Derivatives and Integrals.

## 🧠 How It Works

1. **Input:** The system receives a spoken (STT) or written command from the user.
2. **Reasoning:** The LangGraph agent analyzes the intent, evaluating the current image state and the desired outcome.
3. **Tool Calling:** The agent triggers the specific image processing tools required to complete the task.
4. **Output:** The processed image is updated on the screen, the new state is saved for future steps, and the user receives voice feedback (TTS).

---
*This project aims to remove the technical barriers of image processing, turning it into a natural, conversational experience.*
