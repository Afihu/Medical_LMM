# Pipeline Summary

This pipeline describes a Retrieval-Augmented Generation (RAG) system for medical case analysis using both image and text data. Here is a quick summary of the workflow:

1. **User Input**: A new patient case (text and/or image) is submitted by the user.
2. **Embedding**: The input is processed by a CLIP encoder to generate vector embeddings for both text and image data.
3. **Similarity Search**: The backend queries the "Known Case Library" (Qdrant vector database) using the generated vectors to find the top-5 most similar historical cases.
4. **Context Assembly**: The backend collects the retrieved similar cases and combines them with the new input.
5. **LLM Query**: The combined information is sent to the Gemini API to generate a diagnosis or response.
6. **Result Delivery**: The diagnosis from Gemini, along with relevant case information, is returned to the user as output.
7. **Data Processing**: Historical case studies are pre-processed and stored in the vector database as multi-vector points, each containing embeddings and metadata (diagnosis, symptoms, history, etc.).

This design enables efficient retrieval of relevant cases and leverages LLM capabilities for informed medical analysis.
