from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os, json

# --- Load env ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

if not api_key:
    raise ValueError("❌ Missing GEMINI_API_KEY in .env")

# --- Load your case file --- change the case number as needed to eval different cases, batch eval comes later
with open("split_cases/case_2.json", "r", encoding="utf-8") as f:
    case = json.load(f)

# --- Build sample directly from your JSON ---
sample = SingleTurnSample(
    user_input=case["question"],
    retrieved_contexts=[
        f"Title: {c.get('Title', '')}\n"
        f"History: {c.get('History', '')}\n"
        f"Findings: {c.get('Clinical_Findings', '')}\n"
        f"Discussion: {c.get('Discussion', '')}\n"
        f"Summary: {c.get('Summary_Box_First_Line', '')}"
        for c in case["retrieved_contexts"]
    ],
    response=case["answer"],
    reference=case["ground_truth"],
)

dataset = EvaluationDataset(samples=[sample])

# --- LLM & Embeddings ---
llm = ChatGoogleGenerativeAI(
    google_api_key=api_key,
    model=model_name,
    temperature=0,
    max_output_tokens=5130,
)

embeddings = HuggingFaceEmbeddings(model_name="abhinand/MedEmbed-base-v0.1")

# --- Run evaluation ---
results = evaluate(
    dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings,
)

# --- Print results cleanly ---
print("\n📊 Evaluation Results:")
print("-" * 40)

try:
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
except Exception:
    print(results)

print("-" * 40)

print(f"✅ Evaluation complete!")
